import os
import requests
import pandas as pd
from flask import Flask, request, Response
from dotenv import load_dotenv

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

@app.route("/")
def index():
    return "Telefon-API läuft"

@app.route("/telefon", methods=["POST"])
def telefon():
    response = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">Willkommen bei wowona. Mein Name ist Maria. Wie kann ich Dir helfen?</Say>
    <Record action="/antwort" maxLength="10" playBeep="false" transcribe="false" />
</Response>'''
    return Response(response, mimetype="text/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        recording_url = request.form.get("RecordingUrl")
        print("Recording URL:", recording_url)

        audio_response = requests.get(f"{recording_url}.wav", auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))

        if audio_response.status_code != 200:
            raise Exception(f"Error code: {audio_response.status_code} - {audio_response.json()}")

        with open("aufnahme.wav", "wb") as f:
            f.write(audio_response.content)

        with open("aufnahme.wav", "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
                language="de"
            )

        user_input = transcript.strip()

        df = pd.read_csv("verknuepfte_tabelle_final_bereinigt.csv", sep=";")
        prompt = f"Ein Kunde fragt: '{user_input}'\nSuche in den folgenden Rechnungsdaten eine passende Antwort."

        prompt += "\n\n"
        for index, row in df.iterrows():
            prompt += f"Rechnungsnummer: {row['rechnungsnummer']}, Betrag: {row['betrag']}, Firma: {row['firma']}, Produkt: {row['produktbezeichnung']}\n"

        prompt += "\nAntworte sehr knapp und nur auf Basis dieser Daten."

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        antwort = completion.choices[0].message.content

        response = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">{antwort}</Say>
</Response>'''
        return Response(response, mimetype="text/xml")

    except Exception as e:
        print("Fehler bei der Verarbeitung:", str(e))
        fehler_antwort = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">Es ist ein Fehler aufgetreten. Bitte schaue dir die Software nochmals an. ho.</Say>
</Response>'''
        return Response(fehler_antwort, mimetype="text/xml")