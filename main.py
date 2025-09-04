import os
import requests
import pandas as pd
from flask import Flask, request, Response
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# API-Schlüssel aus Umgebungsvariablen laden
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

app = Flask(__name__)

# CSV-Datei laden
CSV_PATH = "verknuepfte_tabelle_final_bereinigt.csv"
df = pd.read_csv(CSV_PATH, dtype=str)

# GPT vorbereiten
client = OpenAI(api_key=OPENAI_API_KEY)

@app.route("/", methods=["GET"])
def index():
    return "Server läuft."

@app.route("/telefon", methods=["POST"])
def telefon():
    return Response("""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">Willkommen bei wowona. Mein Name ist Maria. Wie kann ich Dir helfen?</Say>
    <Record action="/antwort" maxLength="10" playBeep="false" transcribe="false" />
</Response>""", mimetype="text/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        recording_url = request.form["RecordingUrl"]
        audio_response = requests.get(recording_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        with open("aufnahme.wav", "wb") as f:
            f.write(audio_response.content)

        # Transkription (vereinfacht)
        audio_file = open("aufnahme.wav", "rb")
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="de"
        )
        user_input = transcript.text

        # Datenabfrage vorbereiten
        prompt = f"""Ein Kunde fragt: '{user_input}'
Suche in den folgenden Rechnungsdaten eine passende Antwort.
Wenn möglich, nenne den Betrag, die Bestellnummer, oder den Namen.
CSV-Daten:
{df.head(30).to_string(index=False)}"""

        gpt_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Kundendienst-Assistent."},
                {"role": "user", "content": prompt}
            ]
        )
        antwort = gpt_response.choices[0].message.content

        return Response(f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">{antwort}</Say>
</Response>""", mimetype="text/xml")

    except Exception as e:
        print("Fehler bei der Verarbeitung:", e)
        return Response("""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut.</Say>
</Response>""", mimetype="text/xml")