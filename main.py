
import os
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
import openai
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

app = Flask(__name__)

# CSV-Datei laden
csv_path = "verknuepfte_tabelle_final_bereinigt.csv"
df = pd.read_csv(csv_path, dtype=str, sep=";", encoding="utf-8")

@app.route("/")
def index():
    return "KI-Telefonassistent läuft!"

@app.route("/telefon", methods=["POST"])
def telefon():
    response = VoiceResponse()
    response.say("Willkommen bei wowona. Mein Name ist Maria. Wie kann ich Dir helfen?", language="de-DE")
    response.record(action="/antwort", max_length=10, method="POST", play_beep="false", transcribe="false")
    return Response(str(response), mimetype="text/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        recording_url = request.form["RecordingUrl"] + ".wav"

        # Audio-Datei herunterladen
        audio_response = requests.get(
            recording_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        )

        with open("aufnahme.wav", "wb") as f:
            f.write(audio_response.content)

        # Transkription
        with open("aufnahme.wav", "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )

        user_input = transcript.strip()
        prompt = f"Ein Kunde fragt: '{user_input}'. Nutze die CSV-Daten, um eine sinnvolle Antwort zu geben."

        # GPT-Antwort erzeugen
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )

        gpt_reply = response.choices[0].message.content.strip()

        twiml_response = VoiceResponse()
        twiml_response.say(gpt_reply, language="de-DE")
        return Response(str(twiml_response), mimetype="text/xml")

    except Exception as e:
        print("Fehler:", e)
        response = VoiceResponse()
        response.say("Es ist ein Fehler aufgetreten. Bitte schaue dir die Software nochmals an. Ha ha ha.", language="de-DE")
        return Response(str(response), mimetype="text/xml")
