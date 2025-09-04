import os
from flask import Flask, request, send_file
from twilio.twiml.voice_response import VoiceResponse
import openai
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")

@app.route("/", methods=["GET"])
def index():
    return "Telefon-Assistent läuft."

@app.route("/telefon", methods=["POST"])
def telefon():
    response = VoiceResponse()
    response.say("Willkommen bei wowona. Mein Name ist Maria. Wie kann ich Dir helfen?", language="de-DE")
    response.record(action="/antwort", max_length=10, play_beep=False, transcribe=False)
    return str(response)

@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        recording_url = request.form["RecordingUrl"]
        print(f"Recording URL: {recording_url}")  # Debug

        audio_response = requests.get(f"{recording_url}.wav", auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        audio_response.raise_for_status()

        with open("aufnahme.wav", "wb") as f:
            f.write(audio_response.content)
        print("Audio erfolgreich heruntergeladen.")

        audio = AudioSegment.from_wav("aufnahme.wav")
        audio.export("aufnahme.mp3", format="mp3")
        print("Audio erfolgreich konvertiert.")

        with open("aufnahme.mp3", "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)
        print(f"Transkription: {transcript['text']}")

        user_input = transcript["text"]

        # CSV-Datei laden
        df = pd.read_csv("verknuepfte_tabelle_final_bereinigt.csv", dtype=str)
        print("CSV erfolgreich geladen.")

        # Prompt generieren
        prompt = f"""Ein Kunde fragt: '{user_input}'
Suche in den folgenden Rechnungsdaten eine passende Antwort.
{df.to_string(index=False)}"""

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        antwort = completion.choices[0].message["content"]
        print(f"Antwort: {antwort}")

        return Response(f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">{antwort}</Say>
</Response>""", mimetype="text/xml")

    except Exception as e:
        print(f"Fehler bei der Verarbeitung: {e}")  # <- Hier wird der tatsächliche Fehler ausgegeben
        return Response("""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">Es ist ein Fehler aufgetreten. Bitte schaue dir die Software nochmals an. Ha.</Say>
</Response>""", mimetype="text/xml")


if __name__ == "__main__":
    app.run(debug=True)