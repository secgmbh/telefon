
import os
from flask import Flask, request, Response
import requests
from dotenv import load_dotenv

# Lade Umgebungsvariablen aus .env-Datei
load_dotenv()

# Flask App
app = Flask(__name__)

# Twilio Credentials aus Umgebungsvariablen
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# Home Route
@app.route("/", methods=["GET"])
def index():
    return "Telefon-Assistent läuft."

# Begrüßung und Aufnahme starten
@app.route("/telefon", methods=["POST"])
def telefon():
    response = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">Willkommen bei wowona. Mein Name ist Maria. Wie kann ich Dir helfen?</Say>
    <Record action="/antwort" maxLength="10" playBeep="false" transcribe="false" />
</Response>'''
    return Response(response, mimetype="application/xml")

# Aufnahme entgegennehmen und Antwort generieren
@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        recording_url = request.form.get("RecordingUrl")
        if not recording_url:
            raise ValueError("Keine Aufnahme-URL erhalten.")

        print("Recording URL:", recording_url)

        # Lade Audio-Datei im unterstützten Format (z. B. .mp3)
        audio_response = requests.get(f"{recording_url}.mp3", auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        audio_response.raise_for_status()

        # Speichere temporär
        with open("aufnahme.mp3", "wb") as f:
            f.write(audio_response.content)

        # Hier kannst du die Datei an OpenAI senden oder verarbeiten
        # z. B. mit speech-to-text oder einer Antwort erzeugen

        antwort = "Danke für deine Nachricht. Wir melden uns bei dir."
    except Exception as e:
        print("Fehler bei der Verarbeitung:", str(e))
        antwort = "Es ist ein Fehler aufgetreten. Bitte versuche es später erneut."

    response = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">{antwort}</Say>
</Response>'''
    return Response(response, mimetype="application/xml")

# App exportieren
app = app
