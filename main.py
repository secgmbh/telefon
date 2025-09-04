
import os
import requests
from flask import Flask, request, Response
from dotenv import load_dotenv

# Optional: falls du .env lokal nutzt
load_dotenv()

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "OK"

@app.route("/telefon", methods=["POST"])
def telefon():
    response = """<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say language="de-DE">Willkommen bei wowona. Mein Name ist Maria. Wie kann ich Dir helfen?</Say>
        <Record action="/antwort" maxLength="10" playBeep="false" transcribe="false" />
    </Response>"""
    return Response(response, mimetype="text/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        # Direkt eingebunden wie gewünscht
        recording_url = request.form['RecordingUrl'] + '.mp3'

        sid = "AC1ab8ebc060f9c4350fd8e43cfc2438be"
        token = "1a08c84357f1680778335ee0a12bb7ed"

        audio_response = requests.get(recording_url, auth=(sid, token))

        # Status prüfen
        print("Status Code:", audio_response.status_code)
        audio_response.raise_for_status()

        with open("recording.mp3", "wb") as f:
            f.write(audio_response.content)

        print(f"Aufnahme gespeichert als recording.mp3 ({len(audio_response.content)} Bytes)")

        return twilio_response("Vielen Dank für Ihre Nachricht. Wir melden uns bald bei Ihnen.")

    except Exception as e:
        print(f"Fehler bei der Verarbeitung: {e}")
        return twilio_error_response("Es ist ein Fehler aufgetreten.")

def twilio_response(text):
    return Response(f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say language="de-DE">{text}</Say>
    </Response>""", mimetype="text/xml")

def twilio_error_response(text):
    return twilio_response(text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
