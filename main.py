from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse
import openai
import requests
from io import BytesIO
import os

app = Flask(__name__)

# OpenAI API-Key aus Umgebungsvariable lesen
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/", methods=["GET"])
def index():
    return "KI-Telefonassistent ist aktiv."

@app.route("/telefon", methods=["POST"])
def telefon():
    response = VoiceResponse()
    response.say("Willkommen bei wowona. Bitte stellen Sie Ihre Frage nach dem Piepton.", language="de-DE")
    response.record(
        action="/antwort",
        maxLength=10,
        method="POST",
        playBeep=True,
        transcribe=False
    )
    return str(response)

@app.route("/antwort", methods=["POST"])
def antwort():
    recording_url = request.form.get("RecordingUrl")
    if not recording_url:
        return "<Response><Say>Keine Aufnahme empfangen.</Say></Response>"

    try:
        # Audio von Twilio herunterladen
        audio_response = requests.get(recording_url + ".wav")
        audio_response.raise_for_status()
        audio_file = BytesIO(audio_response.content)

        # Whisper Transkription
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

        # GPT-Antwort generieren
        chat_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": transcript}]
        )
        antwort_text = chat_response.choices[0].message.content.strip()

    except Exception as e:
        print("Fehler:", e)
        return "<Response><Say>Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut.</Say></Response>"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">{antwort_text}</Say>
</Response>"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
