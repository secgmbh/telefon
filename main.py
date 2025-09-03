import os
import requests
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
import openai

app = Flask(__name__)

# OpenAI API-Key aus Umgebungsvariable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Twilio Account SID und Auth Token aus Umgebungsvariablen
account_sid = os.getenv("AC1ab8ebc060f9c4350fd8e43cfc2438be")
auth_token = os.getenv("14e18bba76650b22a18a8958a9329a6d")

@app.route("/", methods=["GET"])
def home():
    return "KI-Telefonassistent ist aktiv."

@app.route("/telefon", methods=["POST"])
def telefon():
    response = VoiceResponse()
    response.say("Willkommen bei wowona. Bitte stellen Sie Ihre Frage nach dem Piepton.", language="de-DE")
    response.record(
        action="/antwort",
        method="POST",
        max_length=10,
        play_beep=True,
        transcribe=False
    )
    return Response(str(response), mimetype="application/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    recording_url = request.form.get("RecordingUrl")

    try:
        if not recording_url:
            raise ValueError("Keine Aufnahme-URL erhalten.")

        audio_url = recording_url + ".wav"

        # Audio mit Authentifizierung herunterladen
        response = requests.get(audio_url, auth=(account_sid, auth_token))
        response.raise_for_status()

        with open("aufnahme.wav", "wb") as f:
            f.write(response.content)

        # OpenAI Whisper Transkription
        with open("aufnahme.wav", "rb") as audio_file:
            transcript_response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="de"
            )
        transcript = transcript_response.strip()

        # GPT-Antwort generieren
        chat_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Du bist ein freundlicher Telefonassistent von wowona."},
                {"role": "user", "content": transcript}
            ]
        )

        antwort = chat_response.choices[0].message.content

    except Exception as e:
        print(f"Fehler: {e}")
        antwort = "Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."

    twilio_response = VoiceResponse()
    twilio_response.say(antwort, language="de-DE")
    return Response(str(twilio_response), mimetype="application/xml")

if __name__ == "__main__":
    app.run(debug=False, port=5000, host="0.0.0.0")
