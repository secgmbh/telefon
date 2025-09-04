import os
import requests
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
from openai import OpenAI
from openai import OpenAIError
from openai import audio

app = Flask(__name__)

# API-Schlüssel aus Umgebungsvariablen lesen
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)


@app.route("/")
def index():
    return "KI Telefonassistent läuft."


@app.route("/telefon", methods=["POST"])
def telefon():
    response = VoiceResponse()
    response.say("Willkommen bei wowona. Mein Name ist Maria. Wie kann ich Dir helfen?", language="de-DE")
    response.record(
        action="/antwort",
        max_length=10,
        method="POST",
        play_beep=True,
        transcribe=False
    )
    return Response(str(response), mimetype="text/xml")


@app.route("/antwort", methods=["POST"])
def antwort():
    recording_url = request.form.get("RecordingUrl", "") + ".wav"
    response = VoiceResponse()

    try:
        audio_response = requests.get(recording_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        audio_response.raise_for_status()

        with open("/tmp/aufnahme.wav", "wb") as f:
            f.write(audio_response.content)

        with open("/tmp/aufnahme.wav", "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )

        frage = transcription.strip()

        chat_completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein hilfsbereiter Kundenservice Assistent für das Unternehmen wowona."},
                {"role": "user", "content": frage}
            ]
        )

        antwort = chat_completion.choices[0].message.content.strip()
        response.say(antwort, language="de-DE")

    except OpenAIError as e:
        print("Fehler:", e)
        response.say("Es ist ein Fehler mit dem Sprachdienst aufgetreten. Bitte versuchen Sie es später erneut.", language="de-DE")

    except Exception as e:
        print("Fehler:", e)
        response.say("Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut.", language="de-DE")

    return Response(str(response), mimetype="text/xml")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)