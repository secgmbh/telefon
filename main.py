import os
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
import requests
from openai import OpenAI
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/telefon", methods=["POST"])
def telefon():
    response = VoiceResponse()
    response.say("Willkommen bei wowona. Mein Name ist Maria. Wie kann ich Dir helfen?", language="de-DE")
    response.record(
        action="/antwort",
        method="POST",
        max_length=10,
        play_beep=False,
        transcribe=False
    )
    return Response(str(response), mimetype="text/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    response = VoiceResponse()
    recording_url = request.form.get("RecordingUrl")

    try:
        # Lade MP3-Datei herunter
        twilio_sid = os.getenv("TWILIO_SID")
        twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        audio_response = requests.get(recording_url + ".mp3", auth=(twilio_sid, twilio_token))
        with open("recording.mp3", "wb") as f:
            f.write(audio_response.content)

        # Konvertiere in WAV
        sound = AudioSegment.from_mp3("recording.mp3")
        sound.export("recording.wav", format="wav")

        # Sende an Whisper
        with open("recording.wav", "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            user_input = transcription.text

        prompt = f"Ein Kunde fragt: '{user_input}'. Bitte antworte höflich und hilfreich auf Deutsch."
        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        antwort = chat_response.choices[0].message.content

        response.say(antwort, language="de-DE")
    except Exception as e:
        print("Fehler:", e)
        response.say("Es ist ein Fehler aufgetreten. Bitte schaue dir die Software nochmals an. Ha ha ha ha.", language="de-DE")

    return Response(str(response), mimetype="text/xml")

if __name__ == "__main__":
    app.run(debug=False, port=5000)