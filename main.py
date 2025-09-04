import os
from flask import Flask, request, Response
from dotenv import load_dotenv
import openai
import requests
from pydub import AudioSegment
from io import BytesIO

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

@app.route("/")
def home():
    return "Telefon-Assistent läuft"

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
        recording_url = request.form.get("RecordingUrl")
        print("Recording URL:", recording_url)

        if not recording_url:
            raise ValueError("Keine Aufnahme-URL erhalten.")

        audio_response = requests.get(f"{recording_url}.wav", auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))

        if audio_response.status_code != 200:
            raise ValueError(f"Fehler beim Abrufen der Audioaufnahme: {audio_response.status_code} - {audio_response.text}")

        audio = AudioSegment.from_file(BytesIO(audio_response.content), format="wav")
        mp3_io = BytesIO()
        audio.export(mp3_io, format="mp3")
        mp3_io.seek(0)

        try:
            transcript = openai.Audio.transcribe("whisper-1", mp3_io)
            antwort = transcript.get("text", "").strip()
            if not antwort:
                raise ValueError("Keine Transkription erhalten.")
        except Exception as e:
            print(f"Transkriptionsfehler: {e}")
            return Response("""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">Leider konnte ich dich nicht verstehen. Bitte wiederhole deine Frage.</Say>
</Response>""", mimetype="text/xml")

        return Response(f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">{antwort}</Say>
</Response>""", mimetype="text/xml")

    except Exception as e:
        print(f"Fehler bei der Verarbeitung: {e}")
        return Response("""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">Es ist ein Fehler aufgetreten. Bitte schaue dir die Software nochmals an. ho.</Say>
</Response>""", mimetype="text/xml")