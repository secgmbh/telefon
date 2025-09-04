import os
from flask import Flask, request, send_file
from twilio.twiml.voice_response import VoiceResponse
import openai
from pydub import AudioSegment
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
    response = VoiceResponse()
    try:
        recording_url = request.form["RecordingUrl"] + ".wav"
        recording_sid = request.form["RecordingSid"]
        audio_file_path = f"{recording_sid}.wav"

        r = requests.get(recording_url, auth=(twilio_sid, twilio_auth_token))
        with open(audio_file_path, "wb") as f:
            f.write(r.content)

        audio = AudioSegment.from_file(audio_file_path)
        audio.export(audio_file_path, format="wav")

        with open(audio_file_path, "rb") as audio_file:
            transcript_response = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
            user_input = transcript_response.text

        os.remove(audio_file_path)

        df = pd.read_csv("verknuepfte_tabelle_final_bereinigt.csv")
        data_string = df.to_string(index=False)
        
        prompt = f"""Ein Kunde fragt: '{user_input}'.
Suche in den folgenden Rechnungsdaten eine passende Antwort.
{data_string}"""

        gpt_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Telefonassistent."},
                {"role": "user", "content": prompt}
            ]
        )

        antwort_text = gpt_response.choices[0].message.content.strip()
        response.say(antwort_text, language="de-DE")
    except Exception as e:
        response.say("Es ist ein Fehler aufgetreten. Bitte schaue dir die Software nochmals an. Ha ha ha ha.", language="de-DE")
    return str(response)

if __name__ == "__main__":
    app.run(debug=True)