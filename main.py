import os
import openai
import pandas as pd
import requests
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# OpenAI API-Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# CSV laden
csv_path = "verknuepfte_tabelle_final_bereinigt.csv"
df = pd.read_csv(csv_path)

@app.route("/")
def home():
    return "KI-Telefonassistent läuft!"

@app.route("/telefon", methods=["POST"])
def telefon():
    response = VoiceResponse()
    response.say("Willkommen bei wowona. Mein Name ist Maria. Wie kann ich Dir helfen?", language="de-DE")
    response.record(action="/antwort", maxLength=10, method="POST", playBeep=False, transcribe=False)
    return Response(str(response), mimetype="application/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        recording_url = request.form["RecordingUrl"] + ".wav"

        # Audio-Datei herunterladen
        audio_file = "aufnahme.wav"
        with requests.get(recording_url, stream=True, auth=(os.getenv("TWILIO_SID"), os.getenv("TWILIO_AUTH_TOKEN"))) as r:
            r.raise_for_status()
            with open(audio_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Transkription mit Whisper v1
        with open(audio_file, "rb") as af:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=af
            ).text

        # GPT-Frage mit CSV-Kontext beantworten
        prompt = f'''Ein Kunde fragt: "{transcript}"
Bitte beantworte die Frage höflich auf Deutsch unter Verwendung der folgenden Produktdaten:

{df.head(10).to_string(index=False)}'''

        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        antwort_text = completion.choices[0].message.content

    except Exception as e:
        print(f"Fehler: {e}")
        antwort_text = "Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."

    response = VoiceResponse()
    response.say(antwort_text, language="de-DE")
    return Response(str(response), mimetype="application/xml")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)