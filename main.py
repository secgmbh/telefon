
import os
import requests
import pandas as pd
from flask import Flask, request, Response
from dotenv import load_dotenv
import openai

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

df = pd.read_csv("verknuepfte_tabelle_final_bereinigt.csv")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "Telefon-Service läuft!"

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
    data = request.form
    recording_url = data.get("RecordingUrl")

    try:
        audio_response = requests.get(recording_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        audio_response.raise_for_status()

        with open("audio.mp3", "wb") as f:
            f.write(audio_response.content)

        with open("audio.mp3", "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)

        user_input = transcript["text"]

        prompt = f"""Ein Kunde fragt: '{user_input}'
Suche in den folgenden Rechnungsdaten eine passende Antwort.
Hier ist die Tabelle:
{df.head(20).to_string(index=False)}
"""

        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        antworttext = gpt_response.choices[0].message.content.strip()

        response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">{antworttext}</Say>
</Response>"""
        return Response(response, mimetype="text/xml")

    except Exception as e:
        print(f"Fehler bei der Verarbeitung: {e}")
        return Response("""<?xml version="1.0" encoding="UTF-8"?><Response><Say language="de-DE">Es ist ein Fehler aufgetreten. Probiere es nochmals. he he he</Say></Response>""", mimetype="text/xml")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
