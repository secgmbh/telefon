
import os
import openai
import pandas as pd
from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse
import requests

app = Flask(__name__)

# 🔐 OpenAI API Key aus Umgebungsvariablen
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 📊 CSV-Datei laden
CSV_PATH = "verknuepfte_tabelle_final_bereinigt.csv"
df = pd.read_csv(CSV_PATH, dtype=str)

@app.route("/", methods=["GET"])
def index():
    return "KI-Telefonassistent läuft."

@app.route("/telefon", methods=["POST"])
def telefon():
    response = VoiceResponse()
    response.say("Willkommen bei wowona. Mein Name ist Maria. Wie kann ich Dir helfen?", language="de-DE")
    response.record(action="/antwort", method="POST", max_length=10, play_beep=false, transcribe=False)
    return str(response)

@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        recording_url = request.form["RecordingUrl"] + ".mp3"
        audio_file = download_audio(recording_url)

        with open(audio_file, "rb") as f:
            transcript_response = openai.audio.transcriptions.create(model="whisper-1", file=f)
        user_input = transcript_response.text
        print("🗣 Transkribiert:", user_input)

        matching_row = find_matching_row(user_input)
        if matching_row is None:
            response_text = "Ich konnte zu dieser Anfrage leider keine Informationen finden."
        else:
            context = matching_row.to_string()
            prompt = f"Ein Kunde fragt: '{user_input}'
Hier sind die zugehörigen Daten:
{context}
Antworte auf Deutsch und kundenfreundlich."
            chat_response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = chat_response.choices[0].message.content.strip()

        response = VoiceResponse()
        response.say(response_text, language="de-DE")
        return str(response)

    except Exception as e:
        print("Fehler bei GPT:", str(e))
        response = VoiceResponse()
        response.say("Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut.", language="de-DE")
        return str(response)

def download_audio(url):
    filename = "/tmp/recording.mp3"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return filename

def find_matching_row(user_input):
    user_input = user_input.lower()
    for col in df.columns:
        if "rechnungsnummer" in col.lower() or "bestellnummer" in col.lower():
            for value in df[col].dropna().unique():
                if isinstance(value, str) and value in user_input:
                    match = df[df[col] == value]
                    if not match.empty:
                        return match.iloc[0]
    return None

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
