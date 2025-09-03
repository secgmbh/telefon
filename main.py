from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse
from openai import OpenAI
import pandas as pd
import requests
import tempfile
import os

# Initialisiere Flask
app = Flask(__name__)

# Lade CSV-Datei mit Produktdaten
df = pd.read_csv("verknuepfte_tabelle_final_bereinigt.csv", sep=";")

# Initialisiere OpenAI-Client mit API-Key aus Umgebungsvariable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transkribiere_audio(audio_url):
    """Lädt Twilio-Audio herunter, speichert es temporär und transkribiert via Whisper"""
    audio_file = requests.get(audio_url + ".wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.content)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="text")

    os.remove(tmp_path)
    return transcript.strip()

def frage_gpt_mit_csv(frage, df):
    """Formuliert eine GPT-Anfrage mit eingebetteter CSV als Kontext"""
    produkt_info = df.to_string(index=False)
    prompt = f"""Du bist ein Kundenberater für ein E-Commerce-Unternehmen.
Hier sind die Produktdaten:

{produkt_info}

Der Kunde hat gefragt: "{frage}"

Gib eine präzise und hilfreiche Antwort auf Deutsch basierend auf den Daten."""

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content.strip()

@app.route("/telefon", methods=["POST"])
def telefon():
    """Antwortet auf den initialen Twilio-Anruf mit einer Sprachnachricht und Aufnahme"""
    response = VoiceResponse()
    response.say("Willkommen beim KI-Telefonassistenten. Bitte stellen Sie Ihre Frage nach dem Piepton.", language="de-DE")
    response.record(
        action="https://telefon-qb6i.onrender.com/antwort",
        method="POST",
        maxLength=10
    )
    return str(response)

@app.route("/antwort", methods=["POST"])
def antwort():
    """Verarbeitet die aufgenommene Sprache, transkribiert sie und gibt GPT-Antwort"""
    recording_url = request.form.get("RecordingUrl")
    if not recording_url:
        return "Fehlende Aufnahme", 400

    frage = transkribiere_audio(recording_url)
    antwort = frage_gpt_mit_csv(frage, df)

    response = VoiceResponse()
    response.say(antwort, language="de-DE")
    return str(response)

# Lokaler Start (nur für Debug-Zwecke)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
