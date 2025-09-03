import os
import openai
import pandas as pd
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)

# Lade CSV-Datei
df = pd.read_csv("verknuepfte_tabelle_final_bereinigt.csv", sep=";")

# API-Key von Umgebungsvariable laden
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/telefon", methods=["POST"])
def telefon():
    response = VoiceResponse()
    response.say("Willkommen bei wowona. Bitte stellen Sie Ihre Frage nach dem Piepton.", language="de-DE")
    response.record(
        action="/antwort",
        method="POST",
        maxLength="10",
        playBeep=True,
        transcribe=False
    )
    return Response(str(response), mimetype="application/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        recording_url = request.form["RecordingUrl"]
        audio_url = recording_url + ".wav"
        audio_file_path = "/tmp/audio.wav"

        import requests
        with open(audio_file_path, "wb") as f:
            f.write(requests.get(audio_url).content)

        with open(audio_file_path, "rb") as audio_file:
            transcription = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        frage = transcription.text

        # Suche in der CSV-Datei
        antwort = "Ich konnte leider keine passende Information finden."
        for _, row in df.iterrows():
            if str(row["Name"]).lower() in frage.lower():
                antwort = f"{row['Name']} hat den Status: {row['Status']}, letzte Bestellung: {row['Bestellung']}."
                break

        response = VoiceResponse()
        response.say(antwort, language="de-DE")
        return Response(str(response), mimetype="application/xml")

    except Exception as e:
        print(f"Fehler: {e}")
        response = VoiceResponse()
        response.say("Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut.", language="de-DE")
        return Response(str(response), mimetype="application/xml")

@app.route("/", methods=["GET"])
def index():
    return "KI-Telefonassistent läuft."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)