import os
import openai
import requests
import csv
import tempfile
import subprocess
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
csv_file = "verknuepfte_tabelle_final_bereinigt.csv"

def get_data_for_invoice(rechnung):
    try:
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Rechnungsnummer") == rechnung:
                    return row
    except Exception as e:
        print(f"Fehler beim Lesen der CSV: {e}")
    return None

@app.route("/telefon", methods=["POST"])
def telefon():
    response = VoiceResponse()
    response.say("Willkommen bei wowona. Mein Name ist Maria. Wie kann ich Dir helfen?", language="de-DE")
    response.record(
        action="/antwort",
        maxLength=10,
        method="POST",
        playBeep="false",
        transcribe="false"
    )
    return Response(str(response), mimetype="application/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        recording_url = request.form["RecordingUrl"]
        wav_file_url = recording_url + ".mp3"

        # Audio herunterladen
        audio_response = requests.get(wav_file_url)
        if audio_response.status_code != 200:
            raise Exception(f"Download fehlgeschlagen: {audio_response.status_code}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_file:
            mp3_file.write(audio_response.content)
            mp3_path = mp3_file.name

        wav_path = mp3_path.replace(".mp3", ".wav")
        subprocess.run(["ffmpeg", "-y", "-i", mp3_path, wav_path], check=True)

        with open(wav_path, "rb") as audio_file:
            transcript_response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        user_input = transcript_response.text.strip()
        print("User Input:", user_input)

        prompt = f"Ein Kunde fragt: '{user_input}'"
Suche in den folgenden Rechnungsdaten eine passende Antwort."

        if "rechnung" in user_input.lower():
            for word in user_input.split():
                if word.isdigit():
                    data = get_data_for_invoice(word)
                    if data:
                        prompt += f"
Rechnungsnummer: {word}
"
                        for key, value in data.items():
                            prompt += f"{key}: {value}
"
                        break

        chat_completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein freundlicher Kundenservice Assistent."},
                {"role": "user", "content": prompt}
            ]
        )

        antwort_text = chat_completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"Fehler: {e}")
        antwort_text = "Es ist ein Fehler aufgetreten. Bitte schaue dir die Software nochmals an. Ha ha ha ha."

    response = VoiceResponse()
    response.say(antwort_text, language="de-DE")
    return Response(str(response), mimetype="application/xml")

if __name__ == "__main__":
    app.run(debug=False)
