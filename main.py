import os
import requests
import openai
import pandas as pd
from flask import Flask, request, Response

# API-Key aus Umgebungsvariable (besser als direkt im Code)
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Hilfsfunktion: Antwort mit GPT auf Basis CSV
def gpt_antwort(transkript):
    try:
        df = pd.read_csv("produktdaten.csv")
        kontext = "\n".join([f"{row['Produktname']}: {row['Status']}" for _, row in df.iterrows()])

        system_prompt = "Du bist ein hilfreicher Kundenservice-Assistent. Nutze die folgenden Produktdaten, um Fragen zu beantworten."
        user_prompt = f"{kontext}\n\nFrage: {transkript}"

        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("Fehler bei GPT:", e)
        return "Ich konnte die Informationen leider nicht abrufen."

@app.route("/")
def index():
    return "KI-Telefonassistent läuft!"

@app.route("/telefon", methods=["POST"])
def telefon():
    response = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">Willkommen bei wowona. Mein Name ist Maria. Wie kann ich Dir helfen?</Say>
    <Record action="/antwort" maxLength="10" method="POST" playBeep="true" transcribe="false" />
</Response>'''
    return Response(response, mimetype="text/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        recording_url = request.form["RecordingUrl"]
        audio_url = f"{recording_url}.wav"

        # Authentifiziere Twilio-Anfrage
        auth = (
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )

        audio_response = requests.get(audio_url, auth=auth)
        audio_response.raise_for_status()

        with open("aufnahme.wav", "wb") as f:
            f.write(audio_response.content)

        with open("aufnahme.wav", "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            ).text

        antworttext = gpt_antwort(transcript)

        return Response(
            f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">{antworttext}</Say>
</Response>''',
            mimetype="text/xml"
        )

    except Exception as e:
        print("Fehler:", e)
        return Response(
            f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="de-DE">Fehler: {str(e)}</Say>
</Response>''',
            mimetype="text/xml"
        )

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")