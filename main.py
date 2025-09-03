from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
import openai
import os
import tempfile
import requests

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

conversation_history = {}

@app.route("/telefon", methods=["POST"])
def telefon():
    response = VoiceResponse()
    response.say("Willkommen bei wowona. Bitte stellen Sie Ihre Frage nach dem Piepton.", language="de-DE")
    response.record(
        action="/antwort",
        maxLength=10,
        method="POST",
        transcribe=False,
        playBeep=True
    )
    return Response(str(response), mimetype="text/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        recording_url = request.form.get("RecordingUrl")
        call_sid = request.form.get("CallSid")

        if not recording_url:
            raise ValueError("RecordingUrl fehlt")

        # 1. Download Audio
        audio_url = f"{recording_url}.mp3"
        audio_data = requests.get(audio_url).content

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        # 2. Transkription via Whisper
        with open(tmp_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)["text"]

        os.remove(tmp_path)

        # 3. GPT antwortet
        history = conversation_history.get(call_sid, [])
        history.append({"role": "user", "content": transcript})

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Du bist ein freundlicher deutschsprachiger Kundendienstassistent."}] + history,
        )
        antwort_text = completion.choices[0].message["content"]
        history.append({"role": "assistant", "content": antwort_text})
        conversation_history[call_sid] = history

        # 4. Antwort zurück an Anrufer
        response = VoiceResponse()
        response.say(antwort_text, language="de-DE")
        response.pause(length=1)
        response.say("Möchten Sie noch etwas wissen?", language="de-DE")
        response.record(
            action="/antwort",
            maxLength=10,
            method="POST",
            transcribe=False,
            playBeep=True
        )
        return Response(str(response), mimetype="text/xml")

    except Exception as e:
        print("Fehler in /antwort:", str(e))
        response = VoiceResponse()
        response.say("Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut.", language="de-DE")
        return Response(str(response), mimetype="text/xml")
