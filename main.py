import os
import requests
from flask import Flask, request, Response
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ==== Konfiguration aus .env ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
PORT = int(os.getenv("PORT", 10000))

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "OK"

# 1) Anruf-Start: Begrüßung + Aufnahme
@app.route("/telefon", methods=["POST"])
def telefon():
    response = """<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say language="de-DE">Willkommen bei wowona. Mein Name ist Maria. Wie kann ich Dir helfen?</Say>
        <Record action="/antwort" method="POST" maxLength="20" timeout="2" finishOnKey="#" trim="trim-silence" playBeep="false" transcribe="false" />
    </Response>"""
    return Response(response, mimetype="text/xml")

# 2) Nach der Aufnahme: WAV holen → OpenAI transkribieren → direkte KI-Antwort
@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        recording_url = request.form.get("RecordingUrl")
        if not recording_url:
            return twilio_response("Es wurde keine Aufnahme übermittelt.")

        # --- Schneller: WAV statt MP3 ---
        wav_url = recording_url + ".wav"

        # Schneller Mini-Retry, falls Twilio gerade noch speichert
        import time
        content = None
        for _ in range(6):  # bis ~1.2s bei 0.2s Sleep
            r = requests.get(wav_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=10)
            if r.status_code == 200 and int(r.headers.get("Content-Length", 0)) > 0:
                content = r.content
                break
            time.sleep(0.2)
        if content is None:
            r = requests.get(wav_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=10)
            r.raise_for_status()
            content = r.content

        print("WAV Bytes:", len(content))

        # --- Transkription ohne Dateisystem ---
        import io
        file_like = io.BytesIO(content)
        file_like.name = "recording.wav"

        tr = client.audio.transcriptions.create(
            model=OPENAI_TRANSCRIBE_MODEL,
            file=file_like,
            language="de",
        )
        transcript = getattr(tr, "text", None) or (tr.get("text") if isinstance(tr, dict) else "")
        print("Transkript:", transcript)

        # --- Direkte Antwort ---
        reply_style = os.getenv("REPLY_STYLE", "du")  # "du" | "sie"
        tone = "freundlich, klar, lösungsorientiert"
        addr = "du" if reply_style.lower() == "du" else "Sie"
        prompt = (
            "Antworte direkt auf das Anliegen des Anrufers in natürlichem Deutsch (Anrede: {addr}). "
            "Ziel: eine hilfreiche, konkrete Antwort mit ggf. 1–2 gezielten Rückfragen, keine Zusammenfassung. "
            "Sei {tone}. Wenn Informationen fehlen, frage präzise nach. Halte dich an 1–3 Sätze, außer es werden konkrete Schritte verlangt.

"
            "Gesagter Inhalt (Roh-Transkript): {transcript}"
        ).format(addr=addr, tone=tone, transcript=transcript). ".format(addr)
          + "Ziel: eine hilfreiche, konkrete Antwort mit ggf. 1-2 gezielten Rückfragen, keine Zusammenfassung. "
          + "Sei {}. Wenn Informationen fehlen, frage präzise nach. Halte dich an 1-3 Sätze, außer es werden konkrete Schritte verlangt.\n\n".format(tone)
          + "Gesagter Inhalt (Roh-Transkript): {}".format(transcript))
