import os
import io
import time
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
REPLY_STYLE = os.getenv("REPLY_STYLE", "du")  # "du" | "sie"
VOICE = os.getenv("TWILIO_VOICE", "Polly.Vicki-Neural")  # professionelle Neural-Stimme

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

# ------- Hilfsfunktionen -------

def twiml_say(text: str, voice: str = VOICE) -> str:
    return '<Say language="de-DE" voice="{voice}">{text}</Say>'.format(
        voice=voice, text=escape_xml(text)
    )

def twiml_record(action: str = "/antwort", max_len: str = "15", timeout: str = "1", beep: str = "false") -> str:
    # Kein finishOnKey mehr, reine Stille-/Längensteuerung
    return (
        '<Record action="{action}" method="POST" maxLength="{ml}" timeout="{to}" '
        'trim="trim-silence" playBeep="{beep}" transcribe="false" />'
    ).format(action=action, ml=max_len, to=timeout, beep=beep)

def twiml_continue(bot_text: str) -> Response:
    # Antwort → kurze Pause → neue Aufnahme
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        + twiml_say(bot_text)
        + '<Pause length="0.2"/>'
        + twiml_record()
        + "</Response>"
    )
    return Response(xml, mimetype="text/xml")

def twiml_end(final_text: str) -> Response:
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        + twiml_say(final_text)
        + "</Response>"
    )
    return Response(xml, mimetype="text/xml")

def escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

def extract_text(response_obj) -> str:
    try:
        if hasattr(response_obj, "output_text"):
            return response_obj.output_text
        if hasattr(response_obj, "output"):
            parts = []
            for item in response_obj.output:
                if isinstance(item, dict) and item.get("type") == "output_text":
                    parts.append(item.get("text", ""))
            return "\n".join([p for p in parts if p])
        if isinstance(response_obj, dict):
            return response_obj.get("output_text") or ""
    except Exception:
        pass
    return ""

# ------- Routen -------

@app.route("/", methods=["GET"])
def index():
    return "OK"

@app.route("/telefon", methods=["POST"])
def telefon():
    # Einstieg: Begrüßung + erste Aufnahme (kurze Runden für geringe Latenz)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        + twiml_say("Willkommen bei wowona. Mein Name ist Petra. Wie kann ich dir helfen?")
        + twiml_record()  # maxLength=15s, timeout=1s Stille
        + "</Response>"
    )
    return Response(xml, mimetype="text/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        recording_url = request.form.get("RecordingUrl")
        if not recording_url:
            # Kein Audio erhalten → direkt neue Runde starten
            return twiml_continue("Ich habe dich leider nicht verstanden. Kannst du dein Anliegen bitte noch einmal kurz schildern?")

        # --- schnelle Datei: WAV statt MP3 ---
        wav_url = recording_url + ".wav"

        # Sehr kurzer Retry, um auf Twilios Speicherung zu warten (min. Latenz)
        content = None
        for _ in range(3):  # ~0.3 s
            r = requests.get(wav_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=5)
            if r.status_code == 200 and int(r.headers.get("Content-Length", 0)) > 0:
                content = r.content
                break
            time.sleep(0.1)
        if content is None:
            r = requests.get(wav_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=5)
            r.raise_for_status()
            content = r.content

        # Transkription (In-Memory)
        file_like = io.BytesIO(content)
        file_like.name = "recording.wav"
        tr = client.audio.transcriptions.create(
            model=OPENAI_TRANSCRIBE_MODEL,
            file=file_like,
            language="de",
        )
        transcript = getattr(tr, "text", None) or (tr.get("text") if isinstance(tr, dict) else "")

        # Direkte Antwort (kurz, konkret, ggf. 1 Rückfrage)
        addr = "du" if REPLY_STYLE.lower() == "du" else "Sie"
        tone = "freundlich, klar, lösungsorientiert"
        prompt = (
            "Antworte direkt auf das Anliegen des Anrufers in natürlichem Deutsch (Anrede: {addr}). "
            "Ziel: eine hilfreiche, konkrete Antwort mit ggf. 1 gezielten Rückfrage, keine Zusammenfassung. "
            "Sei {tone}. Wenn Informationen fehlen, frage präzise nach. Halte dich an 1-3 Sätze.\n\n"
            "Gesagter Inhalt (Roh-Transkript): {transcript}"
        ).format(addr=addr, tone=tone, transcript=transcript)

        resp = client.responses.create(model="gpt-4o-mini", input=[{"role": "user", "content": prompt}])
        bot_text = extract_text(resp) or "Kannst du dazu noch einen Punkt präzisieren?"

        # Keine Abbruchbedingung – wir führen das Gespräch fort
        return twiml_continue(bot_text)

    except Exception as e:
        print("Fehler bei der Verarbeitung:", e)
        return twiml_end("Es ist ein Fehler aufgetreten bei der Verarbeitung. Vielen Dank für deinen Anruf.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
