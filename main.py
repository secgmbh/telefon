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
MAX_TURNS = int(os.getenv("MAX_TURNS", "4"))
VOICE = os.getenv("TWILIO_VOICE", "Polly.Vicki-Neural")

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

# Simpler In-Memory-Zustand pro CallSid
CALL_STATE = {}  # { CallSid: {"turn": int} }

# ------- Hilfen -------

def twiml_say(text, voice=VOICE):
    return '<Say language="de-DE" voice="{voice}">{text}</Say>'.format(
        voice=voice, text=escape_xml(text)
    )

def twiml_record(action="/antwort", max_len="20", timeout="2", beep="false"):
    # finishOnKey="#" => User kann mit Raute sofort abschließen
    return (
        '<Record action="{action}" method="POST" maxLength="{ml}" timeout="{to}" '
        'finishOnKey="#" trim="trim-silence" playBeep="{beep}" transcribe="false" />'
    ).format(action=action, ml=max_len, to=timeout, beep=beep)

def end_conversation_twiML(final_text):
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        + twiml_say(final_text)
        + "</Response>"
    )
    return Response(xml, mimetype="text/xml")

def continue_conversation_twiML(bot_text, followup_hint="Wenn du fertig bist, drücke die Raute."):
    # Antwort + erneute Aufnahme (Loop)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        + twiml_say(bot_text)
        + '<Pause length="1"/>'
        + twiml_say(followup_hint)
        + twiml_record()
        + "</Response>"
    )
    return Response(xml, mimetype="text/xml")

def escape_xml(s):
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

def extract_text(response_obj):
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

def should_end(user_text, turns, max_turns):
    if turns >= max_turns:
        return True
    if not user_text:
        return False  # leeres/unklares Audio -> noch eine Runde probieren
    t = user_text.lower()
    endings = ["nein danke", "nein, danke", "danke, das war's", "das wars", "tschüss", "auf wiedersehen", "passt so", "alles gut", "fertig"]
    return any(phrase in t for phrase in endings)

# ------- Routes -------

@app.route("/", methods=["GET"])
def index():
    return "OK"

@app.route("/telefon", methods=["POST"])
def telefon():
    # Einstieg: Begrüßung + erste Aufnahme
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        + twiml_say("Willkommen bei wowona. Mein Name ist Petra. Wie kann ich dir helfen?")
        + twiml_record()
        + "</Response>"
    )
    return Response(xml, mimetype="text/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        call_sid = request.form.get("CallSid", "")
        state = CALL_STATE.get(call_sid, {"turn": 0})
        state["turn"] = int(state.get("turn", 0)) + 1
        CALL_STATE[call_sid] = state

        recording_url = request.form.get("RecordingUrl")
        if not recording_url:
            # Kein Audio? Frage nochmal.
            return continue_conversation_twiML(
                "Ich habe dich leider nicht verstanden. Kannst du dein Anliegen bitte noch einmal kurz schildern?"
            )

        # --- schneller: WAV statt MP3 ---
        wav_url = recording_url + ".wav"

        # Kurzer Retry, falls Twilio noch nicht fertig gespeichert hat
        content = None
        for _ in range(6):  # ~1.2s
            r = requests.get(wav_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=10)
            if r.status_code == 200 and int(r.headers.get("Content-Length", 0)) > 0:
                content = r.content
                break
            time.sleep(0.2)
        if content is None:
            r = requests.get(wav_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=10)
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
        print("TURN", state["turn"], "TRANSCRIPT:", transcript)

        # Direkte Antwort
        addr = "du" if REPLY_STYLE.lower() == "du" else "Sie"
        tone = "freundlich, klar, lösungsorientiert"
        prompt = (
            "Antworte direkt auf das Anliegen des Anrufers in natürlichem Deutsch (Anrede: {addr}). "
            "Ziel: eine hilfreiche, konkrete Antwort mit ggf. 1-2 gezielten Rückfragen, keine Zusammenfassung. "
            "Sei {tone}. Wenn Informationen fehlen, frage präzise nach. Halte dich an 1-3 Sätze, außer es werden konkrete Schritte verlangt.\n\n"
            "Gesagter Inhalt (Roh-Transkript): {transcript}"
        ).format(addr=addr, tone=tone, transcript=transcript)

        resp = client.responses.create(model="gpt-4o-mini", input=[{"role": "user", "content": prompt}])
        bot_text = extract_text(resp) or "Kannst du mir das bitte noch etwas genauer erklären, damit ich gezielt helfen kann?"

        # Abbruch-Logik
        if should_end(transcript, state["turn"], MAX_TURNS):
            # Aufräumen
            try:
                del CALL_STATE[call_sid]
            except Exception:
                pass
            return end_conversation_twiML(bot_text + " Danke für den Anruf und einen schönen Tag!")

        # Weiterfragen (Loop)
        # Kleiner Hinweis zur Steuerung am Telefon:
        followup_hint = "Wenn du fertig bist, drücke die Raute."
        return continue_conversation_twiML(bot_text, followup_hint)

    except Exception as e:
        print("Fehler bei der Verarbeitung:", e)
        return end_conversation_twiML("Es ist ein Fehler aufgetreten bei der Verarbeitung. Vielen Dank für deinen Anruf.")

# ------- Main -------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
