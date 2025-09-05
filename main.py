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
            f"Antworte direkt auf das Anliegen des Anrufers in natürlichem Deutsch (Anrede: {addr}). "
            f"Ziel: eine hilfreiche, konkrete Antwort mit ggf. 1-2 gezielten Rückfragen, keine Zusammenfassung. "
            f"Sei {tone}. Wenn Informationen fehlen, frage präzise nach. "
            f"Halte dich an 1-3 Sätze, außer es werden konkrete Schritte verlangt.

"
            f"Gesagter Inhalt (Roh-Transkript): {transcript}"
        )

        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": prompt}],
        )
        bot_text = extract_text(resp) or "Danke dir! Kannst du bitte kurz präzisieren, wobei ich dir genau helfen soll?"

        return twilio_response(bot_text)

    except Exception as e:
        print("Fehler bei der Verarbeitung:", e)
        return twilio_response("Es ist ein Fehler aufgetreten bei der Verarbeitung der Aufnahme.")

        wav_url = recording_url + ".wav"

        # Kurzer Retry, falls Datei noch nicht bereit ist
        import time, io
        content = None
        for _ in range(6):
            r = requests.get(wav_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=10)
            if r.status_code == 200 and int(r.headers.get("Content-Length", 1)) > 1:
                content = r.content
                break
            time.sleep(0.2)
        if content is None:
            r = requests.get(wav_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=10)
            r.raise_for_status()
            content = r.content

        print("WAV Bytes:", len(content))

        file_like = io.BytesIO(content)
        file_like.name = "recording.wav"

        tr = client.audio.transcriptions.create(
            model=OPENAI_TRANSCRIBE_MODEL,
            file=file_like,
            language="de"
        )
        transcript = getattr(tr, "text", None) or (tr.get("text") if isinstance(tr, dict) else "")
        print("Transkript:", transcript)

        # --- Direkte Antwort ---
        reply_style = os.getenv("REPLY_STYLE", "du")  # "du" | "sie"
        tone = "freundlich, klar, lösungsorientiert"
        addr = "du" if reply_style.lower() == "du" else "Sie"
        prompt = (
            f"Antworte direkt auf das Anliegen des Anrufers in natürlichem Deutsch (Anrede: {addr}). "
            f"Ziel: eine hilfreiche, konkrete Antwort mit ggf. 1-2 gezielten Rückfragen, keine Zusammenfassung. "
            f"Sei {tone}. Wenn Informationen fehlen, frage präzise nach. "
            f"Halte dich an 1-3 Sätze, außer es werden konkrete Schritte verlangt.\n\n"
            f"Gesagter Inhalt (Roh-Transkript): {transcript}"
        )
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": prompt}]
        )
        bot_text = extract_text(resp) or "Danke dir! Kannst du bitte noch kurz präzisieren, wobei ich dir genau helfen soll?"

        return twilio_response(bot_text)

    except Exception as e:
        print("Fehler bei der Verarbeitung:", e)
        return twilio_response("Es ist ein Fehler aufgetreten bei der Verarbeitung der Aufnahme.")


# ====== Hilfen ======

def twilio_response(text: str) -> Response:
    return Response(f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <Response>
        <Say language=\"de-DE\">{escape_xml(text)}</Say>
    </Response>""", mimetype="text/xml")


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
