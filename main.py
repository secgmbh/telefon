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
        <Record action="/antwort" maxLength="30" playBeep="false" transcribe="false" />
    </Response>"""
    return Response(response, mimetype="text/xml")

# 2) Nach der Aufnahme: MP3 holen → OpenAI transkribieren → kurze KI-Antwort
@app.route("/antwort", methods=["POST"])
def antwort():
    try:
        recording_url = request.form.get("RecordingUrl")
        if not recording_url:
            return twilio_response("Es wurde keine Aufnahme übermittelt.")

        mp3_url = recording_url + ".mp3"
        audio_response = requests.get(mp3_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=30)
        status = audio_response.status_code
        print("Twilio Download Status:", status)
        audio_response.raise_for_status()

        audio_path = "recording.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_response.content)
        print(f"Aufnahme gespeichert als {audio_path} ({len(audio_response.content)} Bytes)")

        # --- Transkription ---
        with open(audio_path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model=OPENAI_TRANSCRIBE_MODEL,
                file=f,
                language="de"
            )
        transcript = getattr(tr, "text", None) or (tr.get("text") if isinstance(tr, dict) else "")
        print("Transkript:", transcript)

        # --- Kurze Zusammenfassung / Antwort ---
        prompt = (
            "Fasse die Kernbotschaft des folgenden Telefonmitschnitts in 1-2 Sätzen zusammen "
            "und schlage, wenn sinnvoll, die nächsten Schritte vor. Antworte auf Deutsch.\n\n"
            f"Transkript: {transcript}"
        )
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": prompt}]
        )
        bot_text = extract_text(resp) or "Vielen Dank für Ihre Nachricht. Wir melden uns bald bei Ihnen."

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
