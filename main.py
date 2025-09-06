# main.py
# FastAPI-App für Telefon-KI per Twilio <Gather> (Speech) + OpenAI Chat + Twilio <Say> (TTS)
# Ohne externe HTTP-Client-Libs (nur urllib aus der Stdlib) -> kein "requests" nötig.

import os
import json
import logging
from typing import List, Dict
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, PlainTextResponse

# ------------------------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------------------------

APP_NAME = "telefon-app"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()  # bei Bedarf ändern
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(APP_NAME)

app = FastAPI(title="Telefon-KI")

# ------------------------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------------------------

def _twiml_response(xml: str, status_code: int = 200) -> Response:
    return Response(content=xml, media_type="application/xml", status_code=status_code)

def _escape_xml(text: str) -> str:
    """Einfaches Escaping für <Say>-Text."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

def _openai_chat(messages: List[Dict[str, str]], temperature: float = 0.6, max_tokens: int = 350) -> str:
    """
    Ruft OpenAI Chat Completions mit urllib auf (kein 'requests' nötig).
    Gibt Assistant-Text zurück oder eine Fehlermeldung.
    """
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY fehlt")
        return "Entschuldigung, mein Schlüssel ist nicht konfiguriert. Bitte den Betreiber informieren."

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")

    req = urlrequest.Request(
        OPENAI_CHAT_URL,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urlrequest.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
            j = json.loads(body)
            # Erwartetes Format: choices[0].message.content
            content = (
                j.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not content:
                log.warning("OpenAI-Antwort leer oder unerwartet: %s", body)
                return "Ich habe darauf keine passende Antwort erhalten. Formuliere es bitte anders."
            return content
    except HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        log.error("OpenAI HTTPError %s: %s", e.code, err_body)
        return "Gerade gibt es ein Problem bei der Verarbeitung. Bitte versuche es gleich nochmal."
    except URLError as e:
        log.error("OpenAI URLError: %s", e)
        return "Ich habe keine Verbindung zum Sprachmodell. Bitte später erneut versuchen."
    except Exception as e:
        log.exception("Unerwarteter Fehler bei OpenAI-Call: %s", e)
        return "Da ist etwas Unerwartetes passiert. Bitte frag mich nochmal."

def _base_url_from_request(req: Request) -> str:
    # z.B. "https://dein-host.onrender.com"
    return str(req.base_url).rstrip("/")

def _gather_twiml(prompt_text: str, action_url: str) -> str:
    """
    Baut ein TwiML mit <Gather input="speech"> und einem Prompt (<Say>).
    """
    prompt_text = _escape_xml(prompt_text)
    # voice="Polly.Vicki-Neural" + language="de-DE" funktionieren ohne Polly-Voices.
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Gather input="speech" language="de-DE" action="{action_url}" method="POST" speechTimeout="auto">
    <Say voice="Polly.Vicki-Neural" language="de-DE">{prompt_text}</Say>
  </Gather>
  <Say voice="Polly.Vicki-Neural" language="de-DE">Ich habe nichts gehört. Auf Wiederhören!</Say>
</Response>"""

def _answer_and_reprompt_twiml(answer_text: str, action_url: str) -> str:
    """
    Antwort sprechen und erneut nach einer weiteren Frage fragen (erneutes Gather).
    """
    answer_text = _escape_xml(answer_text)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Vicki-Neural" language="de-DE">{answer_text}</Say>
  <Gather input="speech" language="de-DE" action="{action_url}" method="POST" speechTimeout="auto">
    <Say voice="Polly.Vicki-Neural" language="de-DE"></Say>
  </Gather>
  <Say voice="Polly.Vicki-Neural" language="de-DE">Alles klar. Tschüss!</Say>
</Response>"""

# ------------------------------------------------------------------------------
# Routen
# ------------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    return (
        "<h1>Telefon-KI ist bereit</h1>"
        "<p>Webhook: <code>POST /telefon_live</code></p>"
        "<p>Action-Route: <code>POST /telefon_live/process</code></p>"
    )

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

@app.post("/telefon_live")
async def telefon_live(request: Request):
    """
    Einstiegs-Webhook von Twilio (Voice → Webhook URL).
    Gibt TwiML mit <Gather> zurück, um Sprache zu erfassen.
    """
    form = await request.form()
    payload = dict(form)
    log.info("Twilio Voice webhook payload: %s", payload)

    base = _base_url_from_request(request)
    action_url = f"{base}/telefon_live/process"

    greeting = (
        "Willkommen bei wowona. Mein Name ist Petra, wie kann ich dir helfen?"
    )
    xml = _gather_twiml(greeting, action_url)
    return _twiml_response(xml)

@app.post("/telefon_live/process")
async def telefon_process(request: Request):
    """
    Action-URL von <Gather>. Erwartet 'SpeechResult' im Twilio-POST.
    Ruft OpenAI mit dem Transkript auf und gibt die Antwort als <Say> aus,
    dann erneut <Gather> für die nächste Frage.
    """
    form = await request.form()
    payload = dict(form)
    log.info("Twilio Gather payload: %s", payload)

    user_text = (payload.get("SpeechResult") or "").strip()
    base = _base_url_from_request(request)
    action_url = f"{base}/telefon_live/process"

    if not user_text:
        # Nichts gehört → erneut Gather anbieten
        xml = _gather_twiml("Ich habe dich nicht verstanden. Was möchtest du wissen?", action_url)
        return _twiml_response(xml)

    # OpenAI aufrufen
    system_prompt = (
        "Du bist eine freundliche, prägnante Telefon-KI auf Deutsch. "
        "Antworte kurz, klar und gut verständlich für Telefon-Audio. "
        "Vermeide lange Listen und Fachjargon."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    answer = _openai_chat(messages)

    # Antwort sprechen + erneut nachfragen
    xml = _answer_and_reprompt_twiml(answer, action_url)
    return _twiml_response(xml)
