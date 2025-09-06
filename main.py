# main.py
# FastAPI-App für Telefon-KI per Twilio <Gather> (Speech) + OpenAI Chat + Twilio <Say> (TTS, Polly Neural)
# Fokus: niedrigere Latenz (barge-in, kurze Antworten) & natürlichere Stimme

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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()  # schneller, okay für Telefon
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

# Natürlichere Stimme: Twilio unterstützt Polly Neural direkt.
# Gute deutsche Stimmen: Polly.Marlene-Neural, Polly.Vicki-Neural
TTS_VOICE = os.getenv("TTS_VOICE", "Polly.Marlene-Neural").strip()
TTS_LANG = os.getenv("TTS_LANG", "de-DE").strip()

# Modell-Feintuning für Tempo
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.4"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "220"))

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

def _openai_chat(messages: List[Dict[str, str]], temperature: float = OPENAI_TEMPERATURE, max_tokens: int = OPENAI_MAX_TOKENS) -> str:
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
            # Keep-Alive kann bei manchen Hosts minimal helfen
            "Connection": "keep-alive",
        },
    )

    try:
        # kürzeres Timeout, um Hänger zu vermeiden
        with urlrequest.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            j = json.loads(body)
            content = (
                j.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not content:
                log.warning("OpenAI-Antwort leer oder unerwartet: %s", body)
                return "Dazu habe ich gerade keine passende Antwort. Formuliere es bitte kurz neu."
            return content
    except HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        log.error("OpenAI HTTPError %s: %s", e.code, err_body)
        return "Im Moment gibt es ein Problem bei der Verarbeitung. Bitte versuche es gleich nochmal."
    except URLError as e:
        log.error("OpenAI URLError: %s", e)
        return "Ich habe gerade keine Verbindung zum Sprachmodell. Bitte später erneut versuchen."
    except Exception as e:
        log.exception("Unerwarteter Fehler bei OpenAI-Call: %s", e)
        return "Da ist etwas Unerwartetes passiert. Bitte frag mich nochmal."

def _base_url_from_request(req: Request) -> str:
    # z.B. "https://dein-host.onrender.com"
    return str(req.base_url).rstrip("/")

def _gather_twiml(prompt_text: str, action_url: str) -> str:
    """
    Baut ein TwiML mit <Gather input="speech"> und einem Prompt (<Say>).
    - bargeIn="true": Anrufer kann schon während der Ansage sprechen.
    - speechModel="phone_call": passende Erkennung für Telefon.
    """
    prompt_text = _escape_xml(prompt_text)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Gather input="speech"
          language="{TTS_LANG}"
          action="{action_url}"
          method="POST"
          speechTimeout="auto"
          bargeIn="true"
          speechModel="phone_call">
    <Say voice="{TTS_VOICE}" language="{TTS_LANG}">{prompt_text}</Say>
  </Gather>
  <Say voice="{TTS_VOICE}" language="{TTS_LANG}">Ich habe nichts gehört. Auf Wiederhören!</Say>
</Response>"""

def _answer_and_reprompt_twiml(answer_text: str, action_url: str) -> str:
    """
    Antwort sprechen und erneut nach einer weiteren Frage fragen (erneutes Gather, wieder mit barge-in).
    """
    answer_text = _escape_xml(answer_text)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="{TTS_VOICE}" language="{TTS_LANG}">{answer_text}</Say>
  <Gather input="speech"
          language="{TTS_LANG}"
          action="{action_url}"
          method="POST"
          speechTimeout="auto"
          bargeIn="true"
          speechModel="phone_call">
    <Say voice="{TTS_VOICE}" language="{TTS_LANG}">Noch eine Frage?</Say>
  </Gather>
  <Say voice="{TTS_VOICE}" language="{TTS_LANG}">Alles klar. Tschüss!</Say>
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

    greeting = "Hallo! Was möchtest du wissen?"
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
        xml = _gather_twiml("Ich habe dich nicht verstanden. Was möchtest du wissen?", action_url)
        return _twiml_response(xml)

    # Kurzer, telefonoptimierter Stil
    system_prompt = (
        "Du bist eine freundliche, prägnante Telefon-KI auf Deutsch. "
        "Antworte in maximal zwei kurzen Sätzen, klar und ohne Fachjargon. "
        "Wenn sinnvoll, nenne eine konkrete nächste Option oder Frage."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    answer = _openai_chat(messages)

    # Antwort sprechen + erneut nachfragen
    xml = _answer_and_reprompt_twiml(answer, action_url)
    return _twiml_response(xml)
