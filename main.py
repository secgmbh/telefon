import os
import json
import time
import base64
import logging
import wave
import requests
from urllib.parse import parse_qs

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import PlainTextResponse

# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("telefon-app")

# =========================================================
# μ-law ⇢ PCM16 (ohne audioop) für optionalen Mitschnitt
# =========================================================
MU_LAW_EXPAND_TABLE = []

def _build_mulaw_table():
    tbl = []
    for i in range(256):
        mu = ~i & 0xFF
        sign = mu & 0x80
        exponent = (mu >> 4) & 0x07
        mantissa = mu & 0x0F
        magnitude = ((mantissa << 4) + 8) << exponent
        sample = magnitude - 0x84
        if sign:
            sample = -sample
        if sample > 32767:
            sample = 32767
        if sample < -32768:
            sample = -32768
        tbl.append(sample)
    return tbl

def mulaw_to_pcm16(mu_bytes: bytes) -> bytes:
    """Konvertiert μ-law 8kHz/8bit -> PCM16 LE (für WAV)."""
    global MU_LAW_EXPAND_TABLE
    if not MU_LAW_EXPAND_TABLE:
        MU_LAW_EXPAND_TABLE[:] = _build_mulaw_table()
    out = bytearray()
    for b in mu_bytes:
        s = MU_LAW_EXPAND_TABLE[b]
        out += int(s).to_bytes(2, "little", signed=True)
    return bytes(out)

def xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&apos;")
    )

# =========================================================
# KI (OpenAI) – einfacher Chat-Wrapper
# =========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Einfacher In-Memory Verlauf pro CallSid
CONV = {}  # { callSid: [{"role":"system/user/assistant","content":"..."}] }
MAX_TURNS = 8  # Verlauf begrenzen

SYSTEM_PROMPT = (
    "Du bist ein freundlicher deutschsprachiger Telefonassistent. "
    "Antworte kurz, klar und gut verständlich, max. 2–3 Sätze. "
    "Keine Emojis, keine Markdown-Formatierung."
)

def llm_answer(call_sid: str, user_text: str) -> str:
    if not OPENAI_API_KEY:
        return "Die KI ist derzeit nicht konfiguriert. Bitte setze den OPENAI_API_KEY."

    hist = CONV.get(call_sid)
    if not hist:
        hist = [{"role": "system", "content": SYSTEM_PROMPT}]
        CONV[call_sid] = hist

    hist.append({"role": "user", "content": user_text})
    # Verlauf begrenzen (aber System behalten)
    trimmed = [hist[0]] + hist[-(2*MAX_TURNS):]
    CONV[call_sid] = trimmed

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": trimmed,
            "temperature": 0.5,
            "max_tokens": 180,
        }
        r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=25)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        # Im Verlauf speichern
        CONV[call_sid].append({"role": "assistant", "content": text})
        # Verlauf wieder beschneiden
        CONV[call_sid] = [CONV[call_sid][0]] + CONV[call_sid][-(2*MAX_TURNS):]
        return text
    except Exception as e:
        log.exception("OpenAI Fehler: %s", e)
        return "Entschuldigung, gerade gab es ein Problem bei der Verarbeitung."

# =========================================================
# FastAPI App
# =========================================================
app = FastAPI(title="Telefon Live + KI (Variante A)", version="1.2.0")

@app.get("/")
async def root():
    return PlainTextResponse("OK")

@app.get("/health")
async def health():
    return {"status": "ok"}

# =========================================================
# Variante A: Twilio <Gather> (Speech)  → LLM → <Say>
# =========================================================
TWILIO_VOICE = os.getenv("TWILIO_VOICE", "alice")
TWILIO_LANG = os.getenv("TWILIO_LANG", "de-DE")

def twiml_response(xml: str) -> Response:
    return Response(content=xml, media_type="application/xml")

@app.post("/telefon_ai")
async def telefon_ai(request: Request):
    """
    Startpunkt für den Dialog: Begrüßung + Gather (Speech) mit speechTimeout=auto.
    """
    greet = os.getenv("AI_GREETING", "Hallo! Du kannst jetzt mit mir sprechen. Womit kann ich helfen?")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Gather input="speech" language="{TWILIO_LANG}" speechTimeout="auto" action="/telefon_ai/handle" method="POST">
    <Say language="{TWILIO_LANG}" voice="{TWILIO_VOICE}">{xml_escape(greet)}</Say>
  </Gather>
  <Say language="{TWILIO_LANG}" voice="{TWILIO_VOICE}">Ich habe keine Eingabe gehört. Bitte versuche es nochmal.</Say>
  <Redirect method="POST">/telefon_ai</Redirect>
</Response>"""
    return twiml_response(twiml)

@app.post("/telefon_ai/handle")
async def telefon_ai_handle(request: Request):
    """
    Wird von Twilio nach dem <Gather> aufgerufen. Holt SpeechResult und fragt die KI.
    Antwort wird per <Say> gesprochen und es folgt ein erneutes <Gather>.
    """
    raw = (await request.body()).decode("utf-8", "ignore")
    form = {k: v[0] if isinstance(v, list) and v else v for k, v in parse_qs(raw).items()}
    call_sid = form.get("CallSid", "")
    speech = (form.get("SpeechResult") or "").strip()
    log.info("Gather handle: CallSid=%s SpeechResult=%r", call_sid, speech)

    if not speech:
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say language="{TWILIO_LANG}" voice="{TWILIO_VOICE}">Das habe ich nicht verstanden. Bitte wiederhole deine Frage.</Say>
  <Redirect method="POST">/telefon_ai</Redirect>
</Response>"""
        return twiml_response(twiml)

    answer = llm_answer(call_sid, speech)
    # Sicherheitshalber auf 900 Zeichen deckeln (TTS-Pragmatik)
    answer = answer[:900]

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say language="{TWILIO_LANG}" voice="{TWILIO_VOICE}">{xml_escape(answer)}</Say>
  <Gather input="speech" language="{TWILIO_LANG}" speechTimeout="auto" action="/telefon_ai/handle" method="POST">
    <Say language="{TWILIO_LANG}" voice="{TWILIO_VOICE}">Noch etwas?</Say>
  </Gather>
  <Say language="{TWILIO_LANG}" voice="{TWILIO_VOICE}">Okay, dann beende ich den Anruf. Auf Wiederhören!</Say>
  <Hangup/>
</Response>"""
    return twiml_response(twiml)

# Optional: Verlauf löschen, wenn Twilio eine Status-Callback-URL hierauf zeigen lässt
@app.post("/telefon_ai/cleanup")
async def telefon_ai_cleanup(request: Request):
    raw = (await request.body()).decode("utf-8", "ignore")
    form = {k: v[0] if isinstance(v, list) and v else v for k, v in parse_qs(raw).items()}
    call_sid = form.get("CallSid", "")
    if call_sid in CONV:
        del CONV[call_sid]
        log.info("Verlauf für %s gelöscht.", call_sid)
    return PlainTextResponse("ok")

# =========================================================
# Dein bestehender Media-Stream (/telefon_live) bleibt erhalten
# =========================================================
@app.post("/telefon_live")
async def telefon_live(request: Request):
    """Twilio Voice Webhook (POST form-encoded) -> TwiML mit <Connect><Stream/> für reines Audio-Streaming."""
    try:
        raw_body = (await request.body()).decode("utf-8", "ignore")
        form = {k: v[0] if isinstance(v, list) and v else v for k, v in parse_qs(raw_body).items()}
        log.info("Twilio Voice webhook payload: %s", form)
    except Exception as e:
        log.warning("Konnte Formdaten nicht parsen: %s", e)

    ws_url = os.environ.get("TWILIO_WS_URL")
    if not ws_url:
        base = str(request.url).split("/telefon_live")[0]
        if base.startswith("https://"):
            ws_url = base.replace("https://", "wss://") + "/twilio-media-stream"
        else:
            ws_url = base.replace("http://", "ws://") + "/twilio-media-stream"

    say_text = os.getenv("TWILIO_SAY_TEXT", "Verbunden. Einen Moment bitte.")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say language="de-DE">{xml_escape(say_text)}</Say>
  <Connect>
    <Stream url="{xml_escape(ws_url)}" track="inbound_track" />
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.websocket("/twilio-media-stream")
async def twilio_media_stream(ws: WebSocket):
    await ws.accept()
    log.info("Twilio WS verbunden")

    # Aufnahme-Parameter (optional)
    record_wav = os.getenv("RECORD_WAV", "false").lower() in ("1", "true", "yes", "on")
    max_seconds = int(os.getenv("RECORD_MAX_SECONDS", "60"))
    frames_per_second = 50  # 20ms Frames
    max_frames = max_seconds * frames_per_second

    frame_count = 0
    byte_count = 0
    last_log_time = time.monotonic()
    last_rate_window = last_log_time
    stream_sid = None
    call_sid = None

    pcm_buf = bytearray() if record_wav else None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            ev = msg.get("event")

            if ev == "connected":
                log.info("WS connected: %s", msg)

            elif ev == "start":
                info = msg.get("start", {})
                stream_sid = info.get("streamSid")
                call_sid = info.get("callSid")
                log.info("Stream gestartet: streamSid=%s callSid=%s", stream_sid, call_sid)

            elif ev == "media":
                media = msg.get("media", {})
                b64 = media.get("payload")
                if b64:
                    mu = base64.b64decode(b64)
                    frame_count += 1
                    byte_count += len(mu)

                    now = time.monotonic()
                    if now - last_log_time >= 1.0:
                        rate = int(byte_count / max(1e-6, now - last_rate_window))
                        log.info("Media: frames=%d, mu-law bytes (rate) ~= %d B/s", frame_count, rate)
                        last_log_time = now
                        last_rate_window = now
                        byte_count = 0

                    if record_wav and len(pcm_buf) < max_frames * 160 * 2:
                        pcm_buf += mulaw_to_pcm16(mu)

            elif ev == "mark":
                pass  # optional

            elif ev == "stop":
                info = msg.get("stop", {})
                log.info("Stream gestoppt: %s", info)
                break

    except WebSocketDisconnect:
        log.info("Twilio WS getrennt")
    except Exception as e:
        log.exception("Fehler im Twilio WS: %s", e)
    finally:
        if record_wav and pcm_buf and stream_sid:
            try:
                fname = f"/tmp/call_{stream_sid}.wav"
                with wave.open(fname, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(8000)
                    wf.writeframes(pcm_buf)
                log.info("WAV gespeichert: %s (%0.1f s)", fname, len(pcm_buf) / (2 * 8000))
            except Exception as e:
                log.warning("Konnte WAV nicht schreiben: %s", e)

        try:
            await ws.close()
        except Exception:
            pass

# =========================================================
# Lokaler Start
# =========================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
