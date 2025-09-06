# main.py
import os
import json
import base64
import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import PlainTextResponse, Response, HTMLResponse
import websockets

# -----------------------------------------------------------------------------
# Konfiguration & Logging
# -----------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("telefon-app")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
OPENAI_VOICE = os.getenv("OPENAI_VOICE", "verse")

if not OPENAI_API_KEY:
    log.warning("OPENAI_API_KEY ist nicht gesetzt! Ohne Key gibt es keine Antworten.")

# 20 ms Frames bei 8 kHz μ-law: 8000 samples/s * 0.02 s = 160 bytes
TWILIO_FRAME_BYTES = 160

app = FastAPI()


# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------
def twiml_stream(ws_url: str) -> str:
    # WICHTIG: bidirectional="true" => Nur so spielt Twilio Audio zurück!
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}" bidirectional="true"/>
  </Connect>
</Response>""".strip()


async def openai_connect() -> websockets.WebSocketClientProtocol:
    """
    Baut die Realtime-WS-Verbindung zu OpenAI auf und konfiguriert
    Audioformate + VAD. Gibt das WebSocket-Objekt zurück.
    """
    url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}&protocol=1.0"
    headers = [
        ("Authorization", f"Bearer {OPENAI_API_KEY}"),
        ("OpenAI-Beta", "realtime=v1"),
    ]
    ws = await websockets.connect(uri=url, extra_headers=headers, max_size=16 * 1024 * 1024)
    log.info("telefon-app: OpenAI WS verbunden")

    # Session-Update: Deutsch, realistische Stimme, μ-law I/O, niedrige Latenz
    sess = {
        "type": "session.update",
        "session": {
            "instructions": (
                "Sprich natürlich und freundlich auf Deutsch. "
                "Halte Antworten kurz. Vermeide lange Pausen."
            ),
            "voice": OPENAI_VOICE,
            "modalities": ["text", "audio"],
            "input_audio_format": {"type": "g711_ulaw", "sample_rate": 8000},
            "output_audio_format": {"type": "g711_ulaw", "sample_rate": 8000},
            "turn_detection": {
                "type": "server_vad",
                # frühes Anspringen + kurze Stille für flotte Reaktionen
                "threshold": 0.5,
                "prefix_padding_ms": 150,
                "silence_duration_ms": 650,
            },
        },
    }
    await ws.send(json.dumps(sess))
    return ws


async def send_greeting(openai_ws: websockets.WebSocketClientProtocol):
    """
    Löst eine kurze Begrüßung aus (Audio).
    """
    greeting_text = (
        "Hallo! Ich bin bereit. Sage mir einfach, womit ich helfen kann."
    )
    msg = {
        "type": "response.create",
        "response": {
            "modalities": ["audio"],
            "instructions": greeting_text,
        },
    }
    await openai_ws.send(json.dumps(msg))
    log.debug("telefon-app: Begrüßung ausgelöst")


def chunk_and_base64(buf: bytearray, chunk_size: int = TWILIO_FRAME_BYTES):
    """
    Zerlegt einen Bytepuffer in Base64-chunks in der gewünschten Framegröße.
    Gibt eine Liste von base64-Strings zurück. Der Rest verbleibt im Puffer.
    """
    out = []
    while len(buf) >= chunk_size:
        piece = bytes(buf[:chunk_size])
        del buf[:chunk_size]
        out.append(base64.b64encode(piece).decode("ascii"))
    return out


# -----------------------------------------------------------------------------
# HTTP-Routen
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <h1>Telefon Realtime</h1>
    <p>Service läuft. Verwende <code>POST /telefon_live</code> als Twilio Voice Webhook.</p>
    <p>Zum Testen im Browser: <a href="/telefon_live">/telefon_live</a> (liefert TwiML)</p>
    """.strip()


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    return "ok"


@app.get("/telefon_live")
async def telefon_live_get(request: Request):
    # Browser-Test: TwiML anzeigen
    host = request.headers.get("host", "localhost")
    ws_url = f"wss://{host}/twilio-media-stream"
    xml = twiml_stream(ws_url)
    return Response(content=xml, media_type="text/xml")


@app.post("/telefon_live")
async def telefon_live_post(request: Request):
    """
    Twilio Voice Webhook (POST). Liefert TwiML zurück, welches
    Twilio mit unserem WS-Endpunkt verbindet – bidirektional!
    """
    form = dict(await request.form())
    log.info("telefon-app: Twilio Voice webhook payload: %s", form)
    host = request.headers.get("host", "localhost")
    ws_url = f"wss://{host}/twilio-media-stream"
    xml = twiml_stream(ws_url)
    return Response(content=xml, media_type="text/xml")


# -----------------------------------------------------------------------------
# WebSocket: Twilio <-> OpenAI Brücke
# -----------------------------------------------------------------------------
@app.websocket("/twilio-media-stream")
async def twilio_media_stream(ws: WebSocket):
    await ws.accept()
    log.info("telefon-app: Twilio WS verbunden")

    openai_ws: Optional[websockets.WebSocketClientProtocol] = None
    stream_sid: Optional[str] = None

    # Buffer für Audio-Out (von OpenAI zu Twilio)
    out_buf = bytearray()

    # Flag & Task-Steuerung
    running = True
    have_uncommitted_audio = False

    async def pump_twilio_to_openai():
        nonlocal running, openai_ws, stream_sid, have_uncommitted_audio
        last_commit = 0.0

        try:
            while running:
                data = await ws.receive_text()
                msg = json.loads(data)

                ev = msg.get("event")
                if ev == "start":
                    stream_sid = msg.get("start", {}).get("streamSid")
                    log.info("telefon-app: WS connected: %s", {"event": "start"})
                    # OpenAI-WS hier aufbauen
                    openai_ws = await openai_connect()
                    # Begrüßung auslösen
                    await send_greeting(openai_ws)

                elif ev == "media":
                    if not openai_ws:
                        continue
                    payload_b64 = msg.get("media", {}).get("payload")
                    if not payload_b64:
                        continue

                    # μ-law (8k) Rohdaten direkt an OpenAI anhängen
                    append_msg = {
                        "type": "input_audio_buffer.append",
                        "audio": payload_b64,  # μ-law/8k in base64
                    }
                    await openai_ws.send(json.dumps(append_msg))
                    have_uncommitted_audio = True

                    # Leichtes Auto-Commit zur flotten Turn-Erkennung
                    now = asyncio.get_event_loop().time()
                    if have_uncommitted_audio and (now - last_commit) > 0.5:
                        await openai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                        have_uncommitted_audio = False
                        last_commit = now

                elif ev == "mark":
                    # Optional: auf Marker reagieren
                    pass

                elif ev == "stop":
                    log.info("telefon-app: Stream gestoppt: %s", msg.get("stop") or msg)
                    running = False
                    break

        except WebSocketDisconnect:
            log.info("telefon-app: Twilio WS getrennt")
        except Exception as e:
            log.exception("telefon-app: Fehler im Twilio->OpenAI Pump: %s", e)
        finally:
            running = False

    async def pump_openai_to_twilio():
        nonlocal running, openai_ws, out_buf, stream_sid
        if not openai_ws:
            # auf Aufbau warten
            for _ in range(40):
                if openai_ws:
                    break
                await asyncio.sleep(0.05)
        if not openai_ws:
            log.error("telefon-app: OpenAI WS nicht verfügbar; kein Audio-Out.")
            return

        try:
            while running:
                raw = await openai_ws.recv()
                try:
                    msg = json.loads(raw)
                except Exception:
                    # Falls OpenAI binäre Frames schickt (sollte nicht passieren in v1.0)
                    continue

                mtype = msg.get("type")

                if mtype == "response.audio.delta":
                    # Audio kommt als base64 (μ-law/8k). Wir schicken in 20ms-Frames an Twilio.
                    audio_b64 = msg.get("audio")
                    if audio_b64:
                        out_buf.extend(base64.b64decode(audio_b64))
                        chunks = chunk_and_base64(out_buf, TWILIO_FRAME_BYTES)
                        for c in chunks:
                            if not stream_sid:
                                continue
                            await ws.send_text(json.dumps({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": c}
                            }))
                elif mtype == "response.completed":
                    # Rest puffern & raus
                    chunks = chunk_and_base64(out_buf, TWILIO_FRAME_BYTES)
                    for c in chunks:
                        if not stream_sid:
                            continue
                        await ws.send_text(json.dumps({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": c}
                        }))
                elif mtype == "error":
                    log.error("telefon-app: OpenAI Fehler: %s", msg)
                # Weitere Typen ignorieren wir bewusst (Logs wären zu laut)

        except Exception as e:
            if running:
                log.exception("telefon-app: Fehler im OpenAI->Twilio Pump: %s", e)
        finally:
            running = False

    async def keepalive_marks():
        """
        Optional: regelmäßig Mark-Events senden; einige Clients nutzen das zum Flushing.
        """
        while running:
            await asyncio.sleep(5)
            if stream_sid:
                try:
                    await ws.send_text(json.dumps({
                        "event": "mark",
                        "streamSid": stream_sid,
                        "mark": {"name": "keepalive"}
                    }))
                except Exception:
                    break

    # Tasks starten
    tasks = [
        asyncio.create_task(pump_twilio_to_openai()),
        asyncio.create_task(pump_openai_to_twilio()),
        asyncio.create_task(keepalive_marks()),
    ]

    # Auf Ende warten
    try:
        await asyncio.gather(*tasks)
    finally:
        # Aufräumen
        for t in tasks:
            if not t.done():
                t.cancel()
        if openai_ws:
            try:
                await openai_ws.close()
            except Exception:
                pass
        try:
            await ws.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Uvicorn Start (nur lokal relevant – Render startet via Gunicorn)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
