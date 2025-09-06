# main.py
import os
import json
import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware

import websockets

# -----------------------------------------------------------------------------
# Konfiguration / Environment
# -----------------------------------------------------------------------------
APP_NAME = "telefon-app"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_REALTIME_MODEL = os.getenv(
    "OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17"
).strip()

# Stimme & Latenz – wie besprochen:
OPENAI_VOICE = os.getenv("OPENAI_VOICE", "verse").strip()           # realistischere Stimme
TURN_SILENCE_MS = int(os.getenv("TURN_SILENCE_MS", "500"))          # ~0,5 s → schnelle Reaktion
OPENAI_LATENCY_MODE = os.getenv("OPENAI_LATENCY_MODE", "low").strip()

# Pfad für Twilio-WS:
TWILIO_WS_PATH = os.getenv("TWILIO_WS_PATH", "/twilio-media-stream").strip()

if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY fehlt – bitte in den Environment-Variablen setzen.")

# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------
app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(APP_NAME)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------
def build_twiml(stream_url: str) -> str:
    """ Baut TwiML, das die Voice-Session per <Connect><Stream> auf unseren WS bridged. """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{stream_url}"/>
  </Connect>
</Response>""".strip()


def build_self_ws_url(request: Request) -> str:
    """ Ermittelt die öffentliche WS-URL für Twilio basierend auf dem Host-Header. """
    host = request.headers.get("host") or request.url.hostname or "localhost"
    scheme = "wss"  # Auf Render via TLS
    return f"{scheme}://{host}{TWILIO_WS_PATH}"

# -----------------------------------------------------------------------------
# Endpunkte
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "app": APP_NAME}

@app.post("/telefon_live")
async def telefon_live(request: Request):
    """
    Twilio Voice Webhook: gibt TwiML zurück, um den eingehenden Anruf
    auf unseren WebSocket-Stream zu verbinden.
    """
    try:
        form = await request.form()
        payload = dict(form)
    except Exception:
        payload = {}

    logger.info("%s: Twilio Voice webhook payload: %s", APP_NAME, payload)

    ws_url = build_self_ws_url(request)

    if not OPENAI_API_KEY:
        text = ("Es tut mir leid, der Dienst ist aktuell nicht konfiguriert. "
                "Bitte später erneut versuchen.")
        error_twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say language="de-DE">{text}</Say>
  <Hangup/>
</Response>""".strip()
        return Response(content=error_twiml, media_type="text/xml")

    twiml = build_twiml(ws_url)
    return Response(content=twiml, media_type="text/xml")

# -----------------------------------------------------------------------------
# WebSocket Bridge: Twilio <-> OpenAI Realtime
# -----------------------------------------------------------------------------
@app.websocket(TWILIO_WS_PATH)
async def twilio_media_stream(ws: WebSocket):
    """
    Bridged Twilio Media Streams mit OpenAI Realtime.
    Audio: G.711 μ-Law 8 kHz (in/out), keine lokale Konvertierung nötig.
    """
    await ws.accept()
    logger.info("%s: Twilio WS verbunden", APP_NAME)

    openai_url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"

    # OpenAI-WebSocket-Verbindung aufbauen (websockets >= 12: additional_headers)
    try:
        openai_ws = await websockets.connect(
            openai_url,
            additional_headers={  # <- FIX gegenüber vorher (extra_headers ➜ additional_headers)
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
            ping_interval=20,
            ping_timeout=20,
            max_size=10_000_000,
        )
    except Exception as e:
        logger.exception("%s: OpenAI WS Verbindung fehlgeschlagen: %s", APP_NAME, e)
        try:
            await ws.send_text(json.dumps({"event": "clear"}))
        except Exception:
            pass
        await ws.close()
        return

    # Session konfigurieren (Stimme, VAD, Formate, Latenz)
    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["audio", "text"],
            "voice": OPENAI_VOICE,
            "instructions": (
                "Antworte kurz, freundlich und auf Deutsch. "
                "Klinge natürlich, vermeide lange Pausen, und beginne zügig zu sprechen."
            ),
            "turn_detection": {
                "type": "server_vad",
                "silence_trigger_ms": TURN_SILENCE_MS,   # → schnelle Reaktion nach ~0,5 s Stille
                "prefix_padding_ms": 120,
                "threshold": 0.5
            },
            "input_audio_format":  { "type": "g711_ulaw", "sample_rate_hz": 8000 },
            "output_audio_format": { "type": "g711_ulaw", "sample_rate_hz": 8000 },
            "latency": OPENAI_LATENCY_MODE,  # "low" für zügigere Ausgaben
        }
    }

    try:
        await openai_ws.send(json.dumps(session_update))
    except Exception as e:
        logger.exception("%s: session.update fehlgeschlagen: %s", APP_NAME, e)
        await openai_ws.close()
        await ws.close()
        return

    twilio_stream_sid: Optional[str] = None
    twilio_open = True
    openai_open = True

    async def pump_twilio_to_openai():
        nonlocal twilio_stream_sid, twilio_open, openai_open
        try:
            while True:
                msg = await ws.receive_text()
                data = json.loads(msg)
                event = data.get("event")

                if event == "connected":
                    logger.info("%s: WS connected: %s", APP_NAME, data)
                elif event == "start":
                    twilio_stream_sid = (data.get("start") or {}).get("streamSid")
                    logger.info(
                        "%s: Stream gestartet: streamSid=%s callSid=%s",
                        APP_NAME,
                        twilio_stream_sid,
                        (data.get("start") or {}).get("callSid"),
                    )
                elif event == "media":
                    media = data.get("media") or {}
                    payload = media.get("payload")
                    if payload:
                        # μ-law/8k von Twilio direkt an OpenAI
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": payload
                        }))
                        # Kein commit nötig – server_vad übernimmt die Turn-Erkennung.
                elif event == "stop":
                    logger.info("%s: Stream gestoppt: %s", APP_NAME, data)
                    break
                # "mark" und andere Events können ignoriert/geloggt werden
        except WebSocketDisconnect:
            logger.info("%s: Twilio WS disconnected", APP_NAME)
        except Exception as e:
            logger.exception("%s: Fehler (Twilio->OpenAI): %s", APP_NAME, e)
        finally:
            twilio_open = False

    async def pump_openai_to_twilio():
        nonlocal twilio_stream_sid, twilio_open, openai_open
        try:
            while True:
                raw = await openai_ws.recv()
                if isinstance(raw, (bytes, bytearray)):
                    continue

                obj = json.loads(raw)
                typ = obj.get("type") or obj.get("event") or ""

                if typ in (
                    "output_audio_chunk",
                    "response.audio.delta",
                    "response.output_audio.delta",
                ):
                    audio_b64 = obj.get("audio")
                    if not audio_b64:
                        delta = obj.get("delta") or {}
                        audio_b64 = delta.get("audio")

                    if audio_b64 and twilio_stream_sid:
                        await ws.send_text(json.dumps({
                            "event": "media",
                            "streamSid": twilio_stream_sid,
                            "media": { "payload": audio_b64 }
                        }))

                elif typ in ("response.completed", "response.stop"):
                    if twilio_stream_sid:
                        await ws.send_text(json.dumps({
                            "event": "mark",
                            "streamSid": twilio_stream_sid,
                            "mark": { "name": "response_completed" }
                        }))

        except websockets.ConnectionClosed:
            logger.info("%s: OpenAI WS geschlossen", APP_NAME)
        except Exception as e:
            logger.exception("%s: Fehler (OpenAI->Twilio): %s", APP_NAME, e)
        finally:
            openai_open = False

    try:
        await asyncio.gather(pump_twilio_to_openai(), pump_openai_to_twilio())
    finally:
        try:
            if not openai_ws.closed:
                await openai_ws.close()
        except Exception:
            pass

        try:
            if ws.client_state.name == "CONNECTED":
                await ws.close()
        except Exception:
            pass

        logger.info("%s: connection closed", APP_NAME)

# -----------------------------------------------------------------------------
# Lokaler Start (optional): uvicorn main:app --reload
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=False)
