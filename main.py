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

# Stimme & Latenz
OPENAI_VOICE = os.getenv("OPENAI_VOICE", "verse").strip()    # realistische Stimme
TURN_SILENCE_MS = int(os.getenv("TURN_SILENCE_MS", "450"))   # schnellere Reaktion
OPENAI_LATENCY_MODE = os.getenv("OPENAI_LATENCY_MODE", "low").strip()

# Fallback-VAD (falls server_vad nicht automatisch triggert)
INACTIVITY_COMMIT_MS = int(os.getenv("INACTIVITY_COMMIT_MS", str(TURN_SILENCE_MS + 100)))

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

    # OpenAI-WebSocket-Verbindung aufbauen (websockets >= 12: additional_headers als Liste von Tupeln)
    try:
        openai_ws = await websockets.connect(
            openai_url,
            additional_headers=[("Authorization", f"Bearer {OPENAI_API_KEY}")],
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
                "silence_trigger_ms": TURN_SILENCE_MS,   # zügige Reaktion
                "prefix_padding_ms": 120,
                "threshold": 0.5
            },
            "input_audio_format":  { "type": "g711_ulaw", "sample_rate_hz": 8000 },
            "output_audio_format": { "type": "g711_ulaw", "sample_rate_hz": 8000 },
            "latency": OPENAI_LATENCY_MODE,
        }
    }

    try:
        await openai_ws.send(json.dumps(session_update))
        # Sofortige Begrüßung ausgeben:
        await openai_ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["audio"],
                "instructions": "Hallo! Ich bin Ihre KI am Telefon. Wie kann ich helfen?",
            }
        }))
    except Exception as e:
        logger.exception("%s: session.update/Begrüßung fehlgeschlagen: %s", APP_NAME, e)
        await openai_ws.close()
        await ws.close()
        return

    twilio_stream_sid: Optional[str] = None
    twilio_open = True
    openai_open = True

    # Fallback-VAD Status
    loop = asyncio.get_running_loop()
    last_audio_ts: float = 0.0
    buffer_open = False
    vad_task_cancelled = False

    async def fallback_vad_committer():
        """ Commit + response.create, wenn kurzzeitig keine Audioframes mehr kommen. """
        nonlocal last_audio_ts, buffer_open, openai_open, vad_task_cancelled
        try:
            while openai_open and not vad_task_cancelled:
                await asyncio.sleep(0.1)
                if buffer_open and last_audio_ts > 0:
                    ms_since = (loop.time() - last_audio_ts) * 1000.0
                    if ms_since >= INACTIVITY_COMMIT_MS:
                        try:
                            await openai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                            await openai_ws.send(json.dumps({
                                "type": "response.create",
                                "response": {"modalities": ["audio"]}
                            }))
                        except Exception as e:
                            logger.exception("%s: Fallback-VAD commit/response fehlgeschlagen: %s", APP_NAME, e)
                        finally:
                            buffer_open = False
        except asyncio.CancelledError:
            pass

    async def pump_twilio_to_openai():
        nonlocal twilio_stream_sid, twilio_open, openai_open, last_audio_ts, buffer_open
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
                        buffer_open = True
                        last_audio_ts = loop.time()
                elif event == "stop":
                    logger.info("%s: Stream gestoppt: %s", APP_NAME, data)
                    break
                # andere Events ignorieren/loggen
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
                    # Falls OpenAI jemals binär sendet: aktuell ignorieren.
                    continue

                obj = json.loads(raw)
                t = (obj.get("type") or obj.get("event") or "").lower()

                # Mögliche Audio-Delta-Events abdecken
                audio_b64: Optional[str] = None
                if t in ("output_audio_chunk", "response.audio.delta", "response.output_audio.delta",
                         "output_audio.delta", "response.delta"):
                    # unterschiedliche Strukturen tolerieren
                    if "audio" in obj and isinstance(obj["audio"], str):
                        audio_b64 = obj["audio"]
                    elif "delta" in obj and isinstance(obj["delta"], dict) and isinstance(obj["delta"].get("audio"), str):
                        audio_b64 = obj["delta"]["audio"]

                if audio_b64 and twilio_stream_sid:
                    await ws.send_text(json.dumps({
                        "event": "media",
                        "streamSid": twilio_stream_sid,
                        "media": { "payload": audio_b64 }
                    }))
                    continue

                if t in ("response.completed", "response.stop"):
                    if twilio_stream_sid:
                        await ws.send_text(json.dumps({
                            "event": "mark",
                            "streamSid": twilio_stream_sid,
                            "mark": { "name": "response_completed" }
                        }))
                    continue

                if t in ("error", "response.error"):
                    logger.error("%s: OpenAI Error: %s", APP_NAME, obj)
                    continue

        except websockets.ConnectionClosed:
            logger.info("%s: OpenAI WS geschlossen", APP_NAME)
        except Exception as e:
            logger.exception("%s: Fehler (OpenAI->Twilio): %s", APP_NAME, e)
        finally:
            openai_open = False

    # Tasks starten (inkl. Fallback-VAD)
    vad_task = asyncio.create_task(fallback_vad_committer())
    try:
        await asyncio.gather(pump_twilio_to_openai(), pump_openai_to_twilio())
    finally:
        # Fallback-VAD beenden
        vad_task_cancelled = True
        try:
            vad_task.cancel()
        except Exception:
            pass

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
