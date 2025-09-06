# main.py
import os
import json
import base64
import asyncio
import logging
from typing import Optional, List, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse

import websockets

# -----------------------------------------------------------------------------
# Konfiguration & Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("telefon-app")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
OPENAI_VOICE = os.getenv("OPENAI_VOICE", "verse")  # realistischere Stimme
# Twilio → OpenAI: sehr kurze Sprechpausen für schnelle Antworten
INACTIVITY_COMMIT_MS = int(os.getenv("INACTIVITY_COMMIT_MS", "550"))  # optional anpassen (z. B. 500–800)

# Hinweis: Twilio sendet G.711 µ-law (PCMU, 8 kHz). Wir sagen OpenAI explizit, was reinkommt und raus soll.
INPUT_AUDIO_FORMAT = os.getenv("INPUT_AUDIO_FORMAT", "g711_ulaw")
OUTPUT_AUDIO_FORMAT = os.getenv("OUTPUT_AUDIO_FORMAT", "g711_ulaw")

app = FastAPI(title="Telefon KI Bridge")

# -----------------------------------------------------------------------------
# Root: kleine Info-Seite
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
      <body style="font-family: system-ui; line-height:1.5">
        <h3>Telefon-KI Bridge läuft ✅</h3>
        <ul>
          <li>Voice Webhook: <code>/telefon_live</code></li>
          <li>Media Stream WS: <code>/twilio-media-stream</code></li>
        </ul>
      </body>
    </html>
    """


# -----------------------------------------------------------------------------
# Twilio Voice → TwiML: verbindet den Anruf mit unserem WebSocket-Endpunkt
# -----------------------------------------------------------------------------
@app.post("/telefon_live")
async def telefon_live(request: Request):
    try:
        form = dict(await request.form())
        log.info("telefon-app: Twilio Voice webhook payload: %s", form)
    except Exception:
        log.info("telefon-app: Twilio Voice webhook payload: (kein Form-Parsen möglich)")

    # WS-Basis ermitteln
    override_ws = os.getenv("WS_BASE_URL")  # z. B. wss://dein-service.onrender.com/twilio-media-stream
    if override_ws:
        ws_url = override_ws
    else:
        # automatisch aus Request ableiten
        # Twilio braucht wss://
        host = request.url.hostname
        ws_url = f"wss://{host}/twilio-media-stream"

    # TwiML mit <Connect><Stream>
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}" />
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


# -----------------------------------------------------------------------------
# Helfer: OpenAI-Realtime WebSocket verbinden
# -----------------------------------------------------------------------------
async def connect_openai_ws() -> websockets.WebSocketClientProtocol:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY fehlt in den Umgebungsvariablen.")

    url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"

    # Bei neueren websockets-Versionen heißt der Header-Parameter "additional_headers"
    # (statt "extra_headers"). Das vermeidet den früheren Fehler.
    headers: List[Tuple[str, str]] = [
        ("Authorization", f"Bearer {OPENAI_API_KEY}"),
        ("OpenAI-Beta", "realtime=v1"),
    ]

    ws = await websockets.connect(
        url,
        additional_headers=headers,  # <— wichtig
        max_size=None,
        compression=None,
    )
    return ws


# -----------------------------------------------------------------------------
# WebSocket-Brücke: Twilio Media Stream  ⇄  OpenAI Realtime
# -----------------------------------------------------------------------------
@app.websocket("/twilio-media-stream")
async def twilio_media_stream(ws: WebSocket):
    await ws.accept()
    log.info("telefon-app: Twilio WS verbunden")

    openai_ws: Optional[websockets.WebSocketClientProtocol] = None
    twilio_stream_sid: Optional[str] = None

    # Fallback-VAD Status
    loop = asyncio.get_running_loop()
    last_audio_ts = 0.0
    buffer_open = False
    silence_task = None

    # Empfangs-/Sende-Tasks
    async def _openai_to_twilio():
        """Pumpt Audio-Delta von OpenAI zu Twilio (Caller)."""
        nonlocal twilio_stream_sid
        try:
            while True:
                raw = await openai_ws.recv()
                if isinstance(raw, bytes):
                    # Normalerweise bekommen wir JSON-Text; bytes ignorieren.
                    continue
                obj = json.loads(raw)
                t = (obj.get("type") or obj.get("event") or "").lower()

                # Mögliche Audio-Delta-Varianten robust behandeln
                audio_b64 = None
                if t in (
                    "response.audio.delta",
                    "response.output_audio.delta",
                    "output_audio.delta",
                    "output_audio_chunk",
                    "response.delta",
                ):
                    if isinstance(obj.get("audio"), str):
                        audio_b64 = obj["audio"]
                    elif isinstance(obj.get("delta"), dict) and isinstance(obj["delta"].get("audio"), str):
                        audio_b64 = obj["delta"]["audio"]

                if audio_b64 and twilio_stream_sid:
                    # Audio an Twilio zurückspielen
                    await ws.send_text(json.dumps({
                        "event": "media",
                        "streamSid": twilio_stream_sid,
                        "media": {"payload": audio_b64}
                    }))

                # (Optional) Logging bei abgeschlossenen Antworten
                if t in ("response.completed", "response.done"):
                    log.debug("OpenAI: response completed")

        except websockets.ConnectionClosed:
            log.info("OpenAI WS geschlossen (openai_to_twilio)")
        except Exception as e:
            log.exception("Fehler in openai_to_twilio: %s", e)

    async def _twilio_to_openai():
        """Pumpt eingehendes Caller-Audio zu OpenAI (input_audio_buffer)."""
        nonlocal twilio_stream_sid, last_audio_ts, buffer_open
        try:
            while True:
                msg = await ws.receive_text()
                data = json.loads(msg)
                event = data.get("event")

                if event == "start":
                    start = data.get("start") or {}
                    twilio_stream_sid = start.get("streamSid")
                    log.info("telefon-app: WS connected: %s", {"event": event, **{k: start.get(k) for k in ("protocol", "version") if start.get(k)}})
                    log.info("telefon-app: Stream gestartet: streamSid=%s callSid=%s", twilio_stream_sid, start.get("callSid"))

                elif event == "media":
                    media = data.get("media") or {}
                    payload = media.get("payload")
                    if payload:
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": payload  # Twilio base64 PCMU
                        }))
                        # Fallback-VAD: Wir haben gerade Audio erhalten
                        buffer_open = True
                        last_audio_ts = loop.time()

                elif event == "mark":
                    # Ignorieren oder für Debug nutzen
                    pass

                elif event == "stop":
                    log.info("telefon-app: Stream gestoppt: %s", data.get("stop") or {})
                    break

        except WebSocketDisconnect:
            log.info("Twilio WS getrennt (twilio_to_openai)")
        except Exception as e:
            log.exception("Fehler in twilio_to_openai: %s", e)

    async def _commit_on_silence():
        """Sehr kurze Stille ⇒ wir committen & fordern Antwort an (Fallback)."""
        nonlocal last_audio_ts, buffer_open
        try:
            while True:
                await asyncio.sleep(0.1)
                if buffer_open and last_audio_ts > 0:
                    ms_since = (loop.time() - last_audio_ts) * 1000.0
                    if ms_since >= INACTIVITY_COMMIT_MS:
                        # Abschluss des Eingabe-Buffers + Antwort anfordern
                        await openai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                        await openai_ws.send(json.dumps({
                            "type": "response.create",
                            "response": {"modalities": ["audio"]}
                        }))
                        buffer_open = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.exception("Fehler in commit_on_silence: %s", e)

    try:
        # Verbindung zu OpenAI
        try:
            openai_ws = await connect_openai_ws()
            log.info("telefon-app: OpenAI WS verbunden")
        except Exception as e:
            log.error("telefon-app: OpenAI WS Verbindung fehlgeschlagen: %s", e, exc_info=True)
            await ws.close()
            return

        # Session konfigurieren: Stimme, Formate, schnelle VAD
        session_update = {
            "type": "session.update",
            "session": {
                "voice": OPENAI_VOICE,
                "modalities": ["audio"],
                "input_audio_format": INPUT_AUDIO_FORMAT,
                "output_audio_format": OUTPUT_AUDIO_FORMAT,
                # Serverseitige Turn-Detection mit kurzer Stille (zusätzlich zu unserem Fallback)
                "turn_detection": {
                    "type": "server_vad",
                    "silence_duration_ms": INACTIVITY_COMMIT_MS
                }
            }
        }
        await openai_ws.send(json.dumps(session_update))

        # >>> NEU: Sofortige Begrüßung abspielen
        await openai_ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["audio"],
                "instructions": "Hallo! Ich bin Ihre KI am Telefon. Wie kann ich helfen?"
            }
        }))

        # Tasks starten
        silence_task = asyncio.create_task(_commit_on_silence())
        to_openai = asyncio.create_task(_twilio_to_openai())
        to_twilio = asyncio.create_task(_openai_to_twilio())

        # Warten, bis eine Seite fertig ist
        done, pending = await asyncio.wait(
            {to_openai, to_twilio},
            return_when=asyncio.FIRST_COMPLETED
        )
        for p in pending:
            p.cancel()

    except Exception as e:
        log.exception("Unerwarteter Fehler in twilio_media_stream: %s", e)
    finally:
        # Tasks sauber beenden
        try:
            if silence_task:
                silence_task.cancel()
        except Exception:
            pass

        # Verbindungen schließen
        try:
            if openai_ws:
                await openai_ws.close()
        except Exception:
            pass

        try:
            await ws.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Lokales Starten
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
