import os
import json
import logging
import time
import wave
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
# μ-law ⇢ PCM16 (ohne audioop)
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

# =========================================================
# FastAPI App
# =========================================================
app = FastAPI(title="Telefon Live", version="1.1.0")

@app.get("/")
async def root():
    return PlainTextResponse("OK")

@app.get("/health")
async def health():
    return {"status": "ok"}

# =========================================================
# Twilio Voice Webhook -> TwiML
# =========================================================
@app.post("/telefon_live")
async def telefon_live(request: Request):
    """Twilio Voice Webhook (POST form-encoded) -> liefert TwiML mit <Connect><Stream/>."""
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
  <Say language="de-DE">{say_text}</Say>
  <Connect>
    <Stream url="{ws_url}" track="inbound_track" />
  </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")

# =========================================================
# Twilio Media Streams WebSocket
# =========================================================
@app.websocket("/twilio-media-stream")
async def twilio_media_stream(ws: WebSocket):
    await ws.accept()
    log.info("Twilio WS verbunden")

    # Aufnahme-Parameter (optional)
    record_wav = os.getenv("RECORD_WAV", "false").lower() in ("1", "true", "yes", "on")
    max_seconds = int(os.getenv("RECORD_MAX_SECONDS", "60"))
    frames_per_second = 50  # 20ms Frames
    max_frames = max_seconds * frames_per_second

    # Laufzeit-Status
    frame_count = 0
    byte_count = 0
    last_log = time.monotonic()
    stream_sid = None
    call_sid = None

    # Puffer für WAV (PCM16), nur wenn RECORD_WAV aktiv
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

                    # alle ~1s ein Fortschrittslog
                    now = time.monotonic()
                    if now - last_log >= 1.0:
                        # ca. 160 μ-law Bytes je 20ms -> ~8000 B/s
                        log.info("Media: frames=%d, mu-law bytes=%d (~%d B/s)",
                                 frame_count, byte_count, int(byte_count / max(1, now - last_log)))
                        last_log = now
                        byte_count = 0  # nur für die Rate

                    # optional aufnehmen
                    if record_wav and len(pcm_buf) < max_frames * 160 * 2:  # 160 samples/frame -> 320 bytes PCM16
                        pcm_buf += mulaw_to_pcm16(mu)

            elif ev == "mark":
                mark = msg.get("mark", {})
                log.debug("Mark: %s", mark)

            elif ev == "stop":
                info = msg.get("stop", {})
                log.info("Stream gestoppt: %s", info)
                break

            else:
                log.debug("Unbekanntes Event: %s", ev)

    except WebSocketDisconnect:
        log.info("Twilio WS getrennt")
    except Exception as e:
        log.exception("Fehler im Twilio WS: %s", e)
    finally:
        # WAV sichern (falls aktiv)
        if record_wav and pcm_buf and stream_sid:
            try:
                fname = f"/tmp/call_{stream_sid}.wav"
                with wave.open(fname, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)   # PCM16
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
