import os
import json
import logging
from urllib.parse import parse_qs
import base64

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
# FastAPI App
# =========================================================
app = FastAPI(title="Telefon Live", version="1.0.0")

# =========================================================
# Health & Root
# =========================================================
@app.get("/")
async def root():
    return PlainTextResponse("OK")

@app.get("/health")
async def health():
    return {"status": "ok"}

# =========================================================
# Twilio Voice Webhook → TwiML: <Connect><Stream/>
#   - Endpoint, den du in Twilio als Voice Webhook hinterlegst
#   - Antwortet mit TwiML, das den Call auf unseren WS streamt
# =========================================================
@app.post("/telefon_live")
async def telefon_live(request: Request):
    """
    Twilio sendet hier application/x-www-form-urlencoded.
    Wir parsen minimal (ohne python-multipart) und liefern TwiML zurück.
    """
    try:
        raw_body = (await request.body()).decode("utf-8", "ignore")
        form = {k: v[0] if isinstance(v, list) and v else v for k, v in parse_qs(raw_body).items()}
        log.info("Twilio Voice webhook payload: %s", form)
    except Exception as e:
        log.warning("Konnte Formdaten nicht parsen: %s", e)

    # WS-Ziel bestimmen (ENV bevorzugt). Fallback: aus Request-URL abgeleitet.
    ws_url = os.environ.get("TWILIO_WS_URL")
    if not ws_url:
        base = str(request.url).split("/telefon_live")[0]
        if base.startswith("https://"):
            ws_url = base.replace("https://", "wss://") + "/twilio-media-stream"
        else:
            ws_url = base.replace("http://", "ws://") + "/twilio-media-stream"

    # Kleine deutsche Ansage + Stream-Connect
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say language="de-DE">Verbunden. Einen Moment bitte.</Say>
  <Connect>
    <Stream url="{ws_url}" track="inbound_track" />
  </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")

# =========================================================
# Twilio Media Streams WebSocket
#   - Twilio sendet JSON-Frames mit event: "connected" | "start" | "media" | "mark" | "stop"
#   - Audio kommt als Base64 μ-law (@8kHz) in media.payload in 20ms Frames
#   - Hier: minimaler Handler; Stelle markiert, wo du Audio/STT einbindest
# =========================================================
@app.websocket("/twilio-media-stream")
async def twilio_media_stream(ws: WebSocket):
    await ws.accept()
    log.info("Twilio WS verbunden")

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            ev = msg.get("event")

            if ev == "connected":
                log.info("WS connected: %s", msg)

            elif ev == "start":
                info = msg.get("start", {})
                log.info("Stream gestartet: streamSid=%s callSid=%s",
                         info.get("streamSid"), info.get("callSid"))

            elif ev == "media":
                media = msg.get("media", {})
                b64 = media.get("payload")
                if b64:
                    # μ-law 8kHz Bytes (20ms) — hier könntest du dekodieren/weiterreichen
                    # payload_bytes = base64.b64decode(b64)
                    # → an STT/Realtime weiterleiten (ggf. vorher μ-law->PCM16 & resampling)
                    pass

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
        try:
            await ws.close()
        except Exception:
            pass

# =========================================================
# (Optional) μ-law → PCM16 (ohne 'audioop')
#   - Aktuell nicht verwendet; praktisch falls du Audio transkodieren willst
# =========================================================
MU_LAW_EXPAND_TABLE = []

def _build_mulaw_table():
    tbl = []
    # ITU G.711 μ-law Expand
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
    """Konvertiert μ-law (8 kHz, 8-bit) → PCM16 little-endian."""
    global MU_LAW_EXPAND_TABLE
    if not MU_LAW_EXPAND_TABLE:
        MU_LAW_EXPAND_TABLE[:] = _build_mulaw_table()
    out = bytearray()
    for b in mu_bytes:
        s = MU_LAW_EXPAND_TABLE[b]
        out += int(s).to_bytes(2, "little", signed=True)
    return bytes(out)

# =========================================================
# Lokaler Start (optional). Auf Render nutzt du gunicorn.
# =========================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
