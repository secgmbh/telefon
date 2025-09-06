# main.py
# FastAPI-Endpunkt für Twilio Media Streams -> OpenAI Realtime
# - Keine Abhängigkeit von audioop (PEP 594)
# - μ-law (G.711) -> PCM16, Resampling 8k -> 24k (linear)
# - Pufferung ≥100 ms vor input_audio_buffer.commit

import os
import json
import base64
import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

import websockets

# ------------------------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
OPENAI_REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"

# Twilio schickt üblicherweise 20ms Frames @ 8 kHz μ-law
TWILIO_SAMPLE_RATE = 8000
OPENAI_SAMPLE_RATE = 24000
MIN_COMMIT_MS = 100  # mindestens 100 ms an OpenAI übergeben, bevor commit

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("realtime")


# ------------------------------------------------------------------------------
# μ-law (G.711) Dekoder -> PCM16 LE @ 8k
# ------------------------------------------------------------------------------
def ulaw_bytes_to_pcm16(ulaw: bytes) -> bytes:
    """
    Dekodiert G.711 μ-law zu PCM16 little-endian (8 kHz).
    Implementiert gemäß ITU G.711, dependency-frei.
    """
    out = bytearray(len(ulaw) * 2)
    for i, b in enumerate(ulaw):
        u = (~b) & 0xFF
        sign = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        magnitude = ((mantissa << 3) + 0x84) << exponent
        sample = magnitude - 0x84
        if sign:
            sample = -sample
        # clamp auf int16
        if sample > 32767:
            sample = 32767
        elif sample < -32768:
            sample = -32768
        out[2 * i] = sample & 0xFF
        out[2 * i + 1] = (sample >> 8) & 0xFF
    return bytes(out)


# ------------------------------------------------------------------------------
# Resampling 8 kHz PCM16 LE -> 24 kHz PCM16 LE (Faktor 3, linear)
# ------------------------------------------------------------------------------
def _pcm16_iter(samples_le: bytes):
    """Erzeugt int16 Samples aus little-endian Bytes."""
    for i in range(0, len(samples_le), 2):
        yield (samples_le[i] | (samples_le[i + 1] << 8)) - (
            65536 if (samples_le[i + 1] & 0x80) else 0
        )


def _int16_to_le_bytes(x: int) -> bytes:
    if x > 32767:
        x = 32767
    elif x < -32768:
        x = -32768
    return bytes(((x & 0xFF), ((x >> 8) & 0xFF)))


def upsample_linear_8k_to_24k(pcm8k_le: bytes) -> bytes:
    """
    Lineare Interpolation von 8 kHz nach 24 kHz (Faktor 3).
    Für jedes Paar (s0, s1) werden drei Samples erzeugt:
    s0, s0 + 1/3*(Δ), s0 + 2/3*(Δ). Letztes Sample wird wiederholt.
    """
    it = list(_pcm16_iter(pcm8k_le))
    n = len(it)
    if n == 0:
        return b""

    out = bytearray()
    for i in range(n - 1):
        s0 = it[i]
        s1 = it[i + 1]
        delta = s1 - s0
        # drei gleichmäßig verteilte Samples
        out += _int16_to_le_bytes(s0)
        out += _int16_to_le_bytes(int(s0 + delta / 3))
        out += _int16_to_le_bytes(int(s0 + 2 * delta / 3))
    # Letztes Ursprungs-Sample: dreifach (ZOH am Ende)
    last = it[-1]
    out += _int16_to_le_bytes(last)
    out += _int16_to_le_bytes(last)
    out += _int16_to_le_bytes(last)
    return bytes(out)


# ------------------------------------------------------------------------------
# OpenAI Realtime WebSocket Client
# ------------------------------------------------------------------------------
class OpenAIRealtimeClient:
    def __init__(self):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._buffer24k = bytearray()
        self._first_response_requested = False

    async def connect(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("Umgebungsvariable OPENAI_API_KEY fehlt.")
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }
        self.ws = await websockets.connect(OPENAI_REALTIME_URL, extra_headers=headers)
        log.info("OpenAI Realtime WebSocket verbunden: %s", OPENAI_REALTIME_MODEL)

        # Hintergrund-Task: Antworten/Ereignisse lesen (nur Logging)
        asyncio.create_task(self._read_events())

    async def close(self):
        if self.ws:
            try:
                await self.commit()  # Rest committen
            except Exception:
                pass
            await self.ws.close()
            log.info("OpenAI Realtime WebSocket geschlossen")

    async def _read_events(self):
        # Einfach alles lesen und ins Log schreiben
        try:
            assert self.ws is not None
            async for msg in self.ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    data = msg
                log.debug("OAI ⇠ %s", data)
        except Exception as e:
            log.warning("OAI Reader beendet: %s", e)

    async def append_pcm24k(self, chunk24k_le: bytes):
        """PCM16 LE @ 24kHz anhängen. Committen, wenn ≥ 100 ms im Puffer."""
        self._buffer24k += chunk24k_le
        # 100 ms @ 24 kHz, 16 Bit mono => 4800 Bytes
        if len(self._buffer24k) >= int(OPENAI_SAMPLE_RATE * 0.1 * 2):
            await self.commit()

    async def commit(self):
        """Puffer (falls vorhanden) an OAI senden und committen."""
        if not self._buffer24k:
            return
        assert self.ws is not None

        # Base64 für input_audio_buffer.append
        b64 = base64.b64encode(self._buffer24k).decode("ascii")
        append_msg = {
            "type": "input_audio_buffer.append",
            "audio": b64,
        }
        await self.ws.send(json.dumps(append_msg))
        await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        log.debug(
            "OAI ⇢ commit: %.1f ms",
            (len(self._buffer24k) / 2) / OPENAI_SAMPLE_RATE * 1000.0,
        )
        self._buffer24k.clear()

        # Optional: beim ersten Commit sofort eine Antwort anfordern
        if not self._first_response_requested:
            self._first_response_requested = True
            await self.ws.send(json.dumps({"type": "response.create"}))


# ------------------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------------------
app = FastAPI()


@app.get("/")
async def root():
    return JSONResponse({"status": "ok", "model": OPENAI_REALTIME_MODEL})


@app.websocket("/twilio-media-stream")
async def twilio_ws(ws: WebSocket):
    # Twilio verbindet per WebSocket und sendet JSON-Events (start, media, stop, ...)
    await ws.accept()
    log.info("Twilio WebSocket verbunden")

    oai = OpenAIRealtimeClient()
    try:
        await oai.connect()

        while True:
            try:
                text = await ws.receive_text()
            except WebSocketDisconnect:
                log.info("Twilio WebSocket getrennt")
                break

            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                log.warning("Ungültiges JSON von Twilio: %r", text[:200])
                continue

            ev = event.get("event")
            if ev == "start":
                stream_sid = event.get("start", {}).get("streamSid")
                log.info("Twilio start: streamSid=%s", stream_sid)

            elif ev == "media":
                payload_b64 = event.get("media", {}).get("payload", "")
                if not payload_b64:
                    continue
                try:
                    ulaw = base64.b64decode(payload_b64)
                except Exception:
                    log.warning("Base64-Decode fehlgeschlagen")
                    continue

                # μ-law -> PCM16 @ 8k
                pcm8k = ulaw_bytes_to_pcm16(ulaw)
                # 8k -> 24k (linear)
                pcm24k = upsample_linear_8k_to_24k(pcm8k)
                # an OAI anhängen (commit automatisch ab ≥100ms)
                await oai.append_pcm24k(pcm24k)

            elif ev == "mark":
                # Optional: Marker/DTMF ignorieren oder loggen
                pass

            elif ev == "stop":
                log.info("Twilio stop erhalten")
                break

            else:
                # andere Events stillschweigend akzeptieren
                pass

    except Exception as e:
        log.exception("Fehler in /twilio-media-stream: %s", e)
    finally:
        await oai.close()
        try:
            await ws.close()
        except Exception:
            pass
        log.info("Twilio-Session beendet")
