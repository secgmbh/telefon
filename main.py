import os
import json
import time
import base64
import asyncio
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from dotenv import load_dotenv
import websockets

load_dotenv()

# --- ENV ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2025-06-03")
STREAM_WSS_PATH = os.getenv("TWILIO_STREAM_PATH", "/twilio-stream")  # muss zu deiner TwiML passen
VOICE = os.getenv("TWILIO_VOICE", "alloy")  # OpenAI Realtime voice
SYSTEM_PROMPT = os.getenv(
    "REALTIME_SYSTEM_PROMPT",
    "Du bist Maria, eine freundliche, präzise deutschsprachige Assistentin. Antworte kurz und konkret.",
)

app = FastAPI()

# ---------- μ-law <-> PCM16 @ 8kHz (ohne audioop) ----------
_ULAW_BIAS = 0x84  # 132
_ULAW_CLIP = 32635

def ulaw_decode_bytes(ulaw_bytes: bytes) -> bytes:
    """Decode G.711 μ-law (8-bit) -> PCM16 little-endian (int16)"""
    u = np.frombuffer(ulaw_bytes, dtype=np.uint8)
    u = np.bitwise_xor(u, 0xFF)

    sign = (u & 0x80) != 0
    exp = (u >> 4) & 0x07
    mant = u & 0x0F

    # sample = ((mant << 4) + 0x08) << exp + 0x84  (anschließend -0x84, mit Vorzeichen)
    magnitude = ((mant.astype(np.int32) << 4) + 0x08) << exp
    magnitude = magnitude + _ULAW_BIAS
    magnitude = magnitude - _ULAW_BIAS

    pcm = magnitude.astype(np.int32)
    pcm[sign] = -pcm[sign]
    pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
    return pcm.tobytes()

def _ulaw_segment_scalar(x: int) -> int:
    """Bestimme Segment (0..7) für μ-law, scalar."""
    # x >= 0, already biased
    if x >= 0x4000: return 7
    if x >= 0x2000: return 6
    if x >= 0x1000: return 5
    if x >= 0x0800: return 4
    if x >= 0x0400: return 3
    if x >= 0x0200: return 2
    if x >= 0x0100: return 1
    return 0

def ulaw_encode_bytes(pcm_bytes: bytes) -> bytes:
    """Encode PCM16 little-endian (int16) -> G.711 μ-law (8-bit)"""
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.int32)
    # Clamp
    x = np.clip(x, -_ULAW_CLIP, _ULAW_CLIP)
    # Sign & magnitude
    sign = x < 0
    x = np.abs(x).astype(np.int32)
    x = x + _ULAW_BIAS
    # Segment + mantissa per sample (scalar loop; Twilio frames sind klein, loop ist ok)
    ulaw = np.empty(x.shape[0], dtype=np.uint8)
    for i in range(x.shape[0]):
        xi = x[i]
        seg = _ulaw_segment_scalar(xi)
        mant = (xi >> (seg + 3)) & 0x0F
        u = ((seg << 4) | mant)
        u ^= 0xFF  # complement
        if sign[i]:
            u |= 0x80
        ulaw[i] = u
    return ulaw.tobytes()

# ----------------- TwiML endpoints -----------------
@app.post("/telefon_live")
async def telefon_live():
    # Twilio ruft diesen Webhook an (Voice: A call comes in -> POST)
    wss_url = os.getenv("TWILIO_STREAM_WSS")
    if not wss_url:
        # Fallback – bitte in .env setzen
        wss_url = "wss://YOUR_DOMAIN" + STREAM_WSS_PATH

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say language="de-DE" voice="Polly.Vicki-Neural">Willkommen bei wowona Live. Einen Moment bitte…</Say>
  <Connect>
    <Stream url="{wss_url}" />
  </Connect>
</Response>"""
    return Response(content=xml, media_type="text/xml")

@app.get("/")
async def root():
    return Response(content="OK", media_type="text/plain")

# ----------------- WebSocket Bridge -----------------
class Bridge:
    def __init__(self, twilio_ws: WebSocket):
        self.twilio_ws = twilio_ws
        self.oai_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.stream_sid: Optional[str] = None
        self.last_media_ts: float = 0.0
        self.commit_task: Optional[asyncio.Task] = None
        self.playing_audio: bool = False
        self.closed = False

    async def open_openai(self):
        url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }
        self.oai_ws = await websockets.connect(url, extra_headers=headers, ping_interval=20)
        # Session: 8 kHz PCM16 für Ein- und Ausgabe -> kein Resampling nötig
        await self.oai_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": SYSTEM_PROMPT,
                "modalities": ["audio", "text"],
                "voice": VOICE,
                "input_audio_format": {"type": "pcm16", "sample_rate_hz": 8000},
                "output_audio_format": {"type": "pcm16", "sample_rate_hz": 8000},
                "language": "de-DE",
            },
        }))

    async def start(self):
        await self.open_openai()
        asyncio.create_task(self._pipe_openai_to_twilio())
        self.commit_task = asyncio.create_task(self._auto_commit_loop())

    async def _auto_commit_loop(self):
        # Commit nach ~250 ms Stille -> niedrige Latenz
        while not self.closed:
            await asyncio.sleep(0.1)
            if not self.last_media_ts:
                continue
            if time.time() - self.last_media_ts > 0.25:
                await self._commit_and_request_response()
                self.last_media_ts = 0.0

    async def _commit_and_request_response(self):
        if not self.oai_ws:
            return
        try:
            await self.oai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await self.oai_ws.send(json.dumps({
                "type": "response.create",
                "response": {"modalities": ["audio", "text"]}
            }))
        except Exception:
            pass

    async def receive_from_twilio(self):
        try:
            while True:
                raw = await self.twilio_ws.receive_text()
                data = json.loads(raw)
                event = data.get("event")
                if event == "start":
                    self.stream_sid = data.get("start", {}).get("streamSid")
                elif event == "media":
                    ulaw_b64 = data["media"]["payload"]
                    ulaw = base64.b64decode(ulaw_b64)
                    pcm16_8k = ulaw_decode_bytes(ulaw)
                    # Barge-in: falls wir gerade spielen, Playback abbrechen
                    if self.playing_audio and self.stream_sid:
                        await self._twilio_send({"event": "clear", "streamSid": self.stream_sid})
                        self.playing_audio = False
                    # an OpenAI anhängen
                    await self.oai_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm16_8k).decode(),
                    }))
                    self.last_media_ts = time.time()
                elif event == "stop":
                    await self._commit_and_request_response()
                    break
        except WebSocketDisconnect:
            pass
        finally:
            await self.close()

    async def _pipe_openai_to_twilio(self):
        if not self.oai_ws:
            return
        try:
            async for msg in self.oai_ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                t = data.get("type")
                if t in ("response.audio.delta", "response.output_audio.delta"):
                    b64 = data.get("delta") or data.get("audio")
                    if not b64:
                        continue
                    pcm16_8k = base64.b64decode(b64)
                    ulaw = ulaw_encode_bytes(pcm16_8k)
                    if self.stream_sid:
                        await self._twilio_send({
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {"payload": base64.b64encode(ulaw).decode()},
                        })
                        self.playing_audio = True
                elif t in ("response.audio.done", "response.output_audio.done"):
                    if self.stream_sid:
                        await self._twilio_send({
                            "event": "mark",
                            "streamSid": self.stream_sid,
                            "mark": {"name": "assistant_turn_done"},
                        })
                        self.playing_audio = False
        except Exception:
            pass

    async def _twilio_send(self, obj: dict):
        try:
            await self.twilio_ws.send_text(json.dumps(obj))
        except Exception:
            pass

    async def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.commit_task:
                self.commit_task.cancel()
        except Exception:
            pass
        try:
            if self.oai_ws:
                await self.oai_ws.close()
        except Exception:
            pass
        try:
            await self.twilio_ws.close()
        except Exception:
            pass

@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    bridge = Bridge(ws)
    await bridge.start()
    await bridge.receive_from_twilio()

# Für lokalen Start:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
