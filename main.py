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
STREAM_WSS_PATH = os.getenv("TWILIO_STREAM_PATH", "/twilio-stream")
VOICE = os.getenv("TWILIO_VOICE", "alloy")
SYSTEM_PROMPT = os.getenv(
    "REALTIME_SYSTEM_PROMPT",
    "Du bist Maria, eine freundliche, präzise deutschsprachige Assistentin. Antworte kurz und konkret.",
)

TWILIO_SUBPROTOCOL = "audio.stream.twilio.com"  # <- Twilio erwartet dieses Subprotocol

app = FastAPI()

# ---------- μ-law <-> PCM16 @ 8kHz (ohne audioop) ----------
_ULAW_BIAS = 0x84  # 132
_ULAW_CLIP = 32635

def ulaw_decode_bytes(ulaw_bytes: bytes) -> bytes:
    u = np.frombuffer(ulaw_bytes, dtype=np.uint8)
    u = np.bitwise_xor(u, 0xFF)
    sign = (u & 0x80) != 0
    exp = (u >> 4) & 0x07
    mant = u & 0x0F
    magnitude = ((mant.astype(np.int32) << 4) + 0x08) << exp
    magnitude = magnitude + _ULAW_BIAS
    magnitude = magnitude - _ULAW_BIAS
    pcm = magnitude.astype(np.int32)
    pcm[sign] = -pcm[sign]
    pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
    return pcm.tobytes()

def _ulaw_segment_scalar(x: int) -> int:
    if x >= 0x4000: return 7
    if x >= 0x2000: return 6
    if x >= 0x1000: return 5
    if x >= 0x0800: return 4
    if x >= 0x0400: return 3
    if x >= 0x0200: return 2
    if x >= 0x0100: return 1
    return 0

def ulaw_encode_bytes(pcm_bytes: bytes) -> bytes:
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.int32)
    x = np.clip(x, -_ULAW_CLIP, _ULAW_CLIP)
    sign = x < 0
    x = np.abs(x).astype(np.int32)
    x = x + _ULAW_BIAS
    ulaw = np.empty(x.shape[0], dtype=np.uint8)
    for i in range(x.shape[0]):
        xi = x[i]
        seg = _ulaw_segment_scalar(xi)
        mant = (xi >> (seg + 3)) & 0x0F
        u = ((seg << 4) | mant)
        u ^= 0xFF
        if sign[i]:
            u |= 0x80
        ulaw[i] = u
    return ulaw.tobytes()

# ----------------- TwiML endpoint -----------------
@app.api_route("/telefon_live", methods=["GET", "POST"])
async def telefon_live():
    wss_url = os.getenv("TWILIO_STREAM_WSS") or ("wss://YOUR_DOMAIN" + STREAM_WSS_PATH)
    print("TwiML requested. Using stream URL:", wss_url)
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
        self.seen_start = False

    async def open_openai(self):
        url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }
        try:
            print("OAI: connecting →", url)
            self.oai_ws = await websockets.connect(url, extra_headers=headers, ping_interval=20)
            print("OAI: connected")
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
            print("OAI: session.update sent")
        except Exception as e:
            print("OAI: failed to connect or configure →", repr(e))
            raise

    async def start(self):
        await self.open_openai()
        asyncio.create_task(self._pipe_openai_to_twilio())
        self.commit_task = asyncio.create_task(self._auto_commit_loop())
        # Watchdog: falls 10s lang kein 'start' kommt, loggen & schließen
        asyncio.create_task(self._start_watchdog())
        print("Bridge started: OpenAI pipe + auto-commit loop running")

    async def _start_watchdog(self):
        await asyncio.sleep(10)
        if not self.seen_start and not self.closed:
            print("Watchdog: 10s ohne 'start' Event. Prüfe Twilio-Stream-Handshake / Subprotocol / Nummern-WebHook.")
            await self.close()

    async def _auto_commit_loop(self):
        while not self.closed:
            await asyncio.sleep(0.1)
            if not self.last_media_ts:
                continue
            if time.time() - self.last_media_ts > 0.25:
                print("Auto-commit: ~250ms pause detected → committing & requesting response")
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
            print("OAI: commit + response.create sent")
        except Exception as e:
            print("OAI: commit/response error →", repr(e))

    async def receive_from_twilio(self):
        try:
            while True:
                # manche Edges schicken binär – deshalb beide Varianten versuchen
                try:
                    raw = await self.twilio_ws.receive_text()
                except Exception:
                    raw_bytes = await self.twilio_ws.receive_bytes()
                    raw = raw_bytes.decode("utf-8", errors="ignore")

                if len(raw) > 140:
                    print("WS RX (truncated):", raw[:140], "…")
                else:
                    print("WS RX:", raw)

                data = json.loads(raw)
                event = data.get("event")
                if event == "start":
                    self.seen_start = True
                    self.stream_sid = data.get("start", {}).get("streamSid")
                    print("Twilio start, streamSid:", self.stream_sid)
                elif event == "media":
                    ulaw_b64 = data["media"]["payload"]
                    ulaw = base64.b64decode(ulaw_b64)
                    pcm16_8k = ulaw_decode_bytes(ulaw)
                    if self.playing_audio and self.stream_sid:
                        print("Barge-in: clearing playback")
                        await self._twilio_send({"event": "clear", "streamSid": self.stream_sid})
                        self.playing_audio = False
                    await self.oai_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm16_8k).decode(),
                    }))
                    self.last_media_ts = time.time()
                elif event == "stop":
                    print("Twilio stop")
                    await self._commit_and_request_response()
                    break
        except WebSocketDisconnect:
            print("Twilio WS disconnect")
        except Exception as e:
            print("Twilio WS error:", repr(e))
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
                    print("OAI → delta audio (bytes):", len(b64))
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
                    print("OAI → audio done")
                    if self.stream_sid:
                        await self._twilio_send({
                            "event": "mark",
                            "streamSid": self.stream_sid,
                            "mark": {"name": "assistant_turn_done"},
                        })
                        self.playing_audio = False
                elif t and t.startswith("error"):
                    print("OAI error event:", data)
        except Exception as e:
            print("OAI pipe error:", repr(e))

    async def _twilio_send(self, obj: dict):
        try:
            await self.twilio_ws.send_text(json.dumps(obj))
            if obj.get("event") == "media":
                payload = obj.get("media", {}).get("payload", "")
                print("WS TX media (bytes):", len(payload))
            else:
                print("WS TX:", obj)
        except Exception as e:
            print("Twilio send error:", repr(e))

    async def close(self):
        if self.closed:
            return
        self.closed = True
        print("Bridge closing …")
        try:
            if self.commit_task:
                self.commit_task.cancel()
        except Exception as e:
            print("commit_task cancel error:", repr(e))
        try:
            if self.oai_ws:
                await self.oai_ws.close()
        except Exception as e:
            print("OAI close error:", repr(e))
        try:
            await self.twilio_ws.close()
        except Exception as e:
            print("Twilio WS close error:", repr(e))

@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    # Header-Check & Subprotocol-Handshake
    headers_dict = dict(ws.headers)
    print("WS headers:", headers_dict)
    try:
        await ws.accept(subprotocol=TWILIO_SUBPROTOCOL)  # ← Twilio erwartet genau dieses Protokoll
        print("WS: accepted with subprotocol =", TWILIO_SUBPROTOCOL)
    except Exception as e:
        print("WS accept error:", repr(e))
        return

    bridge = Bridge(ws)
    try:
        await bridge.start()
        await bridge.receive_from_twilio()
    except Exception as e:
        print("Bridge error:", repr(e))
        await bridge.close()

# Für lokalen Start:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
