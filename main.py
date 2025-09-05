import os
import json
import time
import base64
import audioop
import asyncio
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from dotenv import load_dotenv
import websockets

load_dotenv()

# --- ENV ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2025-06-03")
STREAM_WSS_PATH = os.getenv("TWILIO_STREAM_PATH", "/twilio-stream")  # must match your TwiML
VOICE = os.getenv("TWILIO_VOICE", "alloy")  # OpenAI voice name (e.g., "alloy")
SYSTEM_PROMPT = os.getenv(
    "REALTIME_SYSTEM_PROMPT",
    "Du bist Maria, eine freundliche, präzise deutschsprachige Assistentin. Antworte kurz und konkret.",
)

app = FastAPI()

# ----------------- TwiML endpoints -----------------
@app.post("/telefon_live")
async def telefon_live():
    # Twilio will fetch this to start the bidirectional stream
    # Make sure your service is reachable via TLS; on Render you'll have HTTPS/WSS by default
    # Build WSS url from host header for convenience
    wss_url = os.getenv("TWILIO_STREAM_WSS")
    if not wss_url:
        # Try to infer from request's host via a dummy placeholder; you can hardcode in .env
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
# Twilio <-> OpenAI Realtime in one connection

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
        # Set session defaults: instructions + request audio output by default
        await self.oai_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": SYSTEM_PROMPT,
                "modalities": ["audio", "text"],
                "voice": VOICE,
                "input_audio_format": {"type": "pcm16", "sample_rate_hz": 16000},
                "output_audio_format": {"type": "pcm16", "sample_rate_hz": 16000},
                "language": "de-DE",
            },
        }))

    async def start(self):
        await self.open_openai()
        # Task: read from OpenAI and send to Twilio
        asyncio.create_task(self._pipe_openai_to_twilio())
        # Task: periodic commit if short pause is detected
        self.commit_task = asyncio.create_task(self._auto_commit_loop())

    async def _auto_commit_loop(self):
        # Commit input when we detect ~250ms Pause → keeps latency <~1s
        while not self.closed:
            await asyncio.sleep(0.1)
            if not self.last_media_ts:
                continue
            if time.time() - self.last_media_ts > 0.25:  # ~250 ms Pause
                await self._commit_and_request_response()
                self.last_media_ts = 0.0

    async def _commit_and_request_response(self):
        if not self.oai_ws:
            return
        try:
            await self.oai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await self.oai_ws.send(json.dumps({"type": "response.create", "response": {"modalities": ["audio", "text"]}}))
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
                    pcm8k = audioop.ulaw2lin(ulaw, 2)  # μ-law -> PCM16 @8kHz
                    pcm16k, _ = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, None)
                    # If caller speaks while we're playing, barge-in: clear any playback
                    if self.playing_audio and self.stream_sid:
                        await self._twilio_send({"event": "clear", "streamSid": self.stream_sid})
                        self.playing_audio = False
                    # Append to OpenAI
                    await self.oai_ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm16k).decode()}))
                    self.last_media_ts = time.time()
                elif event == "stop":
                    # Final commit when stream ends
                    await self._commit_and_request_response()
                    break
        except WebSocketDisconnect:
            pass
        finally:
            await self.close()

    async def _pipe_openai_to_twilio(self):
        # Send OpenAI audio chunks back to Twilio as μ-law 8k media frames
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
                    pcm16 = base64.b64decode(b64)
                    # convert to μ-law 8k for Twilio playback
                    pcm8k, _ = audioop.ratecv(pcm16, 2, 1, 16000, 8000, None)
                    ulaw = audioop.lin2ulaw(pcm8k, 2)
                    if self.stream_sid:
                        await self._twilio_send({
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {"payload": base64.b64encode(ulaw).decode()},
                        })
                        self.playing_audio = True
                elif t in ("response.audio.done", "response.output_audio.done"):
                    # Mark end of assistant turn for debugging
                    if self.stream_sid:
                        await self._twilio_send({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": "assistant_turn_done"}})
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


# --------------- Run (for local dev) ---------------
# On Render, use: gunicorn -k uvicorn.workers.UvicornWorker main:app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
