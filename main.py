
import os
import json
import asyncio
import base64
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
import uvicorn
import websockets

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

# ===== FastAPI app =====
app = FastAPI(title="Twilio ↔ OpenAI Realtime Bridge (GA)")

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

@app.post("/incoming-call")
async def incoming_call():
    """Twilio will hit this endpoint first. We respond with TwiML that
    connects the call to our /media-stream websocket using Twilio Media Streams.
    """
    # Note: Twilio will replace WS URL with yours. Ensure it's publicly reachable (wss).
    stream_ws_url = os.environ.get("STREAM_WS_URL", "wss://your-domain.example/media-stream")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Willkommen bei Wowona Live. Einen Moment, ich verbinde Sie mit dem Assistenten.</Say>
  <Connect>
    <Stream url="{stream_ws_url}"/>
  </Connect>
</Response>"""
    return HTMLResponse(content=twiml, media_type="application/xml")

@app.post("/telefon_live")
async def telefon_live():
    # Alias-Endpoint für bestehende Twilio-Konfigurationen
    return await incoming_call()


class CallSession:
    """
    Bridges audio between Twilio Media Streams (8kHz G.711 μ-law) and
    OpenAI Realtime over WebSocket.

    - Uses current GA Realtime schema (nested `audio` block).
    - Sends/receives μ-law (audio/pcmu) both directions.
    - Forwards response.output_audio.delta chunks back to Twilio.
    """

    def __init__(self, twilio_ws: WebSocket):
        self.twilio_ws = twilio_ws
        self.oai_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.closed = False
        self.oai_ready = False
        self.twilio_stream_sid = None

    async def _connect_openai(self):
        url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
        headers = [("Authorization", f"Bearer {OPENAI_API_KEY}")]
        # No beta headers or custom subprotocols.
        self.oai_ws = await websockets.connect(
            url,
            extra_headers=headers,
            ping_interval=20,
            ping_timeout=20,
            max_size=10 * 1024 * 1024,
        )
        # Configure session with GA schema
        session_update = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": "gpt-realtime",
                "output_modalities": ["audio"],
                "instructions": (
                    "Du bist ein freundlicher deutschsprachiger Telefonassistent von Wowona. "
                    "Begrüße kurz und hilf knapp und präzise. "
                    "Falls du etwas nicht sicher weißt, frage nach."
                ),
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcmu"},
                        "turn_detection": {"type": "server_vad"}
                    },
                    "output": {
                        "format": {"type": "audio/pcmu"},
                        "voice": "alloy"
                    }
                },
            }
        }
        await self._oai_send_json(session_update)

        # Optional: send a short greeting proactively
        await self._oai_send_json({"type": "response.create"})

    async def _oai_send_json(self, payload: dict):
        assert self.oai_ws is not None
        await self.oai_ws.send(json.dumps(payload))

    async def _oai_send_audio_chunk(self, base64_ulaw_payload: str):
        # Twilio media payloads already base64-encoded μ-law.
        assert self.oai_ws is not None
        await self._oai_send_json({
            "type": "input_audio_buffer.append",
            "audio": base64_ulaw_payload
        })

    async def _oai_finish_input(self):
        # With server VAD this is optional, but safe to call on Twilio "mark" or pauses
        await self._oai_send_json({"type": "input_audio_buffer.commit"})

    async def _pipe_twilio_to_openai(self):
        try:
            while not self.closed:
                msg = await self.twilio_ws.receive_text()
                data = json.loads(msg)
                event = data.get("event")

                if event == "start":
                    info = data.get("start", {})
                    self.twilio_stream_sid = info.get("streamSid")
                
                elif event == "media":
                    if not self.oai_ready:
                        # Drop audio until OpenAI session is updated
                        continue
                    media = data.get("media", {})
                    payload = media.get("payload")
                    if payload:
                        try:
                            await self._oai_send_audio_chunk(payload)
                        except Exception as e:
                            # Socket probably closed; mark session
                            self.closed = True
                            break
                elif event == "mark":
                    # Use marks to force commit
                    await self._oai_finish_input()
                elif event == "stop":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            self.closed = True

    async def _pipe_openai_to_twilio(self):
        assert self.oai_ws is not None
        try:
            async for raw in self.oai_ws:
                try:
                    evt = json.loads(raw)
                except Exception:
                    continue
                mtype = evt.get("type")

                if mtype == "session.updated":
                    self.oai_ready = True
                elif mtype == "response.output_audio.delta":
                    # New GA event field name: 'delta' (base64 μ-law chunk)
                    delta = evt.get("delta")
                    if delta:
                        await self.twilio_ws.send_text(json.dumps({
                            "event": "media",
                            "media": {"payload": delta}
                        }))
                elif mtype == "response.completed":
                    # Optionally tell Twilio about a mark to help with duplex-barge-in
                    await self.twilio_ws.send_text(json.dumps({
                        "event": "mark", "mark": {"name": "oai_response_end"}
                    }))
                elif mtype == "error":
                    err = evt.get("error", {})
                    print("OpenAI error:", err)
        except Exception as e:
            print("OpenAI ws closed/error:", repr(e))
        finally:
            self.closed = True

    async def run(self):
        await self._connect_openai()
        # Run pumps concurrently
        await asyncio.gather(
            self._pipe_twilio_to_openai(),
            self._pipe_openai_to_twilio(),
        )
        # Cleanup
        try:
            if self.oai_ws:
                await self.oai_ws.close()
        except Exception:
            pass


@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    session = CallSession(ws)
    try:
        await session.run()
    finally:
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
