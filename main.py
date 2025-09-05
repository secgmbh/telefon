
import os
import json
import asyncio
import base64
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
import uvicorn
import websockets

# ====== ENV ======
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

OPENAI_REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")
TWILIO_VOICE = os.environ.get("TWILIO_VOICE", "verse")
REALTIME_SYSTEM_PROMPT = os.environ.get("REALTIME_SYSTEM_PROMPT", (
    "Du bist eine deutschsprachige Telefonassistenz. "
    "Sprich akzentfrei auf Hochdeutsch (de-DE), klare Artikulation, kurze Sätze, natürliches Tempo. "
    "Benutze deutsche Zahlen- und Datumsformate. Bei Unsicherheit: kurz nachfragen."
))
TWIML_GREETING = (os.environ.get("TWIML_GREETING") or "").strip()
STREAM_WS_URL_ENV = (os.environ.get("STREAM_WS_URL") or "").strip()

# ===== FastAPI app =====
app = FastAPI(title="Twilio ↔ OpenAI Realtime Bridge (GA)")

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

@app.get("/")
async def root():
    return PlainTextResponse("service: ok\nendpoints: /healthz, /incoming-call, /telefon_live, /media-stream")

@app.head("/")
async def root_head():
    return PlainTextResponse("ok")

def _twiml(ws_url: str, greeting: Optional[str]) -> str:
    if greeting:
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say language="de-DE">{greeting}</Say>
  <Connect><Stream url="{ws_url}"/></Connect>
</Response>"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect><Stream url="{ws_url}"/></Connect>
</Response>"""

def _infer_ws_url_from_request(request: Request) -> str:
    # Prefer explicit ENV override if present
    if STREAM_WS_URL_ENV:
        return STREAM_WS_URL_ENV
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    scheme = "wss"
    return f"{scheme}://{host}/media-stream"

@app.post("/incoming-call")
async def incoming_call(request: Request):
    ws_url = _infer_ws_url_from_request(request)
    twiml = _twiml(ws_url, TWIML_GREETING if TWIML_GREETING else None)
    return HTMLResponse(content=twiml, media_type="application/xml")

@app.post("/telefon_live")
async def telefon_live(request: Request):
    ws_url = _infer_ws_url_from_request(request)
    twiml = _twiml(ws_url, TWIML_GREETING if TWIML_GREETING else None)
    return HTMLResponse(content=twiml, media_type="application/xml")


class CallSession:
    """Bridge Twilio Media Streams (μ-law/8kHz) with OpenAI Realtime (GA)."""
    def __init__(self, twilio_ws: WebSocket):
        self.twilio_ws = twilio_ws
        self.oai_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.closed = False
        self.oai_ready = False
        self.twilio_stream_sid: Optional[str] = None
        self.assistant_speaking = False
        self._barge_in_triggered = False

    async def _connect_openai(self):
        url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
        headers = [("Authorization", f"Bearer {OPENAI_API_KEY}")]
        try:
            self.oai_ws = await websockets.connect(
                url,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=20,
                max_size=10 * 1024 * 1024,
            )
        except TypeError:
            # websockets>=13 renamed 'extra_headers' to 'additional_headers'
            self.oai_ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=20,
                max_size=10 * 1024 * 1024,
            )
        session_update = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": OPENAI_REALTIME_MODEL,
                "instructions": REALTIME_SYSTEM_PROMPT,
                "output_modalities": ["audio"],
                "input_audio_transcription": {"enabled": True, "language": "de"},
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcmu"},
                        "turn_detection": {"type": "server_vad"}
                    },
                    "output": {
                        "format": {"type": "audio/pcmu"},
                        "voice": TWILIO_VOICE
                    }
                },
            }
        }
        await self._oai_send_json(session_update)

    async def _oai_send_json(self, payload: dict):
        assert self.oai_ws is not None
        await self.oai_ws.send(json.dumps(payload))

    async def _oai_send_audio_chunk(self, base64_ulaw_payload: str):
        # Twilio media payloads are already base64 μ-law
        assert self.oai_ws is not None
        await self._oai_send_json({
            "type": "input_audio_buffer.append",
            "audio": base64_ulaw_payload
        })

    async def _oai_finish_input(self):
        await self._oai_send_json({"type": "input_audio_buffer.commit"})

    async def _twilio_clear(self):
        """Tell Twilio to clear any buffered outbound audio (barge-in)."""
        try:
            if self.twilio_stream_sid:
                await self.twilio_ws.send_text(json.dumps({
                    "event": "clear",
                    "streamSid": self.twilio_stream_sid
                }))
        except Exception as e:
            print("Twilio clear error:", e)

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
                        # Drop audio until session.updated acknowledged
                        continue
                    media = data.get("media", {})
                    payload = media.get("payload")
                    if payload:
                        # Barge-in: if assistant is speaking and caller speaks, stop playback & cancel
                        if self.assistant_speaking and not self._barge_in_triggered:
                            await self._twilio_clear()
                            try:
                                await self._oai_send_json({"type": "response.cancel"})
                            except Exception as e:
                                print("response.cancel failed:", e)
                            self.assistant_speaking = False
                            self._barge_in_triggered = True
                        try:
                            await self._oai_send_audio_chunk(payload)
                        except Exception as e:
                            print("append failed:", e)
                            self.closed = True
                            break
                elif event == "mark":
                    # If you send Twilio <Mark> you can commit here; optional
                    try:
                        await self._oai_finish_input()
                    except Exception:
                        pass
                elif event == "stop":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            self.closed = True

    async def _pipe_openai_to_twilio(self):
        assert self.oai_ws is not None
        while not self.closed:
            try:
                async for raw in self.oai_ws:
                    try:
                        evt = json.loads(raw)
                    except Exception:
                        continue
                    mtype = evt.get("type")

                    if mtype == "session.updated":
                        self.oai_ready = True
                    elif mtype == "response.created":
                        self._barge_in_triggered = False
                    elif mtype == "response.output_audio.delta":
                        # GA: 'delta' carries base64 μ-law chunk
                        delta = evt.get("delta")
                        if delta:
                            self.assistant_speaking = True
                            await self.twilio_ws.send_text(json.dumps({
                                "event": "media",
                                "media": {"payload": delta}
                            }))
                    elif mtype == "response.completed":
                        self.assistant_speaking = False
                        await self.twilio_ws.send_text(json.dumps({
                            "event": "mark", "mark": {"name": "oai_response_end"}
                        }))
                    elif mtype == "response.cancelled":
                        self.assistant_speaking = False
                    elif mtype == "error":
                        print("OpenAI error:", evt.get("error"))
            except Exception as e:
                print("OpenAI ws closed/error:", repr(e))
                # Try to reconnect if Twilio is still connected
                try:
                    if not self.closed:
                        await self._connect_openai()
                        continue
                except Exception as e2:
                    print("Reconnect failed:", e2)
                    self.closed = True
                    break
        # end while

    async def run(self):
        await self._connect_openai()
        await asyncio.gather(
            self._pipe_twilio_to_openai(),
            self._pipe_openai_to_twilio(),
        )
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
