import os
import json
import asyncio
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
import uvicorn
import websockets

# ========= ENV =========
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY")

OPENAI_REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")
TWILIO_VOICE = os.environ.get("TWILIO_VOICE", "verse")
REALTIME_SYSTEM_PROMPT = os.environ.get(
    "REALTIME_SYSTEM_PROMPT",
    "Antworte ausschließlich auf Deutsch (Hochdeutsch, de-DE). "
    "Sprich klar und kurz. Frage nach, wenn etwas unklar ist."
)
TWIML_GREETING = (os.environ.get("TWIML_GREETING") or "").strip()
STREAM_WS_URL = (os.environ.get("STREAM_WS_URL") or "").strip()
AUTO_GREETING = (os.environ.get("AUTO_GREETING") or "").strip()

# ========= APP =========
app = FastAPI(title="Twilio ↔ OpenAI Realtime")

@app.api_route("/healthz", methods=["GET", "HEAD"])
async def healthz():
    # HEAD und GET -> 200 OK (wichtig für Render-Healthcheck)
    return PlainTextResponse("ok")

@app.get("/")
async def root():
    return PlainTextResponse("service: ok\nendpoints: /healthz, /diag, /incoming-call, /telefon_live, /media-stream, /stream-status")

@app.head("/")
async def root_head():
    return PlainTextResponse("ok")

@app.get("/diag")
async def diag():
    """Schnelltest: Verbindet zu OpenAI Realtime und wartet auf session.created/session.updated."""
    url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
    headers = [("Authorization", f"Bearer {OPENAI_API_KEY}")]
    try:
        try:
            ws = await websockets.connect(url, extra_headers=headers, ping_interval=10, ping_timeout=10)
        except TypeError:
            ws = await websockets.connect(url, additional_headers=headers, ping_interval=10, ping_timeout=10)

        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": OPENAI_REALTIME_MODEL,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {"format": {"type": "audio/pcmu"}, "turn_detection": {"type": "server_vad"}},
                    "output": {"format": {"type": "audio/pcmu"}, "voice": TWILIO_VOICE}
                }
            }
        }))
        # Warte auf erste Server-Antwort
        raw = await asyncio.wait_for(ws.recv(), timeout=8)
        await ws.close()
        return PlainTextResponse("diag: ok - " + raw[:240])
    except Exception as e:
        return PlainTextResponse("diag: failed - " + repr(e), status_code=500)

# ===== TwiML helpers =====
def _twiml(ws_url: str, greeting: Optional[str], status_cb: str) -> str:
    say = f'<Say language="de-DE">{greeting}</Say>\n  ' if greeting else ""
    # statusCallback hilft zu erkennen, warum Twilio evtl. nicht verbindet
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  {say}<Connect>
    <Stream url="{ws_url}" track="both" statusCallback="{status_cb}" statusCallbackMethod="POST"/>
  </Connect>
</Response>"""

def _infer_host(request: Request) -> str:
    return request.headers.get("x-forwarded-host") or request.headers.get("host") or ""

def _infer_ws_url(request: Request) -> str:
    if STREAM_WS_URL:
        return STREAM_WS_URL
    host = _infer_host(request)
    return f"wss://{host}/media-stream"

def _infer_status_cb(request: Request) -> str:
    host = _infer_host(request)
    return f"https://{host}/stream-status"

@app.post("/incoming-call")
async def incoming_call(request: Request):
    ws_url = _infer_ws_url(request)
    status_cb = _infer_status_cb(request)
    return HTMLResponse(_twiml(ws_url, TWIML_GREETING or None, status_cb), media_type="application/xml")

@app.api_route("/telefon_live", methods=["GET", "POST"])
async def telefon_live(request: Request):
    ws_url = _infer_ws_url(request)
    status_cb = _infer_status_cb(request)
    # Logge die eingehenden Twilio-Parameter (hilfreich zum Trace)
    try:
        form = await request.form()
        print("Twilio /telefon_live form:", dict(form))
    except Exception:
        pass
    return HTMLResponse(_twiml(ws_url, TWIML_GREETING or None, status_cb), media_type="application/xml")

@app.api_route("/stream-status", methods=["POST", "GET"])
async def stream_status(request: Request):
    # Twilio schickt x-www-form-urlencoded
    try:
        form = await request.form()
        payload = dict(form)
        print("Twilio StreamStatus:", payload)
        return PlainTextResponse("ok")
    except Exception:
        body = (await request.body()).decode("utf-8", "ignore")
        print("Twilio StreamStatus (raw):", body)
        return PlainTextResponse("ok")

# ========= Bridge =========
class CallSession:
    """Bridge Twilio Media Streams (μ-law/8kHz) ↔ OpenAI Realtime (GA)."""
    def __init__(self, twilio_ws: WebSocket):
        self.twilio_ws = twilio_ws
        self.oai_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.closed = False
        self.oai_ready = False
        self.twilio_stream_sid: Optional[str] = None
        self.assistant_speaking = False
        self._barge_in = False

    async def _ws_connect(self, url: str, headers):
        try:
            return await websockets.connect(url, extra_headers=headers, ping_interval=20, ping_timeout=20, max_size=10*1024*1024)
        except TypeError:
            return await websockets.connect(url, additional_headers=headers, ping_interval=20, ping_timeout=20, max_size=10*1024*1024)

    async def _connect_openai(self):
        url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
        headers = [("Authorization", f"Bearer {OPENAI_API_KEY}")]
        self.oai_ws = await self._ws_connect(url, headers)

        # Session konfigurieren
        session_update = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": OPENAI_REALTIME_MODEL,
                "instructions": REALTIME_SYSTEM_PROMPT,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {"format": {"type": "audio/pcmu"}, "turn_detection": {"type": "server_vad"}},
                    "output": {"format": {"type": "audio/pcmu"}, "voice": TWILIO_VOICE}
                },
            }
        }
        await self.oai_ws.send(json.dumps(session_update))

        # Warte auf session.updated, erst dann Audio weiterleiten
        for _ in range(20):  # ~40s
            raw = await asyncio.wait_for(self.oai_ws.recv(), timeout=2)
            try:
                evt = json.loads(raw)
            except Exception:
                continue
            if evt.get("type") == "session.updated":
                print("OpenAI session.updated received")
                self.oai_ready = True
                if AUTO_GREETING:
                    await self.oai_ws.send(json.dumps({
                        "type": "response.create",
                        "response": {"instructions": AUTO_GREETING}
                    }))
                break
            if evt.get("type") == "error":
                raise RuntimeError(f"OpenAI error: {evt.get('error')}")
        if not self.oai_ready:
            raise TimeoutError("OpenAI: no session.updated")

    async def _oai_send(self, payload: dict):
        assert self.oai_ws is not None
        await self.oai_ws.send(json.dumps(payload))

    async def _append_audio(self, b64_ulaw: str):
        await self._oai_send({"type": "input_audio_buffer.append", "audio": b64_ulaw})

    async def _commit(self):
        await self._oai_send({"type": "input_audio_buffer.commit"})

    async def _twilio_clear(self):
        try:
            if self.twilio_stream_sid:
                await self.twilio_ws.send_text(json.dumps({"event": "clear", "streamSid": self.twilio_stream_sid}))
        except Exception as e:
            print("Twilio clear error:", e)

    async def _pipe_twilio_to_openai(self):
        try:
            while not self.closed:
                msg = await self.twilio_ws.receive_text()
                data = json.loads(msg)
                event = data.get("event")

                if event == "start":
                    self.twilio_stream_sid = data.get("start", {}).get("streamSid")
                    print("Twilio WS start:", self.twilio_stream_sid)
                elif event == "media":
                    if not self.oai_ready:
                        continue
                    media = data.get("media", {})
                    payload = media.get("payload")
                    if payload:
                        # Barge-in: wenn Assistent spricht und der Anrufer spricht rein
                        if self.assistant_speaking and not self._barge_in:
                            await self._twilio_clear()
                            try:
                                await self._oai_send({"type": "response.cancel"})
                            except Exception as e:
                                print("response.cancel failed:", e)
                            self.assistant_speaking = False
                            self._barge_in = True
                        try:
                            await self._append_audio(payload)
                        except Exception as e:
                            print("append failed:", e)
                            self.closed = True
                            break
                elif event == "mark":
                    pass
                elif event == "stop":
                    print("Twilio WS stop")
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
                    t = evt.get("type")
                    if t == "response.created":
                        self._barge_in = False
                    elif t == "response.output_audio.delta":
                        delta = evt.get("delta")
                        if delta:
                            self.assistant_speaking = True
                            print("OpenAI audio delta", len(delta))
                            await self.twilio_ws.send_text(json.dumps({"event": "media", "media": {"payload": delta}}))
                    elif t == "response.completed":
                        self.assistant_speaking = False
                        await self.twilio_ws.send_text(json.dumps({"event": "mark", "mark": {"name": "oai_response_end"}}))
                    elif t == "response.cancelled":
                        self.assistant_speaking = False
                    elif t == "error":
                        print("OpenAI error:", evt.get("error"))
            except Exception as e:
                print("OpenAI ws closed/error:", repr(e))
                try:
                    if not self.closed:
                        await self._connect_openai()
                        continue
                except Exception as e2:
                    print("Reconnect failed:", e2)
                    self.closed = True
                    break

    async def run(self):
        await self._connect_openai()
        await asyncio.gather(self._pipe_twilio_to_openai(), self._pipe_openai_to_twilio())
        try:
            if self.oai_ws:
                await self.oai_ws.close()
        except Exception:
            pass

@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    # Twilio erwartet Subprotokoll "audio"
    try:
        requested = ws.headers.get('sec-websocket-protocol')
    except Exception:
        requested = None
    print("WS handshake: client requested subprotocol:", requested)
    await ws.accept(subprotocol="audio")
    session = CallSession(ws)
    try:
        await session.run()
    finally:
        try:
            await ws.close()
            print("WS closed")
        except Exception as e:
            print("WS close error:", e)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
