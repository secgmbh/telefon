
import os
import json
import asyncio
import base64
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
import uvicorn
import aiohttp
import time
import websockets

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")
TWILIO_VOICE = os.environ.get("TWILIO_VOICE", "verse")
REALTIME_SYSTEM_PROMPT = os.environ.get("REALTIME_SYSTEM_PROMPT", (
    "Du bist eine deutschsprachige Telefonassistenz. "
    "Sprich akzentfrei auf Hochdeutsch (de-DE), klare Artikulation, kurze Sätze, natürliches Tempo. "
    "Benutze deutsche Zahlen- und Datumsformate. Bei Unsicherheit: kurz nachfragen."
))
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

# ===== FastAPI app =====
app = FastAPI(title="Twilio ↔ OpenAI Realtime Bridge (GA)")
TWIML_GREETING = os.environ.get("TWIML_GREETING", "Willkommen bei Wowona Live. Einen Moment, ich verbinde Sie mit dem Assistenten.")

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

@app.get("/")
async def root():
    # Render hits '/' for health checks by default; return 200.
    return PlainTextResponse("service: ok\nendpoints: /healthz, /incoming-call, /telefon_live, /media-stream")

@app.head("/")
async def root_head():
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
  <Say>{TWIML_GREETING}</Say>
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
        self.assistant_speaking = False
        self.last_assistant_item_id = None
        self._barge_in_triggered = False
        self._text_buf = []
        self.user_speaking = False
        self._last_media_ts = 0.0

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
                "model": OPENAI_REALTIME_MODEL,
                "output_modalities": ["text"] if USE_AZURE_TTS else ["audio"],
                "instructions": (
                    REALTIME_SYSTEM_PROMPT
                ),
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
        # Twilio media payloads already base64-encoded μ-law.
        assert self.oai_ws is not None
        await self._oai_send_json({
            "type": "input_audio_buffer.append",
            "audio": base64_ulaw_payload
        })

    async def _oai_finish_input(self):
        # Commit the current input audio buffer to trigger server VAD turn end
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

    async def _azure_tts_synthesize(self, text: str) -> bytes:
        """Synthesize German TTS via Azure as 8kHz μ-law bytes ready for Twilio."""
        if not AZURE_TTS_KEY or not AZURE_TTS_REGION:
            raise RuntimeError("Azure TTS not configured")
        ssml = f"""
<speak version='1.0' xml:lang='de-DE'>
  <voice name='{AZURE_TTS_VOICE}'>
    <prosody rate='0%'>{text}</prosody>
  </voice>
</speak>
"""
        url = f"https://{AZURE_TTS_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_TTS_KEY,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "mulaw-8khz-8bit",
            "User-Agent": "wowona-rt-bridge/1.0"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=ssml.encode("utf-8"), headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Azure TTS failed {resp.status}: {body[:200]}")
                data = await resp.read()
                return data

    async def _send_mulaw_to_twilio(self, mulaw_bytes: bytes):
        """Chunk μ-law bytes into ~20ms frames (160 bytes) and send to Twilio as base64 payloads."""
        if not mulaw_bytes:
            return
        frame = 160  # 20ms @ 8kHz μ-law, 8-bit
        for i in range(0, len(mulaw_bytes), frame):
            if self._barge_in_triggered:
                break
            chunk = mulaw_bytes[i:i+frame]
            b64 = base64.b64encode(chunk).decode("ascii")
            await self.twilio_ws.send_text(json.dumps({
                "event": "media",
                "media": {"payload": b64}
            }))
            await asyncio.sleep(0.02)
        """Tell Twilio to clear any buffered outbound audio (barge-in)."""
        try:
            if self.twilio_stream_sid:
                await self.twilio_ws.send_text(json.dumps({
                    "event": "clear",
                    "streamSid": self.twilio_stream_sid
                }))
        except Exception as e:
            print("Twilio clear error:", e)
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
                    # Barge-in: if assistant is speaking and caller speaks, clear buffered audio & cancel current response
                    if self.assistant_speaking and not self._barge_in_triggered:
                        await self._twilio_clear()
                        try:
                            await self._oai_send_json({"type": "response.cancel"})
                        except Exception as e:
                            print("response.cancel failed:", e)
                        # Optional: truncate last assistant item to keep transcript clean
                        if self.last_assistant_item_id:
                            try:
                                await self._oai_send_json({
                                    "type": "conversation.item.truncate",
                                    "item_id": self.last_assistant_item_id,
                                    "content_index": 0,
                                    "audio_end_ms": 0
                                })
                            except Exception as e:
                                print("truncate failed:", e)
                        self.assistant_speaking = False
                        self._barge_in_triggered = True
                    media = data.get("media", {})
                    payload = media.get("payload")
                    self.user_speaking = True
                    self._last_media_ts = time.time()
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
                elif mtype == "response.created":
                    # New response starting; allow future barge-in
                    self._barge_in_triggered = False
                elif mtype == "conversation.item.created":
                    item = evt.get("item", {})
                    role = item.get("role") or item.get("type")
                    if role == "assistant":
                        self.last_assistant_item_id = item.get("id")
                elif mtype == "response.output_audio.delta" and not USE_AZURE_TTS:
                    self.assistant_speaking = True
                elif mtype == "response.output_text.delta" and USE_AZURE_TTS:
                    delta_text = evt.get("delta", "")
                    if delta_text:
                        self._text_buf.append(delta_text)
                        self.assistant_speaking = True
                    # New GA event field name: 'delta' (base64 μ-law chunk)
                    delta = evt.get("delta")
                    if delta:
                        await self.twilio_ws.send_text(json.dumps({
                            "event": "media",
                            "media": {"payload": delta}
                        }))
                elif mtype == "response.completed":
                    # Send buffered text via Azure TTS if enabled
                    if USE_AZURE_TTS and self._text_buf:
                        try:
                            text = "".join(self._text_buf).strip()
                            self._text_buf.clear()
                            if text:
                                audio = await self._azure_tts_synthesize(text)
                                # Reset barge-in flag for fresh playback window
                                self._barge_in_triggered = False
                                await self._send_mulaw_to_twilio(audio)
                        except Exception as e:
                            print("Azure TTS error:", e)
                    self.assistant_speaking = False
                    await self.twilio_ws.send_text(json.dumps({
                        "event": "mark", "mark": {"name": "oai_response_end"}
                    }))
                elif mtype == "response.cancelled":
                    self.assistant_speaking = False
                elif mtype == "error":
                    err = evt.get("error", {})
                    print("OpenAI error:", err)
        except Exception as e:
            print("OpenAI ws closed/error:", repr(e))
        finally:
            self.closed = True

    
    async def _silence_watcher(self, quiet_ms: int = 800):
        """Commit & create response after short silence since last media."""
        try:
            while not self.closed:
                await asyncio.sleep(0.1)
                if not self.user_speaking:
                    continue
                if (time.time() - self._last_media_ts) * 1000 >= quiet_ms:
                    self.user_speaking = False
                    try:
                        await self._oai_send_json({"type": "input_audio_buffer.commit"})
                        
                    except Exception as e:
                        print("silence_watcher error:", e)
        except Exception as e:
            print("silence_watcher crashed:", e)
async def run(self):
        await self._connect_openai()
        # Run pumps concurrently
        watcher_task = asyncio.create_task(self._silence_watcher())
        await asyncio.gather(
            self._pipe_twilio_to_openai(),
            self._pipe_openai_to_twilio(),
            watcher_task,
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
