# main.py (Twilio ↔ OpenAI Realtime, Deutsch) – AIOHTTP WS client (stable)
import os
import json
import asyncio
import traceback
from typing import Optional
from urllib.parse import parse_qs

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, HTMLResponse, PlainTextResponse
from dotenv import load_dotenv
import aiohttp

load_dotenv()

# =========================
# Konfiguration / Env
# =========================
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_REALTIME_MODEL = (os.getenv("OPENAI_REALTIME_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-realtime").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY fehlt (Env oder .env).")

ALLOWED_OAI_VOICES = {"alloy","ash","ballad","coral","echo","sage","shimmer","verse","marin","cedar"}
TWILIO_VOICE = (os.getenv("TWILIO_VOICE") or "verse").strip().lower()
if TWILIO_VOICE not in ALLOWED_OAI_VOICES:
    print(f"[config] Unsupported voice '{TWILIO_VOICE}'. Falling back to 'verse'. Supported: {sorted(ALLOWED_OAI_VOICES)}")
    TWILIO_VOICE = "verse"

AUTO_GREETING = os.getenv("AUTO_GREETING", "Guten Tag! Ich bin Ihr Assistent. Wie kann ich helfen?")
SYSTEM_PROMPT = os.getenv(
    "REALTIME_SYSTEM_PROMPT",
    "Antworte ausschließlich auf Deutsch (Hochdeutsch, de-DE). Sprich natürlich und freundlich. "
    "Antworte in 2–4 Sätzen und frage nach, wenn etwas unklar ist."
)

def _to_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

VAD_SILENCE_MS = _to_int("VAD_SILENCE_MS", 900)
POST_GREETING_MUTE_MS = _to_int("POST_GREETING_MUTE_MS", 1200)
COMMIT_INTERVAL = float(os.getenv("COMMIT_INTERVAL_SEC", "0.9"))

# =========================
# FastAPI
# =========================
app = FastAPI()

@app.api_route("/", methods=["GET","HEAD"])
async def root():
    return HTMLResponse("<h3>Wowona Live – Service läuft ✅</h3>")

@app.api_route("/healthz", methods=["GET","HEAD"])
async def healthz():
    return PlainTextResponse("ok")

@app.get("/diag")
async def diag():
    url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                url,
                headers=headers,
                protocols=["realtime"],
                heartbeat=20,
                autoping=True,
                timeout=20,
                max_msg_size=10*1024*1024
            ) as oai:
                await oai.send_str(json.dumps({
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio","text"],
                        "voice": TWILIO_VOICE,
                        "input_audio_format": "g711_ulaw",
                        "output_audio_format": "g711_ulaw",
                        "turn_detection": {"type":"server_vad","silence_duration_ms": VAD_SILENCE_MS},
                        "instructions": SYSTEM_PROMPT,
                    },
                }))
                # warte auf erste Antwort / Fehler
                try:
                    msg = await asyncio.wait_for(oai.receive(), timeout=5)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get("type") == "session.updated":
                            return PlainTextResponse('diag: ok - session.updated')
                        elif data.get("type") == "error":
                            return PlainTextResponse(f"diag: error - {data}", status_code=500)
                except asyncio.TimeoutError:
                    pass
                return PlainTextResponse('diag: ok - (no immediate reply, socket open)')
    except Exception as e:
        return PlainTextResponse(f'diag: error - {repr(e)}', status_code=500)

def _http_to_ws(url: str) -> str:
    return url.replace("https://", "wss://").replace("http://", "ws://")

@app.api_route("/telefon_live", methods=["GET","POST"])
async def telefon_live(request: Request):
    # TwiML mit bidirektionalem Stream. WICHTIG: kein track="both"
    ws_url = _http_to_ws(str(request.url_for("twilio_stream")))
    status_url = str(request.url_for("stream_status"))
    print(f"TwiML requested. Using stream URL: {ws_url} | statusCallback: {status_url}")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}" statusCallback="{status_url}"/>
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.api_route("/stream-status", methods=["POST","GET"])
async def stream_status(request: Request):
    try:
        ctype = request.headers.get("content-type","")
        raw = (await request.body()).decode("utf-8", "ignore")
        if "application/x-www-form-urlencoded" in ctype:
            data = {k:(v[0] if len(v)==1 else v) for k,v in parse_qs(raw).items()}
            print("Twilio StreamStatus:", data)
        else:
            print("Twilio StreamStatus (raw):", raw or "(empty)")
    except Exception as e:
        print("StreamStatus parse error:", repr(e))
    return PlainTextResponse("ok")

@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    req_proto = ws.headers.get("sec-websocket-protocol", "")
    requested = [p.strip() for p in req_proto.split(",") if p.strip()]
    subproto = None
    if "audio.stream.twilio.com" in requested:
        subproto = "audio.stream.twilio.com"
    elif "audio" in requested:
        subproto = "audio"

    await ws.accept(subprotocol=subproto)
    print(f"WS handshake: client requested subprotocols={requested or ['(none)']}, accepted={subproto or '(none)'}")

    bridge = TwilioOpenAIBridge(ws)
    try:
        await bridge.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("Top-level bridge exception:", repr(e))
        traceback.print_exc()
    finally:
        try:
            await bridge.stop()
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        print("WS: closed")

# =========================
# Brücke Twilio ↔ OpenAI
# =========================
class TwilioOpenAIBridge:
    def __init__(self, twilio_ws: WebSocket):
        self.twilio_ws = twilio_ws
        self.oai_ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._http: Optional[aiohttp.ClientSession] = None
        self.running = True
        self._oai_reader_task: Optional[asyncio.Task] = None
        self._twilio_reader_task: Optional[asyncio.Task] = None
        self._commit_task: Optional[asyncio.Task] = None
        self.twilio_stream_sid: Optional[str] = None
        self._post_greeting_mute = POST_GREETING_MUTE_MS / 1000.0
        self.oai_ready = False
        self._frames_since_commit = 0

    async def _connect_openai(self, attempt: int = 1):
        url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }
        print(f"OAI: connecting → {url}")
        try:
            self._http = aiohttp.ClientSession()
            self.oai_ws = await self._http.ws_connect(
                url,
                headers=headers,
                protocols=["realtime"],
                heartbeat=20,
                autoping=True,
                timeout=20,
                max_msg_size=10*1024*1024,
            )
            print("OAI: connected (aiohttp)")
        except Exception as e:
            print(f"OAI: connect failed (attempt {attempt}):", repr(e))
            if attempt < 3 and self.running:
                await asyncio.sleep(min(1.0 * attempt, 3.0))
                return await self._connect_openai(attempt + 1)
            raise

        # Session-Update
        await self._oai_send_json({
            "type": "session.update",
            "session": {
                "modalities": ["audio","text"],
                "voice": TWILIO_VOICE,
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "turn_detection": {"type":"server_vad","silence_duration_ms": VAD_SILENCE_MS},
                "instructions": SYSTEM_PROMPT,
            },
        }, "session.update")

        # Begrüßung
        await self._oai_send_json({
            "type":"response.create",
            "response": {"modalities":["audio","text"], "instructions": AUTO_GREETING}
        }, "response.create (greeting)")

    async def run(self):
        await self._connect_openai()
        self._twilio_reader_task = asyncio.create_task(self._pipe_twilio_to_openai())
        self._oai_reader_task = asyncio.create_task(self._pipe_openai_to_twilio())
        self._commit_task = asyncio.create_task(self._commit_loop())

        done, pending = await asyncio.wait(
            {self._twilio_reader_task, self._oai_reader_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )

        for t in done:
            if t.exception():
                print("Bridge task exception:", repr(t.exception()))
                traceback.print_exc()

        self.running = False
        for t in pending:
            t.cancel()

    async def stop(self):
        self.running = False
        try:
            if self.oai_ws is not None and not self.oai_ws.closed:
                await self.oai_ws.close()
        except Exception:
            pass
        try:
            if self._http is not None:
                await self._http.close()
        except Exception:
            pass
        self.oai_ws = None
        self._http = None

    async def _pipe_twilio_to_openai(self):
        async for message in self.twilio_ws.iter_text():
            if not self.running:
                break
            try:
                data = json.loads(message)
            except Exception:
                continue

            etype = data.get("event")
            if etype == "connected":
                print("WS RX connected:", data)
            elif etype == "start":
                self.twilio_stream_sid = data.get("start", {}).get("streamSid")
                print("Twilio start, streamSid:", self.twilio_stream_sid)
                await asyncio.sleep(self._post_greeting_mute)
            elif etype == "media":
                if not self.oai_ws or self.oai_ws.closed:
                    print("[skip] media → OAI closed")
                    break
                if not self.oai_ready:
                    # drop until session.updated kommt
                    continue
                payload = data.get("media", {}).get("payload")
                if payload:
                    ok = await self._oai_send_json(
                        {"type": "input_audio_buffer.append", "audio": payload},
                        "input_audio_buffer.append"
                    )
                    if ok:
                        self._frames_since_commit += 1
                    else:
                        break
            elif etype == "stop":
                print("Twilio stop (call ended)")
                break

        print("Twilio reader: stream ended")
        self.running = False

    async def _pipe_openai_to_twilio(self):
        if not self.oai_ws:
            print("OAI reader: socket missing, abort")
            return

        try:
            async for msg in self.oai_ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    raw = msg.data
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                    print(f"OAI closed: code={self.oai_ws.close_code}, reason={(self.oai_ws.close_reason or '')!r}")
                    break
                else:
                    continue

                try:
                    data = json.loads(raw)
                except Exception:
                    continue

                mtype = data.get("type")
                if mtype == "error":
                    print("OAI error event:", data)
                elif mtype == "session.updated":
                    print("OpenAI session.updated received")
                    self.oai_ready = True
                elif mtype == "response.output_audio.delta":
                    audio = data.get("audio")
                    if audio and self.twilio_stream_sid:
                        try:
                            await self.twilio_ws.send_text(json.dumps({
                                "event": "media",
                                "streamSid": self.twilio_stream_sid,
                                "media": {"payload": audio}
                            }))
                        except Exception as e:
                            print("Twilio send error:", repr(e))
                            break
                elif mtype == "response.completed":
                    await asyncio.sleep(0.6)
                else:
                    pass
        except Exception as e:
            print("OAI pipe error:", repr(e))
            traceback.print_exc()
        finally:
            print("OpenAI reader: stream ended")
            self.running = False

    async def _commit_loop(self):
        # schickt periodisch commits, solange Frames eingegangen sind
        try:
            while self.running:
                await asyncio.sleep(COMMIT_INTERVAL)
                if not self.oai_ws or self.oai_ws.closed:
                    break
                if not self.oai_ready:
                    continue
                if self._frames_since_commit > 0:
                    ok = await self._oai_send_json({"type":"input_audio_buffer.commit"}, "input_audio_buffer.commit")
                    if ok:
                        self._frames_since_commit = 0
                    else:
                        break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print("Commit loop error:", repr(e))

    async def _oai_send_json(self, payload: dict, label: str = "") -> bool:
        if not self.oai_ws:
            print(f"[WARN] OAI send skipped ({label}) – socket not open")
            return False
        if self.oai_ws.closed or (self.oai_ws.close_code is not None):
            print(f"[WARN] OAI send skipped ({label}) – socket closed ({self.oai_ws.close_code})")
            return False
        try:
            await self.oai_ws.send_str(json.dumps(payload))
            if label:
                preview = json.dumps(payload)
                if len(preview) > 300:
                    preview = preview[:300] + " …"
                print(f"OAI ⇢ {label}: {preview}")
            return True
        except Exception as e:
            print(f"OAI send error ({label}):", repr(e))
            traceback.print_exc()
            return False
