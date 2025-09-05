import os
import json
import time
import base64
import asyncio
import traceback
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from dotenv import load_dotenv, dotenv_values
import websockets
import urllib.request, urllib.error

load_dotenv()

# ===================== ENV =====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
STREAM_WSS_PATH = os.getenv("TWILIO_STREAM_PATH", "/twilio-stream")  # Fallback, falls TWILIO_STREAM_WSS fehlt
VOICE = os.getenv("TWILIO_VOICE", "alloy")
SYSTEM_PROMPT = os.getenv(
    "REALTIME_SYSTEM_PROMPT",
    "Du bist Maria, eine freundliche, präzise deutschsprachige Assistentin. "
    "Sprich ausschließlich Deutsch. Antworte kurz und direkt."
)
GREETING = os.getenv("FIRST_GREETING", "Hallo, ich bin Maria von wowona. Womit kann ich helfen?")
TWILIO_SUBPROTOCOL = "audio.stream.twilio.com"  # Twilio erwartet dieses Subprotocol

# ===================== APP =====================
app = FastAPI()

# ===================== Hilfsfunktionen (Key-Diagnose & Preflight) =====================
def _mask_key(k: str) -> str:
    if not k:
        return "<empty>"
    return k[:8] + "…" + k[-6:] if len(k) > 14 else k

def _classify_key(k: str) -> str:
    if not k: return "none"
    if k.startswith("sk-svcacct-"): return "service-account (NOT for Realtime)"
    if k.startswith("sk-proj-"): return "project key (OK)"
    if k.startswith("sk-"): return "standard key (OK)"
    return "unknown"

# Logge, von wo der Key kommt
print(f"[KEY CHECK] ENV OPENAI_API_KEY: {_mask_key(OPENAI_API_KEY)} → {_classify_key(OPENAI_API_KEY)}")
try:
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        dv = dotenv_values(env_path)
        file_key = dv.get("OPENAI_API_KEY", "")
        print(f"[KEY CHECK] .env OPENAI_API_KEY: {_mask_key(file_key)} → {_classify_key(file_key)} (from {env_path})")
    else:
        print("[KEY CHECK] .env not found")
except Exception as e:
    print("[KEY CHECK] .env read error:", repr(e))

def _preflight_openai_key(key: str):
    if not key or not key.startswith("sk-"):
        print("Preflight: Kein/ungültiger OPENAI_API_KEY (erwartet 'sk-…').")
        return
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"Preflight: OpenAI key ok → HTTP {resp.status}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        print(f"Preflight: HTTPError {e.code} → {body[:160]} …")
    except Exception as e:
        print("Preflight: request error →", repr(e))

_preflight_openai_key(OPENAI_API_KEY)

# ===================== μ-law <-> PCM16 @ 8kHz =====================
_ULAW_BIAS = 0x84  # 132
_ULAW_CLIP = 32635

def ulaw_decode_bytes(ulaw_bytes: bytes) -> bytes:
    """G.711 μ-law (8-bit) -> PCM16 little-endian (int16)"""
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
    """PCM16 little-endian (int16) -> G.711 μ-law (8-bit)"""
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.int32)
    x = np.clip(x, -_ULAW_CLIP, _ULAW_CLIP)
    sign = x < 0
    x = np.abs(x).astype(np.int32) + _ULAW_BIAS
    ulaw = np.empty(x.shape[0], dtype=np.uint8)
    for i in range(x.shape[0]):
        xi = x[i]
        seg = _ulaw_segment_scalar(xi)
        mant = (xi >> (seg + 3)) & 0x0F
        u = ((seg << 4) | mant) ^ 0xFF
        if sign[i]:
            u |= 0x80
        ulaw[i] = u
    return ulaw.tobytes()

# ===================== sehr einfache VAD (Rauschfilter) =====================
def is_voice(pcm16_le: bytes, threshold: float = 600.0) -> bool:
    """Simple RMS-Schwelle; erhöhe threshold (z. B. 800–1000) bei lauter Umgebung."""
    if not pcm16_le:
        return False
    x = np.frombuffer(pcm16_le, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return False
    rms = float(np.sqrt(np.mean(x * x)))
    return rms >= threshold

# ===================== HTTP: TwiML =====================
@app.api_route("/telefon_live", methods=["GET", "POST"])
async def telefon_live():
    # Entweder fix in ENV (empfohlen): TWILIO_STREAM_WSS=wss://<deine-domain>/twilio-stream
    wss_url = os.getenv("TWILIO_STREAM_WSS") or ("wss://YOUR_DOMAIN" + STREAM_WSS_PATH)
    print("TwiML requested. Using stream URL:", wss_url)
    # Wichtig: KEIN <Say> – Stream sofort starten
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{wss_url}" />
  </Connect>
</Response>"""
    return Response(content=xml, media_type="text/xml")

@app.get("/")
async def root():
    return Response(content="OK", media_type="text/plain")

# ===================== Bridge =====================
class Bridge:
    def __init__(self, twilio_ws: WebSocket):
        self.twilio_ws = twilio_ws
        self.oai_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.stream_sid: Optional[str] = None

        # Timing/Buffering
        self.last_media_ts: float = 0.0
        self.last_commit_ts: float = 0.0
        self.last_activity_ts: float = time.time()  # für Idle-Logik
        self.bytes_buffered: int = 0

        # Tasks/State
        self.commit_task: Optional[asyncio.Task] = None
        self.playing_audio: bool = False
        self.closed = False
        self.seen_start = False

        # Idle-Logik
        self.IDLE_PROMPT_AFTER = 6.0     # s Schweigen → „Noch etwas?“
        self.HANGUP_AFTER_IDLE = 10.0    # weitere s → Verabschieden + close
        self.idle_prompt_sent = False

        # Latenz-Tuning (8 kHz, 16 bit ≈ 16 kB/s)
        self.BUFFER_BYTES_THRESHOLD = 6000   # ~0.375 s Audio bis Commit
        self.FORCE_COMMIT_INTERVAL = 0.7     # spätestens nach 0.7 s committen
        self.SILENCE_COMMIT_GAP = 0.18       # ~180 ms Pause = Commit

    async def open_openai(self):
        url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }
        if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
            print("⚠️  OPENAI_API_KEY fehlt/ungültig. Bitte Environment prüfen.")
        try:
            print("OAI: connecting →", url)
            self.oai_ws = await asyncio.wait_for(
                websockets.connect(url, extra_headers=headers, ping_interval=20),
                timeout=5.0
            )
            print("OAI: connected")
            await self.oai_ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "instructions": SYSTEM_PROMPT,
                    "modalities": ["audio", "text"],
                    "voice": VOICE,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16"
                },
            }))
            print("OAI: session.update sent")
        except Exception as e:
            print("OAI: failed to connect or configure →", repr(e))
            traceback.print_exc()
            raise

    async def start(self):
        await self.open_openai()
        asyncio.create_task(self._pipe_openai_to_twilio())
        self.commit_task = asyncio.create_task(self._auto_commit_loop())
        asyncio.create_task(self._start_watchdog())
        print("Bridge started: OpenAI pipe + auto-commit loop running")

    async def _start_watchdog(self):
        await asyncio.sleep(10)
        if not self.seen_start and not self.closed:
            print("Watchdog: 10s ohne 'start' Event. Prüfe Twilio-Stream-Webhook.")
            await self.close()

    async def _auto_commit_loop(self):
        while not self.closed:
            await asyncio.sleep(0.05)
            now = time.time()

            # Während KI spricht: niemals committen
            if self.playing_audio:
                # Optional: Timeout-Schutz, falls playing stuck (sehr selten)
                continue

            # (1) Commit bei ~180 ms Stille (nur wenn wir zuvor Audio gesammelt haben)
            if self.last_media_ts and (now - self.last_media_ts > self.SILENCE_COMMIT_GAP) and self.bytes_buffered > 0:
                print("Auto-commit (silence): committing after ~180ms pause")
                await self._commit_and_request_response()
                self.bytes_buffered = 0
                self.last_commit_ts = now
                self.last_media_ts = 0.0
                continue

            # (2) Spätestens nach 0.7 s committen, wenn genug Audio gesammelt
            if self.bytes_buffered >= self.BUFFER_BYTES_THRESHOLD and (now - self.last_commit_ts) >= self.FORCE_COMMIT_INTERVAL:
                print(f"Auto-commit (force): bytes_buffered={self.bytes_buffered}, dt={now - self.last_commit_ts:.2f}s")
                await self._commit_and_request_response()
                self.bytes_buffered = 0
                self.last_commit_ts = now

            # (3) Idle-Überwachung (nur wenn KI NICHT spricht)
            idle_for = now - self.last_activity_ts
            if idle_for >= self.IDLE_PROMPT_AFTER and not self.idle_prompt_sent:
                try:
                    await self.oai_ws.send(json.dumps({
                        "type": "response.create",
                        "response": {
                            "modalities": ["audio", "text"],
                            "instructions": "Möchten Sie noch etwas wissen?"
                        }
                    }))
                    print("OAI: idle prompt sent")
                    self.idle_prompt_sent = True
                    self.last_activity_ts = now  # Timer neu starten
                except Exception as e:
                    print("OAI idle prompt error:", repr(e))

            elif self.idle_prompt_sent and idle_for >= (self.IDLE_PROMPT_AFTER + self.HANGUP_AFTER_IDLE):
                try:
                    await self.oai_ws.send(json.dumps({
                        "type": "response.create",
                        "response": {
                            "modalities": ["audio", "text"],
                            "instructions": "Danke für Ihren Anruf. Auf Wiederhören!"
                        }
                    }))
                    print("OAI: goodbye sent; closing bridge soon")
                except Exception as e:
                    print("OAI goodbye error:", repr(e))
                await asyncio.sleep(1.0)
                await self.close()
                return

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
                # Twilio sendet Text- oder Binary-Frames
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
                if event == "connected":
                    pass

                elif event == "start":
                    self.seen_start = True
                    self.stream_sid = data.get("start", {}).get("streamSid")
                    print("Twilio start, streamSid:", self.stream_sid)
                    self.last_commit_ts = time.time()
                    self.last_activity_ts = time.time()
                    self.idle_prompt_sent = False

                    # Proaktive Begrüßung → sofort Audio zurückschicken
                    try:
                        await self.oai_ws.send(json.dumps({
                            "type": "response.create",
                            "response": {
                                "modalities": ["audio", "text"],
                                "instructions": GREETING
                            }
                        }))
                        print("OAI: proactive greeting response.create sent")
                    except Exception as e:
                        print("OAI proactive greeting error:", repr(e))

                elif event == "media":
                    # Wenn die KI gerade spricht → eingehendes Audio ignorieren (verhindert Endlosschleifen)
                    if self.playing_audio:
                        self.last_media_ts = time.time()
                        continue

                    ulaw_b64 = data["media"]["payload"]
                    ulaw = base64.b64decode(ulaw_b64)
                    pcm16_8k = ulaw_decode_bytes(ulaw)

                    # Rauschen ignorieren (VAD)
                    if not is_voice(pcm16_8k):
                        self.last_media_ts = time.time()
                        # Keine Bytes anhängen, kein Commit
                        continue

                    # Barge-in: Falls trotzdem noch Audio gespielt wird, stoppen
                    if self.playing_audio and self.stream_sid:
                        print("Barge-in: clearing playback")
                        await self._twilio_send({"event": "clear", "streamSid": self.stream_sid})
                        self.playing_audio = False

                    # Nur bei „echter“ Stimme an OpenAI anhängen
                    b = base64.b64encode(pcm16_8k).decode()
                    await self.oai_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": b,
                    }))

                    # Buffer-Tracking
                    self.bytes_buffered += len(pcm16_8k)
                    self.last_media_ts = time.time()
                    self.last_activity_ts = time.time()
                    self.idle_prompt_sent = False

                elif event == "stop":
                    print("Twilio stop")
                    if self.bytes_buffered > 0 and not self.playing_audio:
                        await self._commit_and_request_response()
                        self.bytes_buffered = 0
                        self.last_commit_ts = time.time()
                    break

        except WebSocketDisconnect:
            print("Twilio WS disconnect")
        except Exception as e:
            print("Twilio WS error:", repr(e))
            traceback.print_exc()
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
                        self.last_activity_ts = time.time()
                        self.idle_prompt_sent = False

                elif t and t.startswith("error"):
                    print("OAI error event:", data)

        except Exception as e:
            print("OAI pipe error:", repr(e))
            traceback.print_exc()

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
            traceback.print_exc()

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

# ===================== WS Route =====================
@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    # Header-Log & Subprotocol-Handshake
    headers_dict = dict(ws.headers)
    print("WS headers:", headers_dict)
    try:
        await ws.accept(subprotocol=TWILIO_SUBPROTOCOL)
        print("WS: accepted with subprotocol =", TWILIO_SUBPROTOCOL)
    except Exception as e:
        print("WS accept error:", repr(e))
        traceback.print_exc()
        return

    bridge = Bridge(ws)
    try:
        await bridge.start()
        await bridge.receive_from_twilio()
    except Exception as e:
        print("Bridge error:", repr(e))
        traceback.print_exc()
        await bridge.close()

# ===================== Local dev =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
