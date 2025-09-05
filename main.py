import os
import json
import time
import asyncio
import traceback
import websockets

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, Response
from dotenv import load_dotenv

# -------------------------------------------------
# ENV laden
# -------------------------------------------------
load_dotenv()

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
STREAM_WSS_PATH = os.getenv("TWILIO_STREAM_PATH", "/twilio-stream")
VOICE = os.getenv("TWILIO_VOICE", "alloy")
SYSTEM_PROMPT = os.getenv(
    "REALTIME_SYSTEM_PROMPT",
    "Du bist Maria, eine freundliche, präzise deutschsprachige Assistentin. "
    "Sprich ausschließlich Deutsch. Antworte kurz und direkt."
)
GREETING = os.getenv("FIRST_GREETING", "Willkommen bei wowona Live. Wie kann ich helfen?")

TWILIO_SUBPROTOCOL = "audio.stream.twilio.com"

app = FastAPI()

# -------------------------------------------------
# Utility: Key-Check
# -------------------------------------------------
def _mask(k: str) -> str:
    return (k[:8] + "…" + k[-6:]) if k and len(k) > 14 else (k or "<empty>")

def _classify(k: str) -> str:
    if not k: return "none"
    if k.startswith("sk-proj-"): return "project key (OK)"
    if k.startswith("sk-svcacct-"): return "service-account (NOT for Realtime)"
    if k.startswith("sk-"): return "standard key (OK)"
    return "unknown"

print(f"[KEY CHECK] ENV OPENAI_API_KEY: {_mask(OPENAI_API_KEY)} → {_classify(OPENAI_API_KEY)}")

# -------------------------------------------------
# HTTP: Status
# -------------------------------------------------
@app.get("/")
async def index():
    return PlainTextResponse("OK")

# -------------------------------------------------
# HTTP: TwiML (Twilio fragt diesen Endpoint ab)
# -------------------------------------------------
@app.api_route("/telefon_live", methods=["GET", "POST"])
async def telefon_live(request: Request):
    # Robuste WSS-URL-Erzeugung (Fix für TypeError)
    try:
        wss_url = str(request.url_for("twilio_stream")).replace("http", "wss", 1)
    except Exception:
        # Fallback falls Router-Name nicht gefunden → auf Basis der Base-URL bauen
        base = str(request.base_url).rstrip("/")
        wss_url = base.replace("http", "wss", 1) + STREAM_WSS_PATH

    print(f"TwiML requested. Using stream URL: {wss_url}")

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{wss_url}"/>
  </Connect>
</Response>"""
    return Response(content=xml, media_type="text/xml")

# -------------------------------------------------
# Bridge: Twilio <-> OpenAI
# -------------------------------------------------
class Bridge:
    def __init__(self, twilio_ws: WebSocket):
        self.twilio_ws = twilio_ws
        self.oai_ws: websockets.WebSocketClientProtocol | None = None
        self.stream_sid: str | None = None

        self.playing_audio = False
        self.closed = False

        # Einfaches Auto-Commit: wenn ~180ms keine eingehenden Media-Pakete → commit
        self.last_media_ts = 0.0
        self.silence_gap_sec = 0.18
        self.force_interval_sec = 0.8
        self.last_commit_ts = time.time()
        self.bytes_since_commit = 0

    # ----------------- OpenAI Connect -----------------
    async def open_openai(self):
        url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

        # ✅ Korrekte Beta-Header + beide Subprotocols anbieten
        headers = [
            ("Authorization", f"Bearer {OPENAI_API_KEY}"),
            ("OpenAI-Beta", "realtime-v1"),
        ]
        subprotocols = ["realtime", "openai-beta.realtime.v1"]

        print("OAI: connecting →", url)
        print("OAI: offering subprotocols:", subprotocols, "| auth header present:", bool(OPENAI_API_KEY))

        self.oai_ws = await websockets.connect(
            url,
            extra_headers=headers,
            subprotocols=subprotocols,
            ping_interval=20,
        )
        print("OAI: connected (negotiated subprotocol:", getattr(self.oai_ws, "subprotocol", None), ")")

        # ⚠️ Minimal gültiges session.update (alles andere kann invalid_value auslösen)
        await self.oai_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": SYSTEM_PROMPT,
                "modalities": ["audio", "text"],
                "voice": VOICE,
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
            }
        }))
        print("OAI: session.update sent (μ-law)")

        # Proaktiver Gruß
        await self.oai_ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["audio", "text"],
                "instructions": GREETING,
            }
        }))
        print("OAI: proactive greeting response.create sent")

    # ----------------- Start Bridge -----------------
    async def start(self):
        await self.open_openai()
        # Starte beide Richtungen
        await asyncio.gather(
            self._pipe_twilio_to_openai(),
            self._pipe_openai_to_twilio(),
            self._auto_commit_loop(),
        )

    # ----------------- Twilio → OpenAI -----------------
    async def _pipe_twilio_to_openai(self):
        try:
            while True:
                # Twilio sendet Text-Frames (JSON)
                raw = await self.twilio_ws.receive_text()
                if len(raw) > 140:
                    print("WS RX (truncated):", raw[:140], "…")
                else:
                    print("WS RX:", raw)

                data = json.loads(raw)
                event = data.get("event")

                if event == "connected":
                    pass

                elif event == "start":
                    self.stream_sid = data.get("start", {}).get("streamSid")
                    print("Twilio start, streamSid:", self.stream_sid)
                    self.last_commit_ts = time.time()

                elif event == "media":
                    # Während die KI spricht, puffern wir nicht (Echo vermeiden)
                    if self.playing_audio:
                        continue
                    b64_ulaw = data["media"]["payload"]
                    self.last_media_ts = time.time()
                    self.bytes_since_commit += len(b64_ulaw)

                    # μ-law Base64 direkt an OpenAI anhängen
                    await self.oai_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": b64_ulaw
                    }))

                elif event == "stop":
                    print("Twilio stop")
                    # Final commit & response
                    await self._commit_and_request()
                    break

        except WebSocketDisconnect:
            print("Twilio WS disconnect")
        except Exception as e:
            print("Twilio WS error:", repr(e))
            traceback.print_exc()
        finally:
            await self.close()

    # ----------------- OpenAI → Twilio -----------------
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
                    delta_b64 = data.get("delta") or data.get("audio")
                    if not delta_b64:
                        continue
                    if self.stream_sid:
                        await self._twilio_send({
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {"payload": delta_b64}
                        })
                        self.playing_audio = True

                elif t in ("response.audio.done", "response.output_audio.done"):
                    print("OAI → audio done")
                    # Mark für Twilio (optional)
                    if self.stream_sid:
                        await self._twilio_send({
                            "event": "mark",
                            "streamSid": self.stream_sid,
                            "mark": {"name": "assistant_turn_done"}
                        })
                    self.playing_audio = False
                    # nach einer KI-Antwort neuen Turn beginnen → Bytes-Zähler zurücksetzen
                    self.bytes_since_commit = 0

                elif t and t.startswith("error"):
                    print("OAI error event:", data)

        except Exception as e:
            print("OAI pipe error:", repr(e))
            traceback.print_exc()

    # ----------------- Auto-Commit (einfache Stilleerkennung) -----------------
    async def _auto_commit_loop(self):
        # Commit bei ~180ms Stille oder spätestens alle ~0.8s, sofern wir zuletzt Audio empfangen haben
        try:
            while not self.closed:
                await asyncio.sleep(0.05)
                now = time.time()
                if self.playing_audio:
                    continue
                # nur committen, wenn wir seit letztem Commit überhaupt was gesammelt haben
                if self.bytes_since_commit == 0:
                    continue
                # Stillefenster
                if self.last_media_ts and (now - self.last_media_ts) > self.silence_gap_sec:
                    print("Auto-commit (silence) → commit + response.create")
                    await self._commit_and_request()
                # Hard-Intervall
                elif (now - self.last_commit_ts) > self.force_interval_sec:
                    print("Auto-commit (force) → commit + response.create")
                    await self._commit_and_request()
        except Exception as e:
            print("Auto-commit loop error:", repr(e))

    async def _commit_and_request(self):
        try:
            await self.oai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await self.oai_ws.send(json.dumps({"type": "response.create", "response": {"modalities": ["audio", "text"]}}))
            self.last_commit_ts = time.time()
            self.bytes_since_commit = 0
            print("OAI: commit + response.create sent")
        except Exception as e:
            print("OAI commit/response error:", repr(e))

    # ----------------- Twilio senden (Hilfsfunktion) -----------------
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

    # ----------------- Cleanup -----------------
    async def close(self):
        if self.closed:
            return
        self.closed = True
        print("Bridge closing …")
        try:
            if self.oai_ws:
                await self.oai_ws.close()
        except Exception as e:
            print("OAI close error:", repr(e))
        try:
            await self.twilio_ws.close()
        except Exception as e:
            print("Twilio WS close error:", repr(e))


# -------------------------------------------------
# WebSocket-Route für Twilio Media Streams
# -------------------------------------------------
@app.websocket(STREAM_WSS_PATH)
async def twilio_stream(ws: WebSocket):
    headers_dict = dict(ws.headers)
    print("WS headers:", headers_dict)
    # Twilio erwartet das Subprotocol
    await ws.accept(subprotocol=TWILIO_SUBPROTOCOL)
    print("WS: accepted with subprotocol =", TWILIO_SUBPROTOCOL)
    bridge = Bridge(ws)
    await bridge.start()

# -------------------------------------------------
# Local dev
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
