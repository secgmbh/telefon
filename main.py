import os
import json
import asyncio
import traceback
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import Response, HTMLResponse
from fastapi.websockets import WebSocket
from dotenv import load_dotenv

import websockets

load_dotenv()

# =========================
# Konfiguration
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-realtime-preview").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY fehlt (Env oder .env).")

# =========================
# FastAPI
# =========================
app = FastAPI()

@app.get("/")
async def root():
    return HTMLResponse("<h3>Wowona Live – Service läuft ✅</h3>")

@app.post("/telefon_live")
async def telefon_live(request: Request):
    """
    Wird von Twilio Voice aufgerufen. Liefert TwiML, das einen Media Stream
    zu unserem WS-Endpunkt herstellt.
    """
    # Wichtig: URL.replace nur mit 1 Argument benutzen (string → string)
    ws_url = str(request.url_for("twilio_stream")).replace("http", "wss")
    print(f"TwiML requested. Using stream URL: {ws_url}")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}"/>
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    print(f"WS headers: {dict(ws.headers)}")
    # Twilio erwartet dieses Subprotokoll
    await ws.accept(subprotocol="audio.stream.twilio.com")
    negotiated = ws.client_state  # nur zum Debuggen
    print("WS: accepted with subprotocol = audio.stream.twilio.com")

    bridge = TwilioOpenAIBridge(ws)
    try:
        await bridge.run()
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
        self.oai_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.running = True
        self._oai_reader_task: Optional[asyncio.Task] = None
        self._twilio_reader_task: Optional[asyncio.Task] = None

    # ---------- OpenAI Connect mit Backoff ----------
    async def _connect_openai(self, attempt: int = 1):
        url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
        headers = [
            ("Authorization", f"Bearer {OPENAI_API_KEY}"),
            ("OpenAI-Beta", "openai-beta.realtime-v1"),  # exakt so
        ]
        print(f"OAI: connecting → {url}")
        print("OAI: offering subprotocols: ['realtime','openai-beta.realtime-v1'] | auth header present:", bool(OPENAI_API_KEY))
        try:
            self.oai_ws = await websockets.connect(
                url,
                extra_headers=headers,
                subprotocols=["realtime", "openai-beta.realtime-v1"],
                ping_interval=20,
                ping_timeout=20,
                max_size=10 * 1024 * 1024,
            )
            print(f"OAI: connected (negotiated subprotocol: {self.oai_ws.subprotocol} )")
        except Exception as e:
            print(f"OAI: connect failed (attempt {attempt}):", repr(e))
            traceback.print_exc()
            if attempt < 3 and self.running:
                await asyncio.sleep(min(1.0 * attempt, 3.0))
                return await self._connect_openai(attempt + 1)
            raise

        # Session-Update: Nur String-Werte für Audioformate!
        session_update = {
            "type": "session.update",
            "session": {
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                # Serverseitige VAD – OpenAI stoppt Antwort wenn User spricht
                "turn_detection": {"type": "server_vad"},
                # leichte Pausen nach Antworten
                "conversation": {"max_response_output_tokens": 800},
            },
        }
        await self._oai_send_json(session_update, label="session.update")

        # Proaktiver kurzer Gruß (kannst du entfernen, wenn Twilio IVR schon spricht)
        greeting = {
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "instructions": (
                    "Willkommen bei Wowona Live. Was ist Ihr Anliegen? "
                    "Sprechen Sie ganz normal – ich höre aktiv zu."
                ),
            },
        }
        await self._oai_send_json(greeting, label="response.create (greeting)")

    async def run(self):
        # Verbinde zu OpenAI
        await self._connect_openai()

        # Starte beide Richtungen (lesen von Twilio & OpenAI)
        self._twilio_reader_task = asyncio.create_task(self._pipe_twilio_to_openai())
        self._oai_reader_task = asyncio.create_task(self._pipe_openai_to_twilio())

        # Warten bis eine Seite fertig ist
        done, pending = await asyncio.wait(
            {self._twilio_reader_task, self._oai_reader_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )

        # Wenn ein Task fehlschlägt, loggen und andere Tasks stoppen
        for task in done:
            exc = task.exception()
            if exc:
                print("Bridge task exception:", repr(exc))
                traceback.print_exc()

        self.running = False
        for task in pending:
            task.cancel()

    async def stop(self):
        self.running = False
        # OpenAI sauber schließen
        if self.oai_ws:
            try:
                await self.oai_ws.close()
            except Exception:
                pass
            self.oai_ws = None

    # ---------- Pipes ----------
    async def _pipe_twilio_to_openai(self):
        """
        Liest Text-Nachrichten vom Twilio-WS:
        - event = "media" → Base64 μ-law Frames → OpenAI input_audio_buffer.append
        - event = "stop" → Buffer commit
        """
        async for message in self.twilio_ws.iter_text():
            try:
                data = json.loads(message)
            except Exception:
                print("WS RX (non-JSON):", message[:180], "…")
                continue

            etype = data.get("event")
            if etype == "connected":
                print("WS RX:", data)
            elif etype == "start":
                print("Twilio start, streamSid:", data.get("start", {}).get("streamSid"))
            elif etype == "media":
                payload = data.get("media", {}).get("payload")
                if payload:
                    await self._oai_send_json(
                        {"type": "input_audio_buffer.append", "audio": payload},
                        label="input_audio_buffer.append"
                    )
            elif etype == "mark":
                print("WS RX mark:", data)
            elif etype == "stop":
                print("Twilio stop")
                await self._oai_send_json({"type": "input_audio_buffer.commit"}, label="input_audio_buffer.commit")
                # Option: sofort Antwort anfordern, falls nicht Auto-Commit
                await self._oai_send_json({"type": "response.create"}, label="response.create (after stop)")
            else:
                print("WS RX (other):", (json.dumps(data)[:200] + "…"))

        print("Twilio reader: stream ended")

    async def _pipe_openai_to_twilio(self):
        """
        Liest Nachrichten vom OpenAI-WS:
        - response.output_audio.delta → Audio an Twilio zurück
        - response.completed → kurze Pause für Turn-Taking
        - error → loggen (schließt Twilio nicht sofort)
        """
        if not self.oai_ws:
            print("OAI reader: socket missing, abort")
            return

        try:
            async for msg in self.oai_ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    print("OAI RX (non-JSON):", msg[:180], "…")
                    continue

                mtype = data.get("type")
                if mtype == "error":
                    print("OAI error event:", data)
                    # Nicht sofort auflegen – weiter lesen (OpenAI könnte reconnecten)
                elif mtype == "response.output_audio.delta":
                    audio = data.get("audio")
                    if audio:
                        # an Twilio schicken (Base64 μ-law payload)
                        await self.twilio_ws.send_text(json.dumps({
                            "event": "media",
                            "media": {"payload": audio}
                        }))
                elif mtype == "response.completed":
                    # kleine Sprechpause fürs Gegenüber
                    await asyncio.sleep(2.0)
                elif mtype in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"):
                    # Debug-Events (falls aktiv)
                    print("OAI RX:", data)
                else:
                    # Weitere Events ggf. loggen
                    pass
        except websockets.exceptions.ConnectionClosedError as e:
            print("OAI pipe error:", repr(e))
        except Exception as e:
            print("OAI pipe generic error:", repr(e))
            traceback.print_exc()
        finally:
            print("OpenAI reader: stream ended")

    # ---------- Helfer ----------
    async def _oai_send_json(self, payload: dict, label: str = ""):
        if not self.oai_ws:
            print(f"[WARN] OAI send skipped ({label}) – socket not open")
            return
        try:
            await self.oai_ws.send(json.dumps(payload))
            if label:
                # bei großen Logs willkürliche Kürzung
                preview = json.dumps(payload)
                if len(preview) > 300:
                    preview = preview[:300] + " …"
                print(f"OAI ⇢ {label}: {preview}")
        except Exception as e:
            print(f"OAI send error ({label}):", repr(e))
            traceback.print_exc()
