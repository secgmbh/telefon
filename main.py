import os
import json
import asyncio
import websockets
from fastapi import FastAPI, Request
from fastapi.responses import Response, HTMLResponse
from fastapi.websockets import WebSocket
from dotenv import load_dotenv

load_dotenv()

# === Konfiguration ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_REALTIME_MODEL = "gpt-4o-realtime-preview"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY fehlt! Bitte in .env oder Render Config setzen.")

# === FastAPI ===
app = FastAPI()


@app.get("/")
async def root():
    return HTMLResponse("<h1>Service läuft ✅</h1>")


@app.post("/telefon_live")
async def telefon_live(request: Request):
    """Twilio ruft diesen Endpoint auf, um TwiML zu bekommen."""
    ws_url = str(request.url_for("twilio_stream")).replace("http", "wss")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}"/>
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept(subprotocol="audio.stream.twilio.com")
    bridge = TwilioOpenAIBridge(ws)
    try:
        await bridge.start()
    except Exception as e:
        print("Bridge Fehler:", e)
    finally:
        await bridge.stop()
        await ws.close()


# === Brücke zwischen Twilio und OpenAI ===
class TwilioOpenAIBridge:
    def __init__(self, twilio_ws: WebSocket):
        self.twilio_ws = twilio_ws
        self.oai_ws = None
        self.running = True

    async def open_openai(self):
        url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
        headers = [
            ("Authorization", f"Bearer {OPENAI_API_KEY}"),
            ("OpenAI-Beta", "openai-beta.realtime-v1"),
        ]
        self.oai_ws = await websockets.connect(
            url,
            extra_headers=headers,
            subprotocols=["realtime", "openai-beta.realtime-v1"],
            ping_interval=20,
            ping_timeout=20,
            max_size=10 * 1024 * 1024,
        )
        print("OAI: verbunden, Subprotocol =", self.oai_ws.subprotocol)

        # Session-Update mit Audioformat
        await self.oai_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "turn_detection": {"type": "server_vad"},
            }
        }))

        # Begrüßung sofort starten
        await self.oai_ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "instructions": "Willkommen bei Wowona Live. Einen Moment bitte… Was ist Ihr Anliegen?",
            }
        }))

    async def start(self):
        await self.open_openai()
        # Starte beide Pipes gleichzeitig
        await asyncio.gather(
            self._pipe_twilio_to_openai(),
            self._pipe_openai_to_twilio()
        )

    async def stop(self):
        self.running = False
        if self.oai_ws:
            await self.oai_ws.close()

    async def _pipe_twilio_to_openai(self):
        """Audio von Twilio an OpenAI senden"""
        async for message in self.twilio_ws.iter_text():
            try:
                data = json.loads(message)
                if data["event"] == "media":
                    await self.oai_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": data["media"]["payload"]
                    }))
                elif data["event"] == "stop":
                    print("Twilio stop")
                    await self.oai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            except Exception as e:
                print("Fehler in twilio->openai:", e)

    async def _pipe_openai_to_twilio(self):
        """Antworten von OpenAI an Twilio zurückschicken"""
        async for msg in self.oai_ws:
            try:
                data = json.loads(msg)
                if data.get("type") == "error":
                    print("OAI error event:", data)
                elif data.get("type") == "response.output_audio.delta":
                    audio = data["audio"]
                    await self.twilio_ws.send_text(json.dumps({
                        "event": "media",
                        "media": {"payload": audio}
                    }))
                elif data.get("type") == "response.completed":
                    # Delay, damit User sprechen kann
                    await asyncio.sleep(2.5)
            except Exception as e:
                print("Fehler in openai->twilio:", e)
