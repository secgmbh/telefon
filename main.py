import os
import json
import asyncio
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

# .env laden
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = "gpt-4o-realtime-preview"
STREAM_WSS_PATH = "/twilio-stream"

app = FastAPI()


class OpenAIBridge:
    def __init__(self, twilio_ws: WebSocket):
        self.twilio_ws = twilio_ws
        self.oai_ws = None
        self.running = True

    async def open_openai(self):
        url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

        # ✅ Korrekte Beta-Header-Konfiguration
        headers = [
            ("Authorization", f"Bearer {OPENAI_API_KEY}"),
            ("OpenAI-Beta", "realtime-v1"),
        ]

        # Beide Protokolle anbieten
        subprotocols = ["realtime", "openai-beta.realtime.v1"]

        print(f"OAI: connecting → {url}")
        self.oai_ws = await websockets.connect(
            url,
            extra_headers=headers,
            subprotocols=subprotocols,
        )
        print(f"OAI: connected (negotiated subprotocol: {self.oai_ws.subprotocol})")

        # Session anpassen → µ-law (Twilio kompatibel)
        await self.oai_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "voice": "verse",
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "conversation": "phonecall",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.6,
                    "silence_duration_ms": 1200
                }
            }
        }))
        print("OAI: session.update sent (μ-law)")

        # Proaktiver Start-Gruß
        await self.oai_ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "conversation": "phonecall",
                "instructions": "Willkommen bei wowona Live. Wie kann ich Ihnen helfen?",
                "voice": "verse"
            }
        }))
        print("OAI: proactive greeting response.create sent")

    async def _pipe_twilio_to_openai(self):
        try:
            async for msg in self.twilio_ws.iter_text():
                data = json.loads(msg)

                if data["event"] == "media":
                    await self.oai_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": data["media"]["payload"]
                    }))
                elif data["event"] == "start":
                    print(f"Twilio start, streamSid: {data['start']['streamSid']}")
                elif data["event"] == "stop":
                    print("Twilio stop")
                    await self.oai_ws.send(json.dumps({
                        "type": "input_audio_buffer.commit"
                    }))
                    await self.oai_ws.send(json.dumps({
                        "type": "response.create"
                    }))
                    break
        except Exception as e:
            print("Error in Twilio → OpenAI pipe:", e)

    async def _pipe_openai_to_twilio(self):
        try:
            async for msg in self.oai_ws:
                data = json.loads(msg)

                if data["type"] == "response.audio.delta":
                    await self.twilio_ws.send_json({
                        "event": "media",
                        "media": {"payload": data["delta"]}
                    })
                elif data["type"] == "response.completed":
                    print("OAI: response completed")
                elif data["type"] == "error":
                    print("OAI error event:", data)
        except Exception as e:
            print("Error in OpenAI → Twilio pipe:", e)

    async def start(self):
        await self.open_openai()
        await asyncio.gather(
            self._pipe_twilio_to_openai(),
            self._pipe_openai_to_twilio(),
        )


@app.get("/")
async def index():
    return PlainTextResponse("Server läuft.")


@app.post("/telefon_live")
async def telefon_live(request: Request):
    wss_url = request.url_for("twilio_stream").replace("http", "wss")
    print(f"TwiML requested. Using stream URL: {wss_url}")
    twiml = f"""
    <Response>
        <Connect>
            <Stream url="{wss_url}"/>
        </Connect>
    </Response>
    """
    return PlainTextResponse(content=twiml, media_type="application/xml")


@app.websocket(STREAM_WSS_PATH)
async def twilio_stream(websocket: WebSocket):
    await websocket.accept(subprotocol="audio.stream.twilio.com")
    print(f"WS: accepted with subprotocol = {websocket.application_state}")
    bridge = OpenAIBridge(websocket)
    await bridge.start()
