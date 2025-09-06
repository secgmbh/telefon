#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twilio ↔ OpenAI Realtime Bridge (Deutsch)
----------------------------------------

Diese "main.py" implementiert eine robuste Bridge zwischen Twilio Voice Media Streams
(WebSocket) und der OpenAI Realtime API. Besonderes Augenmerk liegt auf den Race-Conditions
rund um `input_audio_buffer.commit` und das Schließen des WebSockets.

Wichtigste Eigenschaften:
- Commit-Gating: `input_audio_buffer.commit` wird nur gesendet, wenn genug Audio im Puffer ist,
  die Twilio-Stream-Verbindung aktiv ist **und** die OpenAI-WS-Verbindung offen ist.
- Kein "finaler Flush" beim Stoppen: Beim Twilio-`stop`-Event werden Restbytes verworfen und
  es werden keine Commits mehr versucht.
- Sauberes Shutdown-Handling, damit keine Writes mehr in eine schließende Transport/WS laufen.
- μ-law (8 kHz) → PCM16 (24 kHz) Resampling via `audioop.ratecv` (reine Stdlib, kein numpy nötig).
- Frühe Begrüßungsantwort direkt nach Aufbau der OpenAI-Session (Deutsch).

Voraussetzungen:
- Python 3.10+
- Abhängigkeiten: aiohttp

  pip install aiohttp

Umgebung:
- OPENAI_API_KEY           (erforderlich)
- OPENAI_REALTIME_MODEL    (optional, Standard siehe DEFAULT_MODEL)
- HOST                     (optional, Standard: 0.0.0.0)
- PORT                     (optional, Standard: 8080)
- MIN_COMMIT_MS            (optional, Standard: 100)
- TWILIO_ENCODING          (optional: "mulaw" | "pcm16"; Standard: "mulaw")

Twilio-Konfiguration:
- Voice Media Streams auf euren /twilio WebSocket-Endpunkt zeigen lassen.
- (Optional) Stream-Status Webhook auf /stream-status konfigurieren.

Hinweis: Diese Datei fokussiert die Eingangsrichtung (Anrufer → OpenAI). Das Zurücksenden
von Audio an Twilio ist projektabhängig und kann separat ergänzt werden.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import sys
import signal
import audioop
from typing import Optional, Tuple

import aiohttp
from aiohttp import web

# -------------------------------
# Konstante Standardeinstellungen
# -------------------------------
DEFAULT_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
OPENAI_REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={DEFAULT_MODEL}"
BYTES_PER_SAMPLE = 2  # PCM16
DST_SAMPLE_RATE = 24_000
SRC_SAMPLE_RATE = 8_000
CHANNELS = 1

MIN_COMMIT_MS = int(os.getenv("MIN_COMMIT_MS", "100"))
TWILIO_ENCODING = os.getenv("TWILIO_ENCODING", "mulaw").lower()  # "mulaw" oder "pcm16"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))

# ----------------------
# Kleine Hilfsfunktionen
# ----------------------

def log(*args):
    print(*args, flush=True)


class AudioConverter:
    """Wandelt Twilio-Chunk (Base64) nach PCM16@24k um.

    - Erwartet Twilio-Input (typisch μ-law 8 kHz) oder PCM16 8 kHz.
    - Benutzt audioop.ratecv zum Resampling (Stateful je Verbindung).
    """

    def __init__(self, src_rate: int = SRC_SAMPLE_RATE, dst_rate: int = DST_SAMPLE_RATE,
                 encoding: str = TWILIO_ENCODING):
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self.encoding = encoding
        self._ratecv_state = None  # State zwischen Chunks beibehalten

    def convert_b64_chunk(self, payload_b64: str) -> bytes:
        raw = base64.b64decode(payload_b64)
        if not raw:
            return b""
        if self.encoding == "mulaw":
            # μ-law (8-bit) → PCM16 (2 bytes)
            pcm16_8k = audioop.ulaw2lin(raw, BYTES_PER_SAMPLE)
        else:
            # Angenommen bereits PCM16@8k
            pcm16_8k = raw
        # Resample 8 kHz → 24 kHz (Mono, 16 Bit)
        pcm16_24k, self._ratecv_state = audioop.ratecv(
            pcm16_8k, BYTES_PER_SAMPLE, CHANNELS, self.src_rate, self.dst_rate, self._ratecv_state
        )
        return pcm16_24k


class OpenAIBridge:
    """Kapselt die OpenAI-Realtime-WebSocket-Verbindung und Commit-Logik."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, min_commit_ms: int = MIN_COMMIT_MS):
        self.api_key = api_key
        self.model = model
        self.url = f"wss://api.openai.com/v1/realtime?model={model}"

        self.session: Optional[aiohttp.ClientSession] = None
        self.oai_ws: Optional[aiohttp.ClientWebSocketResponse] = None

        self._closed = False
        self.active = False  # True zwischen Twilio start/stop
        self._uncommitted_bytes = 0

        self._bytes_per_ms = DST_SAMPLE_RATE * BYTES_PER_SAMPLE // 1000  # 24k * 2 / 1000 = 48
        self._commit_threshold = max(1, int(self._bytes_per_ms * min_commit_ms))

        self.commit_task: Optional[asyncio.Task] = None
        self.reader_task: Optional[asyncio.Task] = None

    # -------------------
    # Lifecycle / Connect
    # -------------------
    async def connect(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        log("Verbinde zur OpenAI Realtime WS …")
        self.oai_ws = await self.session.ws_connect(self.url, headers=headers)
        log("OpenAI WebSocket verbunden.")
        # Nebenläufig: Eingehende Events lesen (optional, hier nur Logging)
        self.reader_task = asyncio.create_task(self._oai_reader())
        # Nebenläufig: Commit-Ticker
        self.commit_task = asyncio.create_task(self._commit_ticker())

    async def start(self):
        self.active = True

    # --------------
    # Audio Handling
    # --------------
    async def append_audio(self, pcm16_24k: bytes):
        if not pcm16_24k:
            return
        if not self._can_send():
            return
        payload = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16_24k).decode("ascii"),
        }
        await self._oai_send_json(payload)
        # Uncommitted zählen – Commit erfolgt im Ticker
        self._uncommitted_bytes += len(pcm16_24k)

    async def send_greeting(self):
        """Erzwingt eine frühe, kurze deutsche Begrüßung/Prompt."""
        if not self._can_send():
            return
        payload = {
            "type": "response.create",
            "response": {
                "instructions": "Begrüße den Anrufer kurz und höflich auf Deutsch und frage, wie du helfen kannst.",
            },
        }
        await self._oai_send_json(payload)

    # ---------
    # Interna
    # ---------
    async def _oai_reader(self):
        try:
            async for msg in self.oai_ws:  # type: ignore[arg-type]
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        log("[OpenAI] Textframe (kein JSON):", msg.data)
                        continue
                    t = data.get("type")
                    # Hier könnt ihr spezifisch auf Events reagieren (z. B. Audio-Ausgabe, Logs, etc.)
                    log(f"[OpenAI] Event: {t}")
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    log(f"[OpenAI] Binärframe ({len(msg.data)} Bytes)")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    log("[OpenAI] WS-Fehler:", self.oai_ws.exception() if self.oai_ws else None)
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log("[OpenAI] Reader error:", repr(e))

    async def _commit_ticker(self):
        try:
            while not self._closed:
                await asyncio.sleep(0.25)
                if (
                    self.active
                    and self.oai_ws is not None
                    and not self.oai_ws.closed
                    and self._uncommitted_bytes >= self._commit_threshold
                ):
                    await self._oai_send_json({"type": "input_audio_buffer.commit"})
                    # Nach erfolgreichem Commit Zähler zurücksetzen
                    self._uncommitted_bytes = 0
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log("Commit-Ticker Fehler:", repr(e))

    def _can_send(self) -> bool:
        return (
            not self._closed
            and self.active
            and self.oai_ws is not None
            and not self.oai_ws.closed
        )

    async def _oai_send_json(self, payload: dict):
        """Sicheres Senden – mit Extra-Guard gegen leere Commits."""
        if not self._can_send():
            return
        # Niemals leeren Commit senden
        if (
            payload.get("type") == "input_audio_buffer.commit"
            and self._uncommitted_bytes < self._commit_threshold
        ):
            return
        try:
            await self.oai_ws.send_str(json.dumps(payload))
        except Exception as e:
            log(f"OAI send error ({payload.get('type','?')}):", repr(e))

    async def close(self):
        if self._closed:
            return
        self._closed = True
        self.active = False
        self._uncommitted_bytes = 0
        # Tasks stoppen
        for task in (self.commit_task, self.reader_task):
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        # WS schließen
        try:
            if self.oai_ws and not self.oai_ws.closed:
                await self.oai_ws.close()
        except Exception:
            pass
        # Session schließen
        try:
            if self.session:
                await self.session.close()
        except Exception:
            pass


# -----------------------
# Twilio-WS-Handler (Aiohttp)
# -----------------------
class TwilioWSHandler:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.converter = AudioConverter()
        self.bridge: Optional[OpenAIBridge] = None
        self.stream_sid: Optional[str] = None

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        log("Twilio: WS verbunden")

        # Bridge für diesen Call anlegen
        self.bridge = OpenAIBridge(api_key=self.api_key)
        await self.bridge.connect()

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        log("Twilio: Unbekannter Textframe:", msg.data)
                        continue

                    event_type = data.get("event")
                    if event_type == "start":
                        self.stream_sid = data.get("start", {}).get("streamSid")
                        log(f"Twilio start: {self.stream_sid}")
                        await self.bridge.start()
                        # Frühe Begrüßung triggern
                        await self.bridge.send_greeting()

                    elif event_type == "media":
                        media = data.get("media", {})
                        payload_b64 = media.get("payload")
                        if payload_b64 and self.bridge:
                            pcm = self.converter.convert_b64_chunk(payload_b64)
                            await self.bridge.append_audio(pcm)

                    elif event_type == "mark":
                        # Ignorieren oder für Debug-Zwecke verwenden
                        pass

                    elif event_type == "stop":
                        log("Twilio stop (Call beendet)")
                        # Inaktiv markieren & Rest verwerfen (kein finaler Commit)
                        if self.bridge:
                            self.bridge.active = False
                            self.bridge._uncommitted_bytes = 0
                        break

                    else:
                        log("Twilio: Unbekanntes Event:", event_type)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    log("Twilio WS-Fehler:", ws.exception())
                    break
        finally:
            # Sicher schließen
            if self.bridge:
                await self.bridge.close()
            await ws.close()
            log("Twilio: Verbindung geschlossen")

        return ws


# -----------------------
# HTTP-Server / Routing
# -----------------------
async def stream_status(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    log("Twilio StreamStatus:", payload)
    return web.Response(text="ok")


async def health(request: web.Request) -> web.Response:
    return web.Response(text="ok")


def build_app() -> web.Application:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log("FEHLER: OPENAI_API_KEY ist nicht gesetzt.")
        sys.exit(1)

    app = web.Application()

    handler = TwilioWSHandler(api_key=api_key)
    app.router.add_get("/twilio", handler.handle)
    app.router.add_post("/stream-status", stream_status)
    app.router.add_get("/", health)

    return app


# ---------------
# Main-Entry-Point
# ---------------
async def _run():
    app = build_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT)
    await site.start()
    log(f"Server läuft auf http://{HOST}:{PORT}")
    # Warten bis SIGINT/SIGTERM
    stop_event = asyncio.Event()

    def _stop(*_):
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _stop)

    await stop_event.wait()
    log("Shutdown …")
    await runner.cleanup()


def main():
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
