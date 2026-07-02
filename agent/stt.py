import asyncio
from dataclasses import dataclass, field
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

FLUX_MODEL = "flux-general-multi"
LANGUAGE_HINTS = ["fr", "en"]

@dataclass
class TranscriptionResult:
    transcript: str
    confidence: float                 
    low_confidence_words: list[str] = field(default_factory=list)


async def transcribe_stream(
    pcm_chunks,
    stop_event: asyncio.Event | None = None,
    timeout: float = 30.0,
    ) -> TranscriptionResult:
        client = AsyncDeepgramClient()
        parts, low_conf, confidences = [], [], []
        done = asyncio.Event()

        async with client.listen.v2.connect(
            model=FLUX_MODEL,
            encoding="linear16",
            sample_rate=16000,
            request_options={"additional_query_parameters": {"language_hint": LANGUAGE_HINTS}},
        ) as connection:

            def on_message(message) -> None:
                if getattr(message, "transcript", None):
                    parts.append(message.transcript)
                for w in getattr(message, "words", None) or []:
                    confidences.append(w.confidence)
                    if w.confidence < 0.75:
                        low_conf.append(w.word)
                if getattr(message, "type", None) == "EndOfTurn":
                    done.set()
                    if stop_event is not None:
                        stop_event.set()

            connection.on(EventType.MESSAGE, on_message)
            listen_task = asyncio.create_task(connection.start_listening())

            async def pump():
                async for chunk in pcm_chunks:
                    if done.is_set():
                        break
                    await connection._send(chunk)

            pump_task = asyncio.create_task(pump())
            try:
                await asyncio.wait_for(done.wait(), timeout=timeout)
            finally:
                pump_task.cancel()
                listen_task.cancel()

        return TranscriptionResult(
            transcript=" ".join(parts).strip(),
            confidence=min(confidences) if confidences else 0.0,
            low_confidence_words=low_conf,
        )