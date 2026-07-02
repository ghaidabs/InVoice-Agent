from dotenv import load_dotenv
load_dotenv()

import asyncio
import sounddevice as sd
from agent.stt import transcribe_stream

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 80
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 1280 frames = 80ms, per Flux's requirement
MAX_SECONDS = 30  # safety cutoff if EndOfTurn never fires


async def mic_chunks(stop_event: asyncio.Event, loop: asyncio.AbstractEventLoop):
    """Yields raw linear16 PCM chunks from the mic until stop_event is set."""
    q: asyncio.Queue = asyncio.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print("⚠️ ", status)
        loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK_FRAMES,
        callback=callback,
    ):
        while not stop_event.is_set():
            try:
                yield await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue


async def main():
    print("🎙️  Speak now — Flux stops automatically once it detects end of turn.")
    stop_event = asyncio.Event()
    loop = asyncio.get_event_loop()

    result = await transcribe_stream(
        mic_chunks(stop_event, loop),
        stop_event=stop_event,
        timeout=MAX_SECONDS,
    )

    print("\nTranscript:", result.transcript or "(empty — nothing recognized)")
    print("Confidence (min word):", round(result.confidence, 3))
    if result.low_confidence_words:
        print("⚠️  Low-confidence words:", ", ".join(result.low_confidence_words))
    else:
        print("✅  No low-confidence words flagged.")


if __name__ == "__main__":
    asyncio.run(main())