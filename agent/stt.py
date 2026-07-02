import os
from dataclasses import dataclass
from pathlib import Path
from deepgram import DeepgramClient

DEFAULT_MODEL = "nova-3"
DEFAULT_LANGUAGE = "fr"

_client = None

def _get_client() -> DeepgramClient:
    global _client
    if _client is None:
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set (check .env)")
        _client = DeepgramClient(api_key=api_key)
    return _client


@dataclass
class TranscriptionResult:
    transcript: str
    confidence: float


def transcribe_file(audio_path: str | Path) -> TranscriptionResult:
    client = _get_client()
    with open(audio_path, "rb") as f:
        response = client.listen.v1.media.transcribe_file(
            request=f,
            model=DEFAULT_MODEL,
            language=DEFAULT_LANGUAGE,
            smart_format=True,
            punctuate=True,
            numerals=True,
        )
    alt = response.results.channels[0].alternatives[0]
    return TranscriptionResult(transcript=alt.transcript, confidence=alt.confidence)