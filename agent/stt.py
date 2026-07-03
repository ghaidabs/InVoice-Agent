import os
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import sounddevice as sd
from deepgram import DeepgramClient
from deepgram.core.events import EventType

DEFAULT_MODEL = "nova-3"
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 480


@dataclass
class StreamingTranscript:
    transcript: str
    is_final: bool
    speech_final: bool
    confidence: float
    language: Optional[str] = None


@dataclass
class _StreamState:
    connection: object = None
    mic_stream: object = None
    audio_thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    audio_queue: "queue.Queue[bytes]" = field(default_factory=queue.Queue)


_client: Optional[DeepgramClient] = None


def _get_client() -> DeepgramClient:
    global _client
    if _client is None:
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set (check .env)")
        _client = DeepgramClient(api_key=api_key)
    return _client


class StreamingSTT:
    def __init__(
        self,
        on_transcript: Callable[[StreamingTranscript], None],
        on_error: Optional[Callable[[Exception], None]] = None,
        on_speech_started: Optional[Callable[[], None]] = None,
        on_utterance_end: Optional[Callable[[], None]] = None,
        model: str = DEFAULT_MODEL,
        sample_rate: int = SAMPLE_RATE,
        ):
            self._on_transcript = on_transcript
            self._on_error = on_error
            self._on_speech_started = on_speech_started
            self._on_utterance_end = on_utterance_end
            self._model = model
            self._sample_rate = sample_rate
            self._state = _StreamState()


    def start(self) -> None:
        client = _get_client()

        connect_kwargs = dict(
            model=self._model,
            encoding="linear16",
            sample_rate=self._sample_rate,
            channels=CHANNELS,
            smart_format=True,
            punctuate=True,
            interim_results=True,
            utterance_end_ms="5000",
            vad_events=True,
            numerals=True,
            language="multi",
        )

        connection_cm = client.listen.v1.connect(**connect_kwargs)
        connection = connection_cm.__enter__()
        self._state.connection = connection
        self._state.connection_cm = connection_cm

        connection.on(EventType.OPEN, lambda _: None)
        connection.on(EventType.MESSAGE, self._handle_message)
        connection.on(EventType.ERROR, self._handle_error)
        connection.on(EventType.CLOSE, lambda _: None)

        listen_thread = threading.Thread(target=connection.start_listening, daemon=True)
        listen_thread.start()
        self._state.listen_thread = listen_thread
        print("DEBUG: past start_listening")   

        self._state.stop_event.clear()
        self._start_microphone()
        print("DEBUG: mic started")           


    def stop(self) -> None:
        self._state.stop_event.set()

        if self._state.mic_stream is not None:
            try:
                self._state.mic_stream.stop()
                self._state.mic_stream.close()
            except Exception:
                pass
            self._state.mic_stream = None

        if self._state.audio_thread is not None:
            self._state.audio_thread.join(timeout=2)
            self._state.audio_thread = None

        connection = self._state.connection
        if connection is not None:
            try:
                connection.send_close_stream()
            except Exception:
                pass
            try:
                self._state.connection_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._state.connection = None


    def _start_microphone(self) -> None:
        def audio_callback(indata, frames, time_info, status):
            if status:
                pass
            self._state.audio_queue.put(bytes(indata))

        stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=BLOCKSIZE,
            channels=CHANNELS,
            dtype="int16",
            callback=audio_callback,
        )
        stream.start()
        self._state.mic_stream = stream

        def sender_loop():
            while not self._state.stop_event.is_set():
                try:
                    chunk = self._state.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    self._state.connection.send_media(chunk)
                except Exception as exc:
                    self._safe_call(self._on_error, exc)
                    break

        thread = threading.Thread(target=sender_loop, daemon=True)
        thread.start()
        self._state.audio_thread = thread

    def _handle_message(self, message) -> None:
        msg_type = getattr(message, "type", None)

        if msg_type == "Results":
            alt = message.channel.alternatives[0]
            transcript_text = alt.transcript
            if not transcript_text:
                return
            result = StreamingTranscript(
                transcript=transcript_text,
                is_final=bool(getattr(message, "is_final", False)),
                speech_final=bool(getattr(message, "speech_final", False)),
                confidence=float(getattr(alt, "confidence", 0.0)),
                language=(getattr(alt, "languages", None) or [None])[0],
            )
            self._safe_call(self._on_transcript, result)

        elif msg_type == "SpeechStarted":
            self._safe_call(self._on_speech_started)

        elif msg_type == "UtteranceEnd":
            self._safe_call(self._on_utterance_end)

    def _handle_error(self, error) -> None:
        self._safe_call(self._on_error, error)

    @staticmethod
    def _safe_call(fn, *args):
        if fn is None:
            return
        try:
            fn(*args)
        except Exception:
            pass