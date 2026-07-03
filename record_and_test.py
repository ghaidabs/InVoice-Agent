# test_streaming_stt.py
from dotenv import load_dotenv
load_dotenv()

from agent.stt import StreamingSTT, StreamingTranscript

def on_transcript(t: StreamingTranscript):
    tag = "FINAL" if t.speech_final else ("interim" if not t.is_final else "final-ish")
    lang = f" [{t.language}]" if t.language else ""
    print(f"[{tag}]{lang} {t.transcript}")

def on_error(err):
    print(f"ERROR: {err}")

def on_speech_started():
    print("... speech started ...")

def on_utterance_end():
    print("... utterance end ...")

if __name__ == "__main__":
    stt = StreamingSTT(
        on_transcript=on_transcript,
        on_error=on_error,
        on_speech_started=on_speech_started,
        on_utterance_end=on_utterance_end,
    )
    stt.start()
    print("🎙️  Listening... speak in French or English. Press Enter to stop.\n")
    input()
    stt.stop()
    print("stopped.")