from dotenv import load_dotenv
load_dotenv()

import sounddevice as sd
from scipy.io.wavfile import write
from agent.stt import transcribe_file

SAMPLE_RATE = 16000
DURATION = 15  # seconds — adjust if your phrase is longer

def record(path="mic_test.wav"):
    print(f"🎙️  Recording for {DURATION}s... speak now.")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    write(path, SAMPLE_RATE, audio)
    print("✅ Saved:", path)
    return path

if __name__ == "__main__":
    path = record()
    result = transcribe_file(path)
    print("\nTranscript:", result.transcript)
    print("Confidence:", round(result.confidence, 3))