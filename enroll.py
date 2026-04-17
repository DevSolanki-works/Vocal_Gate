import pyaudio
import wave
import os

# --- SETTINGS ---
FS = 44100
SECONDS = 3
OUTPUT_DIR = "authorized_user_data"
NUM_SAMPLES = 5

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def record_sample(sample_num):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=FS,
        input=True,
        frames_per_buffer=1024
    )

    print(f"\n🎤 Recording sample {sample_num}/{NUM_SAMPLES} — speak naturally for {SECONDS}s...")
    frames = []
    for _ in range(0, int(FS / 1024 * SECONDS)):
        # exception_on_overflow=False avoids crashes if mic buffer fills up
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    filename = os.path.join(OUTPUT_DIR, f"sample_{sample_num}.wav")
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(FS)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"  ✅ Saved → {filename}")


def main():
    print("=" * 40)
    print(" VocalGate — Enrollment")
    print("=" * 40)
    print(f"You will record {NUM_SAMPLES} voice samples to build your vocal fingerprint.")
    print("\n💡 Tips for best accuracy:")
    print("   • Speak in a quiet room")
    print("   • Use the same phrase or count aloud each time")
    print("   • Maintain a consistent distance from the mic")
    print("   • Speak at a natural volume — not too loud, not a whisper")

    for i in range(1, NUM_SAMPLES + 1):
        if i > 1:
            input("\nPress Enter when you're ready for the next sample...")
        record_sample(i)

    print(f"\n🎉 Enrollment complete! {NUM_SAMPLES} samples saved to '{OUTPUT_DIR}/'.")
    print("   Next step: run  python train_model.py")


if __name__ == "__main__":
    main()