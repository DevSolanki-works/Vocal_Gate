import pyaudio
import wave
import numpy as np
from scipy.io.wavfile import write
import os
import time

# --- SETTINGS ---
FS = 44100  # Sample rate (standard high-fidelity)
SECONDS = 3 # Duration of each clip
OUTPUT_DIR = "authorized_user_data"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def record_sample(sample_num):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    
    print(f"\n🎤 Recording sample {sample_num}/5...")
    frames = []
    for _ in range(0, int(44100 / 1024 * 3)): # 3 seconds
        data = stream.read(1024)
        frames.append(data)
        
    stream.stop_stream()
    stream.close()
    p.terminate()

    filename = os.path.join(OUTPUT_DIR, f"sample_{sample_num}.wav")
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"✅ Saved to {filename}")

def main():
    print("--- VocalGate Enrollment System ---")
    print(f"You will record {5} samples to build your vocal fingerprint.")
    
    for i in range(1, 6):
        record_sample(i)
        if i < 5:
            input("\nPress Enter when you're ready for the next sample...")
            
    print("\n🎉 Enrollment complete! You have built your raw dataset.")

if __name__ == "__main__":
    main()
