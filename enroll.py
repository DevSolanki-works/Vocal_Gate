import sounddevice as sd
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
    print(f"\n🎤 Preparing to record sample {sample_num}/5...")
    print("Wait for the 'RECORDING' prompt and say: 'IDENTITY VERIFIED'")
    time.sleep(1)
    
    print("🔴 RECORDING...")
    recording = sd.rec(int(SECONDS * FS), samplerate=FS, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("✅ DONE.")
    
    filename = os.path.join(OUTPUT_DIR, f"sample_{sample_num}.wav")
    write(filename, FS, recording)
    print(f"💾 Saved to {filename}")

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
