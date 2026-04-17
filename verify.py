import os
import wave
import pickle
import librosa
import numpy as np
import pyaudio
import time

# --- SETTINGS ---
MODEL_FILE = "vocal_model.gmm"
THRESHOLD = -25  # Lower this (e.g., -30) if it's too strict; raise it (e.g., -20) if too loose.
FS = 44100
SECONDS = 3

def record_live():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=FS, input=True, frames_per_buffer=1024)
    
    print("\n🎤 [LISTENING] State your identity...")
    frames = []
    for _ in range(0, int(FS / 1024 * SECONDS)):
        data = stream.read(1024)
        frames.append(data)
        
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save temporary file for analysis
    temp_file = "temp_verify.wav"
    wf = wave.open(temp_file, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(FS)
    wf.writeframes(b''.join(frames))
    wf.close()
    return temp_file

def verify_voice():
    # 1. Load the trained GMM
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: Model file '{MODEL_FILE}' not found. Train the model first!")
        return

    with open(MODEL_FILE, 'rb') as f:
        gmm = pickle.load(f)

    # 2. Record live sample
    audio_path = record_live()

    # 3. Extract Features
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20).T
        
        # 4. Score against GMM
        scores = gmm.score(features)
        avg_score = np.mean(scores)
        
        print(f"\n📊 Confidence Score: {avg_score:.2f}")
        print(f"📡 Threshold Set to: {THRESHOLD}")

        # 5. The Decision
        if avg_score >= THRESHOLD:
            print("\n" + "="*30)
            print("🔓 ACCESS GRANTED")
            print(f"Welcome back, Dev.")
            print("="*30)
            # You can add a system command here, e.g., os.startfile("secret_folder")
        else:
            print("\n" + "!"*30)
            print("🔒 ACCESS DENIED")
            print("Voice Profile Mismatch.")
            print("!"*30)

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    verify_voice()
