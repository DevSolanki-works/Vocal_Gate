import os
import wave
import pickle
import librosa
import numpy as np
import pyaudio

# --- SETTINGS ---
MODEL_FILE = "vocal_model.gmm"
# Set this to the value train_model.py printed as "Suggested THRESHOLD"
# After re-training, the script will tell you exactly what to put here.
THRESHOLD = -80   # <-- UPDATE after re-running train_model.py
FS = 44100
SECONDS = 3
TARGET_SR = 22050  # Must match train_model.py
N_MFCC = 20


def extract_features(audio_path):
    """
    Identical feature pipeline to train_model.py.
    If this function differs from training, the scores will be garbage.
    """
    audio, sr = librosa.load(audio_path, sr=TARGET_SR)
    audio, _ = librosa.effects.trim(audio, top_db=25)

    if len(audio) == 0:
        raise ValueError("Recorded audio is silent — please speak clearly into the mic.")

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    delta1 = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)

    combined = np.vstack([mfccs, delta1, delta2])
    return combined.T


def record_live():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=FS,
        input=True,
        frames_per_buffer=1024
    )

    print("\n🎤 [LISTENING] State your identity...")
    frames = []
    for _ in range(0, int(FS / 1024 * SECONDS)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    temp_file = "temp_verify.wav"
    wf = wave.open(temp_file, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(FS)
    wf.writeframes(b''.join(frames))
    wf.close()
    return temp_file


def verify_voice():
    # 1. Load model bundle (GMM + scaler)
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Model file '{MODEL_FILE}' not found. Run train_model.py first!")
        return

    with open(MODEL_FILE, 'rb') as f:
        bundle = pickle.load(f)

    # Support both old (bare GMM) and new (dict) formats
    if isinstance(bundle, dict):
        gmm = bundle['gmm']
        scaler = bundle['scaler']
    else:
        gmm = bundle
        scaler = None
        print("⚠️  Old model format detected. Please re-run train_model.py for accurate results.")

    # 2. Record live audio
    audio_path = record_live()

    try:
        # 3. Extract features — identical pipeline to training
        features = extract_features(audio_path)

        # 4. Apply the SAME scaler used during training (KEY FIX)
        if scaler is not None:
            features = scaler.transform(features)

        # 5. Score — gmm.score() already returns the mean log-likelihood per frame
        score = gmm.score(features)

        print(f"\n📊 Confidence Score : {score:.2f}")
        print(f"📡 Threshold        : {THRESHOLD}")

        # 6. Decision
        if score >= THRESHOLD:
            print("\n" + "=" * 30)
            print("🔓 ACCESS GRANTED")
            print("Welcome back, Dev.")
            print("=" * 30)
            # Uncomment to trigger a system action on success:
            # import subprocess; subprocess.Popen(["explorer", "secret_folder"])
        else:
            gap = THRESHOLD - score
            print("\n" + "!" * 30)
            print("🔒 ACCESS DENIED")
            print(f"Voice Profile Mismatch. (missed by {gap:.1f})")
            print("!" * 30)
            print("\n💡 If this is your voice and it keeps failing:")
            print("   1. Re-run enroll.py in a quiet room")
            print("   2. Re-run train_model.py and update THRESHOLD to its suggestion")

    except Exception as e:
        print(f"❌ An error occurred during analysis: {e}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    verify_voice()
