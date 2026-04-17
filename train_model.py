import os
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pickle

# --- SETTINGS ---
DATA_DIR = "authorized_user_data"
MODEL_FILE = "vocal_model.gmm"
N_MFCC = 20
TARGET_SR = 22050  # Consistent resample rate for both train + verify


def extract_features(audio_path):
    """
    Load audio, trim silence, then return stacked MFCC + delta + delta-delta.
    Shape: (T, 60)  — T frames, 60 features each.
    """
    audio, sr = librosa.load(audio_path, sr=TARGET_SR)

    # Remove leading/trailing silence so the model learns voice, not silence
    audio, _ = librosa.effects.trim(audio, top_db=25)

    if len(audio) == 0:
        raise ValueError(f"Audio file '{audio_path}' is silent after trimming.")

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    delta1 = librosa.feature.delta(mfccs)          # 1st-order dynamics
    delta2 = librosa.feature.delta(mfccs, order=2) # 2nd-order dynamics

    # Stack → (60, T), transpose → (T, 60)
    combined = np.vstack([mfccs, delta1, delta2])
    return combined.T


def train_vocal_model():
    print("🧠 Starting Feature Extraction...")
    all_features = []

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.wav')])
    if not files:
        print(f"❌ No .wav files found in '{DATA_DIR}'. Run enroll.py first!")
        return

    for file in files:
        path = os.path.join(DATA_DIR, file)
        print(f"  📂 Processing {file}...")
        try:
            features = extract_features(path)
            all_features.append(features)
        except Exception as e:
            print(f"  ⚠️  Skipped {file}: {e}")

    if not all_features:
        print("❌ No usable audio files. Re-run enroll.py.")
        return

    X = np.vstack(all_features)
    print(f"  📊 Extracted {X.shape[0]} frames × {X.shape[1]} features.")

    # --- KEY FIX 1: Normalize features (zero mean, unit variance) ---
    # Without this, raw MFCCs have wildly different scales that destroy GMM scores.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- KEY FIX 2: Fewer components to avoid overfitting a small dataset ---
    # 16 components for ~5×3s clips = almost certain overfit. Use 8 max.
    n_components = min(8, max(2, X.shape[0] // 50))
    print(f"  🏗️  Training GMM with {n_components} components...")
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='diag',
        n_init=5,         # Multiple random restarts for a more stable fit
        max_iter=300,
        random_state=42
    )
    gmm.fit(X_scaled)

    # --- Save both the model AND the scaler together ---
    # verify.py MUST apply the same scaler — models are paired.
    model_bundle = {'gmm': gmm, 'scaler': scaler}
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model_bundle, f)

    # Print the training score so you know where to set the threshold
    train_score = gmm.score(X_scaled)
    suggested_threshold = train_score - 8.0   # ~8 units of margin below training score

    print(f"\n✅ Model saved to '{MODEL_FILE}'")
    print(f"📊 Training score (your voice on your own model): {train_score:.2f}")
    print(f"💡 Suggested THRESHOLD for verify.py: {suggested_threshold:.1f}")
    print(f"   (Open verify.py and set  THRESHOLD = {suggested_threshold:.1f})")


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        print(f"❌ Error: No files found in '{DATA_DIR}'. Run enroll.py first!")
    else:
        train_vocal_model()
