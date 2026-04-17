import os
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle

# --- SETTINGS ---
DATA_DIR = "authorized_user_data"
MODEL_FILE = "vocal_model.gmm"

def extract_features(audio_path):
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Extract MFCCs (20 features is a good standard)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    
    # Transpose so rows are time-steps and columns are features
    return mfccs.T

def train_vocal_model():
    print("🧠 Starting Feature Extraction...")
    all_features = []
    
    # Get all .wav files from the directory
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.wav')]
    
    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        print(f"📂 Processing {file}...")
        features = extract_features(file_path)
        all_features.append(features)
    
    # Combine all features into one massive matrix
    X = np.vstack(all_features)
    print(f"📊 Extracted {X.shape[0]} feature vectors.")

    # Train the Gaussian Mixture Model
    # 16 components is usually enough to capture a single person's voice "cloud"
    print("🏗️ Training Gaussian Mixture Model...")
    gmm = GaussianMixture(n_components=16, covariance_type='diag', n_init=3)
    gmm.fit(X)
    
    # Save the model to disk
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(gmm, f)
    
    print(f"✅ Success! Your vocal fingerprint is saved as '{MODEL_FILE}'.")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        print(f"❌ Error: No files found in {DATA_DIR}. Run enroll.py first!")
    else:
        train_vocal_model()


