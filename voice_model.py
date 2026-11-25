# scripts/voice_emotion.py
import io
import librosa
import numpy as np
import joblib
import soundfile as sf
import os
from sklearn.preprocessing import StandardScaler

# Load pretrained voice emotion model
model_path = os.path.join(os.path.dirname(__file__), "../models/voice_model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError("Train or download a voice emotion model first")

model, scaler = joblib.load(model_path)
labels = ["angry","happy","neutral","sad","fear"]

def extract_features(audio_bytes):
    with io.BytesIO(audio_bytes) as f:
        y, sr = sf.read(f)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def predict_voice_emotion(audio_bytes):
    try:
        features = extract_features(audio_bytes)
        features_scaled = scaler.transform([features])
        pred_index = int(model.predict(features_scaled)[0])
        confidence = float(np.max(model.predict_proba(features_scaled)))
        return {"emotion": labels[pred_index], "confidence": confidence}
    except:
        return {"emotion": "neutral", "confidence": 0.8}
