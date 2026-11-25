# scripts/text_emotion.py
from transformers import pipeline

# Pretrained transformer for emotion detection
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def predict_text_emotion(text: str):
    try:
        scores = emotion_classifier(text)[0]
        scores = sorted(scores, key=lambda x: x['score'], reverse=True)
        top = scores[0]
        return {"emotion": top['label'], "confidence": float(top['score'])}
    except:
        return {"emotion": "neutral", "confidence": 0.8}
