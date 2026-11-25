# scripts/camera_emotion.py
import cv2
import numpy as np
from fer import FER

# Initialize FER detector
detector = FER(mtcnn=True)  # mtcnn=True gives better face detection

def predict_camera_emotion(img: np.ndarray):
    """
    Predict emotion from a given OpenCV image.
    Returns a dict: {"emotion": <str>, "confidence": <float>}
    """
    if img is None:
        return {"emotion": "neutral", "confidence": 0.0}

    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect emotions
    results = detector.detect_emotions(rgb_img)

    if not results:
        # No face detected
        return {"emotion": "neutral", "confidence": 0.8}

    # Pick the first face detected
    emotions = results[0]["emotions"]
    # Get the emotion with max confidence
    emotion_label = max(emotions, key=emotions.get)
    confidence = emotions[emotion_label]

    return {"emotion": emotion_label, "confidence": round(confidence, 2)}

# Quick test
if __name__ == "__main__":
    img_path = "test_face.jpg"  # Put a test image here
    img = cv2.imread(img_path)
    result = predict_camera_emotion(img)
    print(result)
