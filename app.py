import os
import sys
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

# Add scripts folder to path
scripts_folder = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_folder not in sys.path:
    sys.path.insert(0, scripts_folder)

from scripts.text_model import predict_text_emotion
from scripts.voice_model import predict_voice_emotion
from scripts.camera_model import predict_camera_emotion

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_text', methods=['POST'])
def predict_text():
    txt = request.form.get('text_input', '').strip()
    if not txt:
        return jsonify({'error': 'Please enter something first!'})
    result = predict_text_emotion(txt)
    result['source'] = 'text'
    return jsonify(result)

@app.route('/predict_voice', methods=['POST'])
def predict_voice():
    data = request.get_json()
    audio_b64 = data.get('voice_data', '')
    if not audio_b64:
        return jsonify({'error': 'No audio received'})
    audio_bytes = base64.b64decode(audio_b64.split(',')[-1])
    result = predict_voice_emotion(audio_bytes)
    result['source'] = 'voice'
    return jsonify(result)

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    data = request.get_json()
    img_b64 = data.get('image_data', '')
    if not img_b64:
        return jsonify({'error': 'No image received'})
    img_bytes = base64.b64decode(img_b64.split(',')[-1])
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image data'})
    result = predict_camera_emotion(img)
    result['source'] = 'camera'
    return jsonify(result)

if __name__ == '__main__':
    print(f"Scripts folder: {scripts_folder}")
    print("Starting AuraSense Flask app...")
    app.run(debug=True)
