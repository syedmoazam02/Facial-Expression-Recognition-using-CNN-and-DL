import os
import base64
import numpy as np
from flask import Flask, request, render_template, send_from_directory, jsonify
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from werkzeug.utils import secure_filename

# ---------------- CONFIG -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotiondetector.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Labels aligned with mini_XCEPTION FER-2013 model
LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
INPUT_SHAPE = (64, 64)

EMOJI_MAP = {
    'angry': '😡',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'sad': '😢',
    'surprise': '😲',
    'neutral': '😐'
}
# -----------------------------------------

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
print(f"Loading emotion model from {MODEL_PATH}...")
model = load_model(MODEL_PATH, compile=False)
print("Emotion model loaded successfully.")

# Helper: check allowed file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper: preprocess image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=INPUT_SHAPE, color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def run_prediction(filepath):
    """Run model prediction on preprocessed image and format result breakdown."""
    img_array = preprocess_image(filepath)
    preds = model.predict(img_array, verbose=0)[0]
    
    top_idx = int(np.argmax(preds))
    top_label = LABELS[top_idx]
    top_confidence = round(float(preds[top_idx]) * 100, 1)

    breakdown = []
    for idx, label in enumerate(LABELS):
        score = round(float(preds[idx]) * 100, 1)
        breakdown.append({
            'label': label,
            'name': label.capitalize(),
            'emoji': EMOJI_MAP.get(label, '🙂'),
            'score': score
        })
    
    # Sort breakdown by confidence descending
    breakdown.sort(key=lambda x: x['score'], reverse=True)

    return {
        'label': top_label,
        'title': top_label.capitalize(),
        'confidence': top_confidence,
        'emoji': EMOJI_MAP.get(top_label, '🙂'),
        'breakdown': breakdown
    }

# Routes
@app.route('/')
def home():
    return render_template(
        "result.html",
        result=None,
        confidence=None,
        emoji=None,
        breakdown=None,
        image_url=None
    )

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("result.html", result="No file uploaded", image_url=None)

    file = request.files['file']
    if not file or file.filename == '':
        return render_template("result.html", result="No file selected", image_url=None)

    if allowed_file(file.filename):
        safe_name = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        file.save(filepath)

        pred_data = run_prediction(filepath)
        image_url = f"/uploads/{safe_name}"

        return render_template(
            "result.html",
            result=pred_data['label'],
            confidence=pred_data['confidence'],
            emoji=pred_data['emoji'],
            breakdown=pred_data['breakdown'],
            image_url=image_url
        )
    else:
        return render_template("result.html", result="Invalid file type (png, jpg, jpeg, webp allowed)", image_url=None)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint supporting JSON base64 webcam capture or multipart upload."""
    try:
        # 1. Base64 snapshot from webcam
        if request.is_json and 'image_base64' in request.json:
            data = request.json['image_base64']
            if ',' in data:
                data = data.split(',', 1)[1]
            img_bytes = base64.b64decode(data)
            filename = f"capture_{os.urandom(6).hex()}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'wb') as f:
                f.write(img_bytes)

            pred_data = run_prediction(filepath)
            pred_data['image_url'] = f"/uploads/{filename}"
            return jsonify({'success': True, 'data': pred_data})

        # 2. File upload in multipart
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                safe_name = f"{os.urandom(4).hex()}_{secure_filename(file.filename)}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
                file.save(filepath)

                pred_data = run_prediction(filepath)
                pred_data['image_url'] = f"/uploads/{safe_name}"
                return jsonify({'success': True, 'data': pred_data})

        return jsonify({'success': False, 'error': 'No valid file or image data provided'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("Starting Facial Emotion Recognition Server on http://127.0.0.1:5000 ...")
    app.run(debug=True, port=5000)
