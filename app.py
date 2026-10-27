import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename

# ---------------- CONFIG -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "emotiondetector.tflite")
H5_MODEL_PATH = os.path.join(BASE_DIR, "emotiondetector.h5")

# On Vercel / serverless, filesystem is read-only except /tmp
if os.environ.get('VERCEL'):
    UPLOAD_FOLDER = "/tmp/uploads"
else:
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

try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except Exception:
    pass

# Load model: prioritize lightweight LiteRT/TFLite (fits in Vercel 500MB limit)
interpreter = None
input_details = None
output_details = None
h5_model = None

if os.path.exists(TFLITE_MODEL_PATH):
    try:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite.python.interpreter import Interpreter

        print(f"Loading TFLite model from {TFLITE_MODEL_PATH}...")
        interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("TFLite model loaded successfully.")
    except Exception as e:
        print(f"Could not load TFLite model: {e}")

if interpreter is None and os.path.exists(H5_MODEL_PATH):
    try:
        from tensorflow.keras.models import load_model  # type: ignore
        print(f"Falling back to H5 model from {H5_MODEL_PATH}...")
        h5_model = load_model(H5_MODEL_PATH, compile=False)
        print("H5 model loaded successfully.")
    except Exception as e:
        print(f"Could not load H5 model: {e}")

# Helper: check allowed file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper: preprocess image using Pillow (no TensorFlow dependency needed)
def preprocess_image(img_input):
    """
    Preprocess image from either a filepath or a file-like object/PIL Image.
    Converts to 64x64 grayscale and normalizes pixel values to [0, 1].
    """
    if isinstance(img_input, Image.Image):
        img = img_input
    else:
        img = Image.open(img_input)

    img = img.convert('L').resize(INPUT_SHAPE, Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (64, 64, 1)
    img_array = np.expand_dims(img_array, axis=0)   # (1, 64, 64, 1)
    return img_array

def run_prediction(img_input):
    """Run model prediction on image and format result breakdown."""
    img_array = preprocess_image(img_input)

    if interpreter is not None:
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]
    elif h5_model is not None:
        preds = h5_model.predict(img_array, verbose=0)[0]
    else:
        raise RuntimeError("No emotion recognition model available.")

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

        try:
            with open(filepath, 'rb') as f:
                b64_data = base64.b64encode(f.read()).decode('utf-8')
            ext = safe_name.rsplit('.', 1)[1].lower() if '.' in safe_name else 'jpeg'
            mime = 'image/png' if ext == 'png' else 'image/jpeg'
            image_url = f"data:{mime};base64,{b64_data}"
        except Exception:
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
            pred_data = run_prediction(io.BytesIO(img_bytes))
            pred_data['image_url'] = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
            return jsonify({'success': True, 'data': pred_data})

        # 2. File upload in multipart
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                safe_name = f"{os.urandom(4).hex()}_{secure_filename(file.filename)}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
                file.save(filepath)

                pred_data = run_prediction(filepath)
                try:
                    with open(filepath, 'rb') as f:
                        b64_data = base64.b64encode(f.read()).decode('utf-8')
                    ext = safe_name.rsplit('.', 1)[1].lower() if '.' in safe_name else 'jpeg'
                    mime = 'image/png' if ext == 'png' else 'image/jpeg'
                    pred_data['image_url'] = f"data:{mime};base64,{b64_data}"
                except Exception:
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
