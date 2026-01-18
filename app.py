from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import io
from PIL import Image

app = Flask(__name__)

# Load model and class indices
MODEL_PATH = "model/cotton_leaf_model.keras"
CLASS_INDEX_PATH = "model/class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)
CLASS_NAMES = {v: k for k, v in class_indices.items()}

def predict_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0]
    class_id = int(np.argmax(pred))
    confidence = float(np.max(pred) * 100)
    return CLASS_NAMES[class_id], confidence

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    label, confidence = predict_image(file)
    return jsonify({"label": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)
