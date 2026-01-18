import tensorflow as tf
import cv2
import numpy as np
import json
import sys

MODEL_PATH = "../model/cotton_leaf_model.keras"
CLASS_INDEX_PATH = "../model/class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)

CLASS_NAMES = {v: k for k, v in class_indices.items()}

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("❌ Image not found")
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    return CLASS_NAMES[class_id], confidence

if __name__ == "__main__":
    image_path = sys.argv[1]
    label, conf = predict_image(image_path)
    print(f"🌿 Prediction: {label}")
    print(f"📊 Confidence: {conf*100:.2f}%")
