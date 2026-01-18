import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json

MODEL_PATH = "../model/cotton_leaf_model.keras"
TEST_DIR = "../dataset/test"

model = tf.keras.models.load_model(MODEL_PATH)

test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)

with open("../model/class_indices.json") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

print("\nConfusion Matrix:")
print(confusion_matrix(test_data.classes, y_pred))

print("\nClassification Report:")
print(classification_report(
    test_data.classes,
    y_pred,
    target_names=class_names
))
