# Cotton Leaf Disease Detection

A web application that uses **AI (TensorFlow + Keras)** to detect diseases in cotton leaves. Users can **upload images** or use a **live camera capture** to get predictions with confidence scores. Built with **Flask** and a mobile-friendly **Bootstrap UI**.

---

## Features

- Detects 7 types of leaf conditions:
  1. Alternaria Leaf Spot (Fungal)
  2. Anthracnose (Bacteria)
  3. Bacterial Blight
  4. Healthy
  5. Leaf Curl Virus
  6. Thrips (Insect)
  7. Whiteflies (Insect)
- Upload image or capture via live camera
- Shows prediction with confidence bar
- Mobile-friendly UI
- Option to clear uploaded image

---

## Installation

1. **Clone the repository**  
```bash
git clone https://github.com/your-username/cotton-leaf-disease-detection.git
cd cotton-leaf-disease-detection

Create virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies
```bash
pip install -r requirements.txt


Download or place your trained models

model/cotton_leaf_model.keras

model/cotton_leaf_model.h5

model/class_indices.json

Usage

Run the Flask app:

python app.py


Open your browser and go to:

http://127.0.0.1:5000


Upload a cotton leaf image or use the live camera to get predictions.

Confidence bar shows the prediction certainty.

Click "Clear" to remove the uploaded image.

Folder Structure
cotton-leaf-disease-detection/
│
├─ app.py                 # Flask application
├─ evaluate.py            # Evaluate model on test data
├─ predict.py             # Predict single images
├─ requirements.txt       # Python dependencies
├─ model/                 # Trained models and class indices
├─ templates/             # HTML templates
└─ static/                # CSS, JS, images

Dependencies

Python 3.10+
TensorFlow
Keras
OpenCV
Flask
NumPy
scikit-learn
Matplotlib (optional for plots)
Pillow
