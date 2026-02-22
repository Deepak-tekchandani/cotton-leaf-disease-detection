# 🌿 Cotton Leaf Disease Detection — AI Web Application

An **AI-powered Deep Learning web application** that detects cotton plant leaf diseases using **Computer Vision** and **Convolutional Neural Networks (CNN)**.  
The system allows users to upload or capture a cotton leaf image and instantly receive disease predictions along with a confidence score through an interactive web interface.

---

## 🚀 Key Highlights

- 🤖 Deep Learning–based disease classification (CNN + Transfer Learning)
- 🌐 Flask-based AI web application
- 📱 Mobile-friendly interface with live camera capture
- 📊 Prediction confidence visualization
- ⚡ Real-time image prediction
- 🧹 Clear/reset image functionality
- 🎨 Responsive Bootstrap-based UI

---

## 🦠 Detected Diseases

The model is trained to classify the following cotton leaf conditions:

- Alternaria Leaf Spot (Fungal)
- Anthracnose (Bacterial)
- Bacterial Blight
- Healthy Leaf
- Leaf Curl Virus
- Thrips Insect Damage
- Whiteflies Insect Damage

---

## 🛠 Tech Stack

### 🧠 AI / Machine Learning
- Python
- TensorFlow
- Keras
- Convolutional Neural Networks (CNN)
- Transfer Learning (InceptionV3)

### ⚙️ Backend
- Flask (Python Web Framework)

### 🎨 Frontend
- HTML
- CSS
- Bootstrap
- JavaScript

### 🧰 Tools & Libraries
- OpenCV
- NumPy
- Git & GitHub

---

## 🧪 Model Training

The deep learning model was trained using transfer learning techniques:

- `.keras` — Modern recommended model format
- `.h5` — Legacy compatibility format

---

## 🧠 How It Works

Upload or capture a cotton leaf image → Model processes image → CNN predicts disease → Result displayed with confidence percentage.

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
