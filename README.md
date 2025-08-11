# Hand Sign Detection Using MediaPipe, YOLOv8n, and Random Forest Classifier

## 📌 Overview

This project implements a **real-time hand sign detection system** that captures webcam input, extracts hand landmarks, and predicts corresponding alphabet signs. The goal is to recognize multiple hand gestures and construct sentences from the detected letters.

The system uses:

* **YOLOv8n** for robust hand landmark detection
* **MediaPipe** for precise 21-point hand keypoint extraction
* **Random Forest Classifier** for classification of signs into alphabet letters
* **OpenCV** for real-time video capture and visualization

---

## 🚀 Features

* Real-time hand gesture recognition
* Converts detected signs into alphabet letters
* Sentence construction from sequential predictions
* High accuracy with optimized Random Forest model
* Custom dataset creation and training pipeline

---

## 🛠️ Tech Stack

* **Python 3.12+**
* **YOLOv8 Pose Detection**
* **MediaPipe Hands**
* **Scikit-learn**
* **OpenCV**
* **NumPy & Pandas**
* **Pickle** (for model storage)

---

## 📂 Project Structure

```
HandSignDetection/
│
├── dataset/                     # Collected landmark datasets
├── models/                      # Trained Random Forest model
├── scripts/
│   ├── data_collection.py       # Capture hand landmarks for dataset
│   ├── train_classifier.py      # Train Random Forest model
│   ├── inference_classifier.py  # Real-time gesture detection
│
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
└── LICENSE                      # License file
```

---

## 🔧 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/HandSignDetection.git
cd HandSignDetection
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Creation

Run the dataset collection script:

```bash
python scripts/data_collection.py
```

* Press `q` to quit after capturing enough samples.
* The collected landmarks will be saved in `.csv` format for training.

---

## 🏋️ Model Training

Train the **Random Forest Classifier**:

```bash
python scripts/train_classifier.py
```

* The trained model will be saved in `models/random_forest.pkl`.

---

## 🎯 Real-Time Inference

Run the real-time gesture detection:

```bash
python scripts/inference_classifier.py
```

* Webcam will open and start detecting signs.
* The predicted letter will appear on the screen.

---

## 📈 Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix


---
##📷 System Workflow

![Image](https://github.com/user-attachments/assets/990c200e-19cf-4edf-9554-c9646be844a6)

## ✨ Future Improvements

* Support for two-hand gesture detection
* Integration with Speech-to-Text
* Mobile app version
* Larger dataset for improved accuracy

