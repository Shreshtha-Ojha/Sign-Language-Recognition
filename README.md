# Sign Language Recognition System (AI/ML)

A comprehensive Sign Language Recognition system exploring multiple deep learning and computer vision approaches including CNNs, ANN, MediaPipe-based hand landmark detection, and continuous gesture recognition.

---

## Project Motivation

Communication barriers faced by the hearing- and speech-impaired community remain a significant challenge.  
This project aims to reduce that gap by leveraging computer vision and machine learning to recognize sign language gestures and convert them into meaningful outputs.

---

## Repository Overview

This repository is designed as a combined research and implementation project.  
Multiple approaches were implemented, evaluated, and compared to understand their effectiveness in real-time sign language recognition.

---

## Implemented Approaches

- CNN-based static sign language recognition  
- ANN-based gesture classification  
- MediaPipe hand landmark–based recognition  
- Continuous sign language recognition pipeline  
- Game-based learning modules for improved user engagement  

---
```
Sign-Language-Recognition/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── cnn_model.h5
│   └── ann_model.h5
│
├── src/
│   ├── mediapipe_utils.py
│   ├── cnn.py
│   ├── ann.py
│   ├── train.py
│   └── realtime.py
│
├── games/
│   ├── asl_game/
│   │   └── game.py
│   └── ninja_game/
│       └── game.py
│
└── docs/
    └── presentation.pptx

```

---

## Technical Approach

### Hand Detection
- MediaPipe Hands
- 21 hand landmarks extracted per frame

### Models Used
- Convolutional Neural Networks (CNN)
- Artificial Neural Networks (ANN)

### Learning Pipeline
1. Data collection and preprocessing  
2. Feature extraction (images and landmarks)  
3. Model training and validation  
4. Real-time inference using webcam input  

---

## Tech Stack

- Programming Language: Python  
- Libraries and Frameworks:
  - OpenCV
  - MediaPipe
  - TensorFlow / Keras
  - NumPy
  - Pandas  
- Domains:
  - Computer Vision
  - Deep Learning
  - Human-Computer Interaction  

---

## Interactive Components

To improve usability and engagement, the project includes:
- ASL-based learning game
- Ninja game demonstrating gesture-based control

These components show how sign language recognition can be integrated into educational and interactive applications.

---

## How to Run the Project

### Prerequisites

- Python 3.8 or higher
- Git
- Webcam (for real-time sign recognition)

Check Python version:
```bash
python --version

Clone the Repository
git clone https://github.com/<your-username>/Sign-Language-Recognition.git
cd Sign-Language-Recognition

Create and Activate Virtual Environment
python -m venv venv

venv\Scripts\activate

Install Dependencies
pip install -r requirements.txt
pip install opencv-python mediapipe tensorflow numpy pandas

Dataset Setup
Preprocess the dataset:
python src/data/preprocessing.py

Train Models (Optional)
python src/training/train_cnn.py
python src/training/train_ann.py

Run Real-Time Sign Language Recognition
python src/inference/realtime.py

Run Interactive Applications
Run ASL learning game:
python apps/asl_game/game.py

Run Ninja game:
python apps/ninja_game/game.py



