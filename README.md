# Medical Condition Prediction and Prescription System

## Overview
This project implements a deep learning system that predicts medical conditions and recommends prescriptions based on patient symptoms described in Vietnamese. The system uses LSTM (Long Short-Term Memory) neural networks to process natural language input and provide medical predictions.

## Features
- Natural Language Processing for Vietnamese medical symptoms
- Dual prediction system:
  - Disease/condition prediction
  - Prescription recommendation
- Multiple optimizer configurations for model training
- Interactive interface for real-time predictions

## Technical Stack
- Python 3.10
- TensorFlow
- PyVi (Vietnamese NLP)
- Streamlit (Web Interface)
- Pandas & NumPy
- Scikit-learn

## Model Architecture
- Input Layer: Text embedding layer
- Hidden Layer: LSTM with 64 units
- Output Layers: 
  - Disease prediction (889 classes)
  - Prescription prediction (2963 classes)

## Performance
Based on different optimizer configurations:
- Best Disease Prediction Accuracy: 83.00%
- Best Prescription Accuracy: 81.89%
- Optimal Learning Rate: 0.01

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Predict-and-prescribe-medications-based-on-deep-learning.git
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run src/app.py
```

#### Note: This project is for educational purposes only. So i don't have the medical data. And this is not a perfect project. If you want to use this project, you need to collect your own medical data. And modify the code.