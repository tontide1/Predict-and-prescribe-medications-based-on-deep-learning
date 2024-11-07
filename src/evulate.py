import joblib
import numpy as np
import pandas as pd
from pyvi import ViTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

model = load_model("C:\\Users\\tontide1\\my-data\\AI\\final\\data\\tokenizer_data\\trained_model.keras")
vocab = joblib.load("C:\\Users\\tontide1\\my-data\\AI\\final\\data\\tokenizer_data\\vocab.pkl")
label_encoder_disease = joblib.load("C:\\Users\\tontide1\\my-data\\AI\\final\\data\\tokenizer_data\\label_encoder_disease.pkl")
label_encoder_prescription = joblib.load(r"C:\Users\tontide1\my-data\AI\final\data\tokenizer_data\label_encoder_prescription.pkl")

def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path)
    data.fillna({'Van_de_benh_nhan': '', 'Benh': '', 'Don_thuoc': ''}, inplace=True)
    data.drop_duplicates(inplace=True)
    data['Tokenized_Problem'] = data['Van_de_benh_nhan'].apply(ViTokenizer.tokenize)
    return data

def build_vocab_and_encode_sequences(data, max_len=100):
    vocab = {}
    index = 1 
    sequences = []
    for text in data['Tokenized_Problem']:
        sequence = []
        for word in text.split():
            if word not in vocab:
                vocab[word] = index
                index += 1
            sequence.append(vocab[word])
        sequences.append(sequence)
    return vocab, pad_sequences(sequences, maxlen=max_len)

def encode_labels(data):
    label_encoder_disease = LabelEncoder()
    label_encoder_prescription = LabelEncoder()
    disease_labels = to_categorical(label_encoder_disease.fit_transform(data['Benh']))
    prescription_labels = to_categorical(label_encoder_prescription.fit_transform(data['Don_thuoc']))
    return label_encoder_disease, label_encoder_prescription, disease_labels, prescription_labels

file_path = 'C:\\Users\\tontide1\\my-data\\AI\\final\\data\\processed_medical_data_AI.xlsx'
data = load_and_preprocess_data(file_path)

# xây dựng từ điển và mã hoá nhãn
vocab, padded_sequences = build_vocab_and_encode_sequences(data)

label_encoder_disease, label_encoder_prescription, disease_labels, prescription_labels = encode_labels(data)

X_train, X_test, y_train_disease, y_test_disease, y_train_prescription, y_test_prescription = train_test_split(
    padded_sequences, disease_labels, prescription_labels, test_size=0.2, random_state=42
)

test_loss, test_disease_loss, test_prescription_loss, test_disease_accuracy, test_prescription_accuracy = model.evaluate(
    X_test,
    {'disease_output': y_test_disease, 'prescription_output': y_test_prescription},
    verbose=1
)


# print("Training and test sets created.")
# print("X_train shape:", X_train.shape)
# print("y_train_disease shape:", y_train_disease.shape)
# print("y_train_prescription shape:", y_train_prescription.shape)
# print("X_test shape:", X_test.shape)
# print("y_test_disease shape:", y_test_disease.shape)
# print("y_test_prescription shape:", y_test_prescription.shape)

print(f"Test Loss (Total): {test_loss}")
print(f"Test Disease Loss: {test_disease_loss}")
print(f"Test Prescription Loss: {test_prescription_loss}")
print(f"Test Disease Accuracy: {test_disease_accuracy}")
print(f"Test Prescription Accuracy: {test_prescription_accuracy}")