import pandas as pd
import numpy as np
from pyvi import ViTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from collections import defaultdict
from tensorflow.keras.optimizers import Adam

def load_and_preprocess_data(file_path):
    """Load and preprocess data from an Excel file."""
    data = pd.read_excel(file_path)
    data.fillna({'Van_de_benh_nhan': '', 'Benh': '', 'Don_thuoc': ''}, inplace=True)
    data.drop_duplicates(inplace=True)
    data['Tokenized_Problem'] = data['Van_de_benh_nhan'].apply(ViTokenizer.tokenize)
    return data

def build_vocab_and_encode_sequences(data, max_len=100):
    """Build vocabulary and encode sequences."""
    vocab = defaultdict(lambda: len(vocab) + 1)
    sequences = []
    for text in data['Tokenized_Problem']:
        sequence = [vocab[word] for word in text.split()]
        sequences.append(sequence)
    return vocab, pad_sequences(sequences, maxlen=max_len)

def encode_labels(data):
    """Encode labels for disease and prescription."""
    label_encoder_disease = LabelEncoder()
    label_encoder_prescription = LabelEncoder()
    disease_labels = to_categorical(label_encoder_disease.fit_transform(data['Benh']))
    prescription_labels = to_categorical(label_encoder_prescription.fit_transform(data['Don_thuoc']))
    return label_encoder_disease, label_encoder_prescription, disease_labels, prescription_labels

def build_model(vocab_size, num_disease_classes, num_prescription_classes, input_length=100):
    """Build and compile the model."""
    input_layer = Input(shape=(input_length,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=64)(input_layer)
    lstm_layer = LSTM(64)(embedding_layer)

    disease_output = Dense(num_disease_classes, activation='softmax', name='disease_output')(lstm_layer)
    prescription_output = Dense(num_prescription_classes, activation='softmax', name='prescription_output')(lstm_layer)

    model = Model(inputs=input_layer, outputs=[disease_output, prescription_output])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'disease_output': 'categorical_crossentropy', 'prescription_output': 'categorical_crossentropy'},
        metrics={'disease_output': 'accuracy', 'prescription_output': 'accuracy'}
    )
    return model

def train_model(model, padded_sequences, disease_labels, prescription_labels, epochs=100, batch_size=64):
    """Train the model."""
    model.fit(
        padded_sequences,
        {'disease_output': disease_labels, 'prescription_output': prescription_labels},
        epochs=epochs,
        batch_size=batch_size,
        # validation_split=validation_split
    )

def make_prediction(input_text, model, vocab, label_encoder_disease, label_encoder_prescription, max_len=100):
    """Make predictions for a given input text."""
    tokenized_text = ViTokenizer.tokenize(input_text)
    sequence = [vocab[word] for word in tokenized_text.split() if word in vocab]
    padded_sequence = pad_sequences([sequence], maxlen=max_len)
    prediction = model.predict(padded_sequence)

    disease_prediction = label_encoder_disease.inverse_transform(np.argmax(prediction[0], axis=1))
    prescription_prediction = label_encoder_prescription.inverse_transform(np.argmax(prediction[1], axis=1))

    return disease_prediction[0], prescription_prediction[0]

def main():
    """Main function for interaction."""
    file_path = 'C:\\Users\\tontide1\\my-data\\AI_PROJECT_NEW\\data\\processed_medical_data_AI.xlsx'
    data = load_and_preprocess_data(file_path)

    vocab, padded_sequences = build_vocab_and_encode_sequences(data)
    label_encoder_disease, label_encoder_prescription, disease_labels, prescription_labels = encode_labels(data)

    model = build_model(len(vocab) + 1, len(label_encoder_disease.classes_), len(label_encoder_prescription.classes_))
    train_model(model, padded_sequences, disease_labels, prescription_labels)

    while True:
        input_text = input("Nhập mô tả bệnh (hoặc gõ 'exit' để thoát): ")
        if input_text.lower() == 'exit':
            break
        disease, prescription = make_prediction(input_text, model, vocab, label_encoder_disease, label_encoder_prescription)
        print(f"Vấn đề của bệnh nhân: {input_text}")
        print(f"Bệnh dự đoán: {disease}")
        print(f"Đơn thuốc dự đoán: {prescription}")

if __name__ == "__main__":
    main()