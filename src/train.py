import numpy as np
import pandas as pd
from pyvi import ViTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import joblib

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



def build_model(vocab_size, num_disease_classes, num_prescription_classes, input_length=100):
    input_layer = Input(shape=(input_length,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=64)(input_layer)
    lstm_layer = LSTM(64)(embedding_layer)

    disease_output = Dense(num_disease_classes, activation='softmax', name='disease_output')(lstm_layer)
    prescription_output = Dense(num_prescription_classes, activation='softmax', name='prescription_output')(lstm_layer)

    model = Model(inputs=input_layer, outputs=[disease_output, prescription_output])
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss={'disease_output': 'categorical_crossentropy', 'prescription_output': 'categorical_crossentropy'},
        metrics={'disease_output': 'accuracy', 'prescription_output': 'accuracy'}
    )
    return model

# Train and save the model
file_path = 'C:\\Users\\tontide1\\my-data\\AI\\AI_PROJECT_NEW (1)\\AI_PROJECT_NEW\\data\\processed_medical_data_AI.xlsx'
data = load_and_preprocess_data(file_path)

vocab, padded_sequences = build_vocab_and_encode_sequences(data)
label_encoder_disease, label_encoder_prescription, disease_labels, prescription_labels = encode_labels(data)

model = build_model(len(vocab) + 1, len(label_encoder_disease.classes_), len(label_encoder_prescription.classes_))
# model.fit(padded_sequences, {'disease_output': disease_labels, 'prescription_output': prescription_labels}, epochs=50, batch_size=32)
history = model.fit(
    padded_sequences,
    {'disease_output': disease_labels, 'prescription_output': prescription_labels},
    epochs=60,
    batch_size=64
)

avg_disease_accuracy = sum(history.history['disease_output_accuracy']) / len(history.history['disease_output_accuracy'])
avg_prescription_accuracy = sum(history.history['prescription_output_accuracy']) / len(history.history['prescription_output_accuracy'])

print('avg_disease_accuracy', avg_disease_accuracy)
print('avg_prescription_accuracy' ,avg_prescription_accuracy)
print("Model and encoders saved successfully.")

# lưu lại model và các pkl
model.save("C:\\Users\\tontide1\\my-data\\AI\\final\\data\\tokenizer_data\\trained_model.keras")  
joblib.dump(vocab, "C:\\Users\\tontide1\\my-data\\AI\\final\\data\\tokenizer_data\\vocab.pkl")    
joblib.dump(label_encoder_disease, "C:\\Users\\tontide1\\my-data\\AI\\final\\data\\tokenizer_data\\label_encoder_disease.pkl")
joblib.dump(label_encoder_prescription, "C:\\Users\\tontide1\\my-data\\AI\\final\\data\\tokenizer_data\\label_encoder_prescription.pkl")