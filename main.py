import numpy as np
import pandas as pd
import time
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_data
from preprocessing import randomize, standardize_data
from utils import plot_confusion_matrix
from cnn_lstm_model import create_cnn_lstm_model
from convlstm_model import create_convlstm_model
from bilstm_model import create_bilstm_model
from cnn_model import create_cnn_model

# Load datasets
healthy_eeg = load_data('path_to_healthy_eeg.csv')
long_covid_eeg = load_data('path_to_long_covid_eeg.csv')

# Preprocess datasets
# Add your preprocessing steps here

# Combine datasets
# Add your dataset combining steps here

# Standardize data
train_data, test_data = standardize_data(train_data, test_data)

# Convert labels to categorical
num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define input shape
input_shape = (train_data.shape[1], train_data.shape[2], 1)

# Train and evaluate models
models = {
    "CNN_LSTM": create_cnn_lstm_model(input_shape, num_classes),
    "CONVLSTM": create_convlstm_model(input_shape, num_classes),
    "BiLSTM": create_bilstm_model((train_data.shape[1], train_data.shape[2]), num_classes),
    "CNN": create_cnn_model(input_shape, num_classes)
}

histories = {}
test_scores = {}

for model_name, model in models.items():
    history = model.fit(train_data, y_train, batch_size=32, validation_split=0.2, epochs=20, verbose=2)
    histories[model_name] = history
    test_scores[model_name] = model.evaluate(test_data, y_test, verbose=0)
    print(f"{model_name} - Test accuracy: {test_scores[model_name][1]}")

    y_pred = model.predict(test_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    print(f"{model_name} Classification Report:")
    print(classification_report(y_test_classes, y_pred_classes))

    cm = confusion_matrix(y_test_classes, y_pred_classes)
    plot_confusion_matrix(cm, classes=['Healthy', 'Disease'], title=f'{model_name} Confusion Matrix')

