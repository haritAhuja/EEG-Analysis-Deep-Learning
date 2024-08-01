from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
import keras_metrics
from keras_self_attention import SeqSelfAttention

def create_bilstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(LSTM(20, return_sequences=True)))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Bidirectional(LSTM(20, return_sequences=False)))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), 
                           keras_metrics.f1_score(), keras_metrics.true_positive(), 
                           keras_metrics.true_negative(), keras_metrics.false_positive(),
                           keras_metrics.false_negative()])
    return model