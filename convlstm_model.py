from keras.models import Sequential
from keras.layers import ConvLSTM2D, Dropout, Flatten, Dense
import keras_metrics
from keras_self_attention import SeqSelfAttention

def create_convlstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(ConvLSTM2D(filters=32, kernel_size=(2, 2), strides=(1, 1), activation='relu', 
                         input_shape=input_shape, return_sequences=True))
    model.add(ConvLSTM2D(filters=32, kernel_size=(2, 2), strides=(1, 1), activation='relu', 
                         return_sequences=True))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), 
                           keras_metrics.f1_score(), keras_metrics.true_positive(), 
                           keras_metrics.true_negative(), keras_metrics.false_positive(),
                           keras_metrics.false_negative()])
    return model