from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras import regularizers
import keras_metrics

def create_cnn_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu', 
                     input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), 
                           keras_metrics.f1_score(), keras_metrics.true_positive(), 
                           keras_metrics.true_negative(), keras_metrics.false_positive(),
                           keras_metrics.false_negative()])
    return model