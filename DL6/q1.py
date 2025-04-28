import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train, X_test = X_train / 255.0, X_test / 255.0
    
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    return X_train, X_test, y_train, y_test

def build_model(input_shape, num_classes=10):
    model = Sequential()
    
    model.add(layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu", input_shape=input_shape))
    model.add(layers.MaxPool2D(2))
    
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, activation="softmax"))
    
    return model

def main():
    X_train, X_test, y_train, y_test = load_data()
    
    model = build_model(X_train[0].shape)
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    checkpoint_cb = ModelCheckpoint(
    filepath='./DL6/q2/checkpoints2/cp-{epoch:04d}.keras',
    save_best_only=False,  # 가장 좋은 성능의 모델만 저장할 경우 True
    save_weights_only=False,  # True이면 가중치만 저장
    monitor='val_loss',
    verbose=1)
    
    model.fit(X_train, y_train, epochs=5, batch_size=64, shuffle=True, verbose=2, callbacks=[checkpoint_cb])
    
    # TODO: H5 형식으로 모델 저장
    model.save('./DL6/h5_model.h5')
    
    # TODO: SavedModel 형식으로 모델 저장
    model.save('./DL6/saved_model.keras')

if __name__ == "__main__":
    main()
