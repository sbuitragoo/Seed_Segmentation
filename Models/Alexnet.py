import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from tensorflow import keras
import keras.layers as layers


from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras import layers,losses,models
import numpy as np
import matplotlib.pyplot as plt


def get_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), 
                            strides=(4, 4), activation="relu", 
                            input_shape=(227, 227, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides= (2, 2)))
    model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), 
                            strides=(1, 1), activation="relu", 
                            padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), 
                            strides=(1, 1), activation="relu", 
                            padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), 
                            strides=(1, 1), activation="relu", 
                            padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), 
                            strides=(1, 1), activation="relu", 
                            padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation="softmax"))
    
    return model

if __name__ == "__main__":

    pass

#Revisar Upsample para este modelo