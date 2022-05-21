from DataManagementScripts.dataset import parse_labelfile
from tf.keras.model import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

class Prediction():

    def __init__(self):
        pass



    def DisplayData(self, dataToDisplay):
        
        plt.figure(figsize=(15, 15))

        title = ['Input Image', 'Predicted Mask']

        for i in range(len(dataToDisplay)):
            plt.subplot(1, len(dataToDisplay), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(dataToDisplay[i]))
            plt.axis('off')
        plt.show()

    def Categorical2Mask(self, X, labels):
        
        Y = np.zeros(X.shape[0:2] + [3], dtype="uint8")
        for i, key in enumerate(labels):
            Y[...,0] = np.where(X==i, labels[key][0], Y[...,0])
            Y[...,1] = np.where(X==i, labels[key][1], Y[...,1])
            Y[...,2] = np.where(X==i, labels[key][2], Y[...,2])
        return Y
        
    def Predict(self, imagePath, labelsPath, outPath, weightsPath, size=224):

        self.imageSize = size
        self.classes = 3

        img_path = imagePath
        out_path = outPath
        weights_path = weightsPath
        labels_Path = labelsPath
        

        labels = parse_labelfile(labels_Path)

        img = plt.imread(img_path)/255
        X = tf.convert_to_tensor(img)
        X = tf.image.resize(X, (self.imageSize, self.imageSize))
        X = tf.expand_dims(X, 0)

        model = load_model("model.h5")

        Y = model.predict(X)
        Y = tf.argmax(Y, axis=-1)
        Y = self.Categorical2Mask(Y[0], labels)
        Y = cv2.resize(Y, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.DisplayData([img, Y])

        if out_path != None:
            Y = cv2.cvtColor(Y, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(out_path+"prediction"), Y)
