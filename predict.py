from DataManagementScripts.dataset import parse_labelfile
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
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
        
    def Predict(self, imagePath, labelsPath, outPath, size=224):

        self.imageSize = size
        self.classes = 3
        
        try:
            os.mkdir(outPath)
        except:
            pass

        labels = parse_labelfile(labelsPath)

        img = plt.imread(imagePath)/255
        X = tf.convert_to_tensor(img)
        X = tf.image.resize(X, (self.imageSize, self.imageSize))
        X = tf.expand_dims(X, 0)

        model = tf.keras.models.load_model("model.h5")

        Y = model.predict(X)
        Y = tf.argmax(Y, axis=-1)
        Y = self.Categorical2Mask(Y[0], labels)
        Y = cv2.resize(Y, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.DisplayData([img, Y])

        if outPath != None:
            Y = cv2.cvtColor(Y, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(outPath+"prediction"), Y)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    params = subparser.add_parser('params')
    params.add_argument('--i', type=str, required=True,
                        help="Path to the images")
    params.add_argument('--l', type=str, required=True,
                        help="Path to the labels")
    params.add_argument('--out', type=int, required=False,
                        help="Path to save predictions")                                     
    params.add_argument('--size', type=int, required=False,
                        help="Image Size")

    arguments = parser.parse_args()

    if arguments.command == "params":

        prediction = Prediction()
        prediction.Predict(imagePath=arguments.i, labelsPath=arguments.l, outPath=arguments.out, size=arguments.size)
