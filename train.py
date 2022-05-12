from Model import ModelToUse 
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import argparse
import numpy as np

class Training():

    def __init__(self):
        
        self.imageSize = 224
        self.classes = 3
        self.callback = tf.keras.callbacks.ModelCheckpoint(filepath = self.checkpoint_path,
                                                            verbose = 1,
                                                            save_weights_only = True)
        self.model = ModelToUse().get_model()
        self.epochs = 100
        self.batchSize = 32
        self.checkpoint_path = "weights"

    def loadTrainingDataset(self, trainImagePath, trainMaskPath, quantity=0.75,resize=True):
        
        self.trainImagePath = trainImagePath
        image_data = []

        if quantity:
            for img in sorted(os.listdir(self.trainImagePath)[:int(len(os.listdir(self.trainImagePath))*quantity)]):
                image = cv2.imread(os.path.join(self.trainImagePath, img))
                if resize:
                    image = cv2.resize(image, (224,224))
                    image_data.append(image)
                else:
                    image_data.append(image)
            image_data = np.array(image_data)

        self.trainMaskPath = trainMaskPath
        mask_data = []

        if quantity:
            for img in sorted(os.listdir(self.trainMaskPath)[:int(len(os.listdir(self.trainMaskPath))*quantity)]):
                image = cv2.imread(os.path.join(self.trainMaskPath, img))
                if resize:
                    image = cv2.resize(image, (224,224))
                    mask_data.append(image)
                else:
                    mask_data.append(image)
            mask_data = np.array(mask_data)  

        return image_data, mask_data

    def loadValidationDataset(self, valImagePath, valMaskPath, quantity=0.75,resize=True):
        
        self.valImagePath = valImagePath
        image_data = []

        if quantity:
            for img in sorted(os.listdir(self.valImagePath)[int(len(os.listdir(self.valImagePath))*quantity):]):
                image = cv2.imread(os.path.join(self.valImagePath, img))
                if resize:
                    image = cv2.resize(image, (224,224))
                    image_data.append(image)
                else:
                    image_data.append(image)
            image_data = np.array(image_data)

        self.valMaskPath = valMaskPath
        mask_data = []

        if quantity:
            for img in sorted(os.listdir(self.valMaskPath)[int(len(os.listdir(self.valMaskPath))*quantity):]):
                image = cv2.imread(os.path.join(self.valMaskPath, img))
                if resize:
                    image = cv2.resize(image, (224,224))
                    mask_data.append(image)
                else:
                    mask_data.append(image)
            mask_data = np.array(mask_data)  

        return image_data, mask_data

    def build(self):

        try: 
            os.mkdir(self.checkpoint_path)
        except:
            pass

        self.model.save_weights(self.checkpoint_path.format(epoch=0))
        self.model.compile(optimizer='adam', metrics=['accuracy'],
                            loss = tf.keras.losses.SparseCategoricalCrossentropy())

        self.modelHistory = model.fit(
                                self.trainDataset,
                                epochs = self.epochs,
                                validation_data = self.validationDataset,
                                batch_size = self.batchSize,
                                callbacks = [self.callback])
        
        return self.model, self.modelHistory

    def startTraining(self, imagePath, maskPath):

        self.loadTrainingDataset(imagePath, maskPath)
        self.loadValidationDataset(imagePath, maskPath)
        model, history = self.build()
        return model, history

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    params = subparser.add_parser('params')
    params.add_argument('--i', type=str, required=True,
                        help="Path to the images")
    params.add_argument('--m', type=str, required=True,
                        help="Path to the masks")


    arguments = parser.parse_args()

    if arguments == "params":

        training = Training()

        model, history = training.startTraining(arguments.i, arguments.m)
