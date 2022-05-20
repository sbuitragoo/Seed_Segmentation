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
        self.checkpoint_path = "weights"
        self.callback = tf.keras.callbacks.ModelCheckpoint(filepath = self.checkpoint_path,
                                                            verbose = 1,
                                                            save_weights_only = True)

    def loadTrainingDataset(self, trainImagePath, trainMaskPath, quantity=0.75, size=224, resize=True):
        
        self.trainImagePath = trainImagePath
        image_data = []

        if quantity:
            for img in sorted(os.listdir(self.trainImagePath)[:int(len(os.listdir(self.trainImagePath))*quantity)]):
                image = cv2.imread(os.path.join(self.trainImagePath, img))
                if resize:
                    image = cv2.resize(image, (size,size))
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
                    image = cv2.resize(image, (size,size))
                    mask_data.append(image)
                else:
                    mask_data.append(image)
            mask_data = np.array(mask_data)  

        return image_data, mask_data

    def loadValidationDataset(self, valImagePath, valMaskPath, quantity=0.75, size=224, resize=True):
        
        self.valImagePath = valImagePath
        image_data = []

        if quantity:
            for img in sorted(os.listdir(self.valImagePath)[int(len(os.listdir(self.valImagePath))*quantity):]):
                image = cv2.imread(os.path.join(self.valImagePath, img))
                if resize:
                    image = cv2.resize(image, (size,size))
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
                    image = cv2.resize(image, (size,size))
                    mask_data.append(image)
                else:
                    mask_data.append(image)
            mask_data = np.array(mask_data)  

        return image_data, mask_data

    def build(self, epochs = None, batchSize = None, model = 'mobilenet'):

        try: 
            os.mkdir(self.checkpoint_path)
        except:
            pass

        if epochs != None: 
            self.epochs = int(epochs) 
        else:
            self.epochs = 100
        
        if batchSize != None: 
            self.batchSize = int(batchSize)
        else: 
            self.batchSize = 64

        self.model = ModelToUse().get_model(model=model)

        self.model.save_weights(self.checkpoint_path.format(epoch=0))
        self.model.compile(optimizer='adam',
                            loss = "categorical_crossentropy")

        self.modelHistory = self.model.fit(
                                self.trainImages, self.trainTargets,
                                epochs = self.epochs,
                                validation_data = (self.valImages, self.valTargets),
                                batch_size = self.batchSize,
                                callbacks = [self.callback])
        
        return self.model, self.modelHistory


    def transform_targets(self, targets):
        targets_encoded = []
        for target in targets:
            target_encoded =  tf.one_hot(target, depth=len(np.unique(target).shape))
            targets_encoded.append(target_encoded)
        return np.array(targets_encoded)


    def startTraining(self, imagePath, maskPath, model, size, epochs, batchSize):

        self.trainImages, self.trainTargets = self.loadTrainingDataset(imagePath, maskPath, size=size)
        self.valImages, self.valTargets = self.loadValidationDataset(imagePath, maskPath, size=size)

        self.trainImages = self.trainImages/255.0
        self.valImages = self.valImages/255.0

        self.trainTargets = self.transform_targets(self.trainTargets)
        self.Valtargets = self.transform_targets(self.valTargets)

        model, history = self.build(model=model, epochs=epochs, batchSize=batchSize)
        return model, history

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    params = subparser.add_parser('params')
    params.add_argument('--i', type=str, required=True,
                        help="Path to the images")
    params.add_argument('--m', type=str, required=True,
                        help="Path to the masks")
    params.add_argument('--model', type=str, required=False,
                        help="Model to use")
    params.add_argument('--epochs', type=str, required=False,
                        help="Epochs")
    params.add_argument('--batch', type=str, required=False,
                        help="Batch Size")
    params.add_argument('--size', type=str, required=False,
                        help="Image Size")


    arguments = parser.parse_args()

    def retData(input1, input2):
        return input1, input2

    if arguments.command == "params":

        training = Training()

        model, history = training.startTraining(imagePath = arguments.i, maskPath = arguments.m, model = arguments.model, size = arguments.size, epochs = arguments.epochs, batchSize = arguments.batch)

        retData(model, history)
