from Model import ModelToUse 
import tensorflow as tf
import matplotlib.pyplot as plt


class Training():

    def __init__(self):
        
        self.imageSize = 224
        self.classes = 3
        self.callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
                                                            verbose = 1,
                                                            save_weights_only = True)
        self.model = ModelToUse().get_model()
        self.epochs = 100
        self.batchSize = 32


    def loadTrainingDataset(self, trainPath):
        
        #Por definir
        #self.trainDataset = ""
        pass

    def loadValidationDataset(self, valPath):
        
        #Por definir
        #self.validationDataset = validationData
        pass

    def build(self):
        self.model.save_weights(checkpoint_path.format(epoch=0))
        self.model.compile(optimizer='adam', metrics=['accuracy'],
                            loss = tf.keras.losses.SparseCategoricalCrossentropy())

        self.modelHistory = model.fit(
                                self.trainDataset,
                                epochs = self.epochs,
                                validation_data = self.validationDataset,
                                batch_size = self.batchSize,
                                callbacks = [self.callback])
        
        return self.model, self.modelHistory

    def startTraining(self, trainPath, valPath):
        self.loadTrainingDataset(trainPath)
        self.loadValidationDadaset(valPath)
        model, history = self.build()
        return model, history

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    params = subparser.add_parser('params')
    params.add_argument('--t', type=str, required=True,
                        help="Path to the training images")
    params.add_argument('--v', type=str, required=True,
                        help="Path to the validation images")

    arguments = parser.parse_args()

    if arguments == "params":

        training = Training()

        model, history = training.startTraining(arguments.t, arguments.q)
