import tensorflow as tf 

from Models.MobileNetV2 import MobileNet
from Models.Unet import Unet
from Models.InceptionResnetV2_2 import InceptionResnet

class ModelToUse():

    def __init__(self):

        self.models = {'mobilenetv2': MobileNet.get_model,
                        'unet':Unet.get_model,
                        'inceptionresnetv2':InceptionResnet.get_model}

    def print_available_models(self):
        print('Availaible models: ')
        for model in self.models.keys():
            print(f'\t{model}')


    def get_model(self,model='mobilenetv2'):
        model = model.lower()
        try: 
            model_keras = self.models[model]
            return model_keras
        except KeyError: 
            print(f'Model {model} is not avalaible')
            print(f"Posible models: {', '.join(self.models.keys())}")
            exit()

if __name__ == '__main__':
    Model = ModelToUse()
    Model.print_available_models()
    model = Model.get_model()
    model().summary()
    tf.keras.utils.plot_model(model(),to_file='./model.png',show_shapes=False,show_layer_names=False)