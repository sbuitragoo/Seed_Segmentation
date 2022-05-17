import tensorflow as tf 

from Models.MobileNetV2 import get_model as getMobilenet
from Models.Unet import get_model as getUnet
from Models.InceptionResnetV2_2 import get_model as getIRv2
from Models.UnetMobilenet import get_model as getUM

class ModelToUse():

    def __init__(self):

        self.models = {'mobilenetv2': getMobilenet,
                        'unet': getUnet,
                        'inceptionresnetv2': getIRv2,
                        'unetmobilenet': getUM}

    def print_available_models(self):
        print('Availaible models: ')
        for model in self.models.keys():
            print(f'\t{model}')


    def get_model(self, model='mobilenetv2', **kwargs):
        model = model.lower()
        try: 
            model_keras = self.models[model](**kwargs)
            return model_keras
        except KeyError: 
            print(f'Model {model} is not avalaible')
            print(f"Posible models: {', '.join(self.models.keys())}")
            exit()

if __name__ == '__main__':
    pass
    # Model = ModelToUse()
    # Model.print_available_models()
    # model = Model.get_model()
    # model().summary()
    # tf.keras.utils.plot_model(model(),to_file='./model.png',show_shapes=False,show_layer_names=False)