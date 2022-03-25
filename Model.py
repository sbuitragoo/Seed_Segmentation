import tensorflow as tf 

from Models.MobileNetV2 import get_model as get_MobileNetV2
from Models.Unet import get_model as get_UNET
from Models.InceptionResnetV2_2 import get_model as get_InceptionResnetV2

MODELS = {'mobilenetv2': get_MobileNetV2,
          'unet':get_UNET,
          'inceptionresnetv2':get_InceptionResnetV2}

def print_available_models():
    print('Availaible models: ')
    for model in MODELS.keys():
        print(f'\t{model}')


def get_model(model='mobilenetv2', **kwargs):
    model = model.lower()
    try: 
        model_keras = MODELS[model](**kwargs)
        return model_keras
    except KeyError: 
        print(f'Model {model} is not avalaible')
        print(f"Posible models: {', '.join(MODELS.keys())}")
        exit()

if __name__ == '__main__':
    print_available_models()
    model = get_model(output_channels=2)
    model.summary()
    tf.keras.utils.plot_model(model,to_file='./model.png',show_shapes=False,show_layer_names=False)