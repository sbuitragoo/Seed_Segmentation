import numpy as np 
import cv2
import os

def load_data(path, size=224, quantity=None, resize=False):
    """Loads a set of images 

    Args:
        path (str): path to the images
        quantity (int, optional): the quantity of images to load. Defaults to None.
        resize (bool, optional): if it is wanted to resize the images. Defaults to False.

    Returns:
        _type_: _description_
    """

    data = []

    if quantity:
        for img in sorted(os.listdir(path)[:quantity]):
            image = cv2.imread(os.path.join(path, img))
            if resize:
                image = cv2.resize(image, (size,size))
                data.append(image)
            else:
                data.append(image)
        data = np.array(data)
        return data
    
    else:
        for img in sorted(os.listdir(path)):
            image = cv2.imread(os.path.join(path, img))
            if resize:
                image = cv2.resize(image, (224,224))
                data.append(image)
            else:
                data.append(image)
        data = np.array(data)
        return data
    
        