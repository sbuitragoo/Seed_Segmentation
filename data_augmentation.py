import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.processing.image import ImageDataGenerator
from shutil import copy2, move
from absl import app, flags, logging
import argparse

class DataAugmentation():

    def __init__(self, imagePath, maskPath, numberOfImages):
        
        self.imagePath = imagePath
        self.maskDir = maskPath
        self.numberOfImages = numberOfImages
        self.rotationRange = 30
        self.widthSiftRange = 0.2
        self.zoomRange = 0.2
        self.horizontalFlip = True

    def makeDirectories(self):

        try:
            os.mkdir(augmentedPath)
            self.augmentedPath = augmentedDataPath
        except:
            pass

        try:
            os.mkdir(os.path.join(self.augmentedPath, "images"))
            self.augmentedImagesPath = os.path.join(self.augmentedPath, "images")
        except:
            pass

        try: 
            os.mkdir(os.path.join(self.augmentedPath, "targets"))
            self.augmentedTargetPath = os.path.join(self.augmentedPath, "targets")
        except:
            pass
        
    

    def augmentation(self):

        self.imageSize = cv2.imread(os.path.join(self.maskDir, os.listdir(self.maskDir)[0)).shape[:2]
        
        dataGenerator = ImageDataGenerator(
                rotation_range = self.rotationRange,
                width_shift_range = self.widthSiftRange,
                zoom_range = self.zoomRange,
                horizontal_flip = self.horizontalFlip)



        image_generator = datagen.flow_from_directory(directory=self.imagePath,target_size=self.imageSize,
                                                    save_to_dir=self.augmentedImagesPath, class_mode=None,
                                                    save_format='jpg', seed=42)

        masks_generator = datagen.flow_from_directory(directory=self.maskDir,target_size=self.imageSize,
                                                    save_to_dir=self.augmentedTargetPath, class_mode=None,
                                                    save_format='png', seed=42)
        
        n_iter = int(self.numberOfImages/32)

        if numberOfImages % 32:
            n_iter += 1
        
        print(f"Generating {n_iter*32} images")
        for i in range(n_iter):
            image_generator.next()
            masks_generator.next()

    def beginAugmentation(self):
        self.makeDirectories()
        self.augmentation()


if __name__ == "__main__":

    parser = argparse.AargumentParser()
    subparser = parser.add_subparsers(dest='command')
    auto = subparser.add_parser('auto')
    auto.add_argument('--imp', type=str, required=True,
                    help="Path where the images are located")
    auto.add_argument('--mp', type=str, required=True,
                    help="Path where the targets are located")
    auto.add_argument('--nim', type=str, required=True,
                    help="Number of images")

    arguments = parser.parse_args()

    if arguments.command == 'auto':
        dataAugmentation = DataAugmentation(arguments.imp, arguments.mp, arguments.nim)
        dataAugmentation.beginAugmentation()

    else:
        imp = input("Please type images path:")
        mp = input("Please type targets path")
        nim = input("Please type the number of images")
        dataAugmentation = DataAugmentation(imp, mp, nim)
        dataAugmentation.beginAugmentation()
