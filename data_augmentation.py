import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copy2, move

class DataAugmentation():

    def __init__(self, imagePath, maskPath, numberOfImages):
        
        self.imagePath = imagePath
        self.maskDir = maskPath
        self.numberOfImages = numberOfImages
        self.rotationRange = 30
        self.widthShiftRange = 0.2 
        self.heightShiftRange = 0.2
        self.zoomRange = 0.2
        self.horizontalFlip = True
        self.verticalFlip = True

    def makeDirectories(self):

        try:
            self.augmentedPath = "augmentedDataPath"
            os.mkdir(self.augmentedPath)
        except:
            pass

        try:
            self.augmentedImagesPath = os.path.join(self.augmentedPath, "images")
            os.mkdir(os.path.join(self.augmentedPath, "images"))
        except:
            pass

        try: 
            self.augmentedTargetPath = os.path.join(self.augmentedPath, "targets")
            os.mkdir(os.path.join(self.augmentedPath, "targets"))
        except:
            pass
        
    

    def augmentation(self):

        self.imageSize = cv2.imread(os.path.join(self.maskDir, os.listdir(self.maskDir)[0])).shape[:2]
        
        dataGenerator = ImageDataGenerator(
                featurewise_center = False,
                featurewise_std_normalization = False,
                rotation_range = self.rotationRange,
                width_shift_range = self.widthShiftRange,
                height_shift_range = self.heightShiftRange,
                zoom_range = self.zoomRange,
                horizontal_flip = self.horizontalFlip,
                vertical_flip = self.verticalFlip)



        image_generator = dataGenerator.flow_from_directory(directory=self.imagePath,
                                                    target_size=self.imageSize,
                                                    save_to_dir=self.augmentedImagesPath,
                                                    class_mode=None,
                                                    save_format='jpg', seed=42)

        masks_generator = dataGenerator.flow_from_directory(directory=self.maskDir,
                                                    target_size=self.imageSize,
                                                    save_to_dir=self.augmentedTargetPath,
                                                    class_mode=None,
                                                    save_format='png', seed=42)
        
        n_iter = int(int(self.numberOfImages)/32)

        if int(self.numberOfImages) % 32:
            n_iter += 1
        
        
        print(f"Copying Original Images")
        for img_path, mask_path in zip(os.listdir(self.imagePath), os.listdir(self.maskDir)):
            copy2(os.path.join(self.imagePath, img_path), self.augmentedImagesPath)
            copy2(os.path.join(self.maskDir, mask_path), self. augmentedTargetPath)

        print(f"Generating {n_iter*32} images")
        for i in range(n_iter):
            image_generator.next()
            masks_generator.next()

    def beginAugmentation(self):
        self.makeDirectories()
        self.augmentation()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
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
