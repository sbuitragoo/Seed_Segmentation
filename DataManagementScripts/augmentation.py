from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from shutil import copy2
import tensorflow as tf
import numpy as np
import argparse
import os


class DataAugmentation():

    def __init__(self):
        self.datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=180,
            width_shift_range=0.5,
            height_shift_range=0.5,
            zoom_range = 0.7,
            shear_range = 45,
            vertical_flip = True,
            horizontal_flip = True
            )

    def DataGenerator(self, imageDir, maskDir, labelsPath, numImages):

        self.imageDir = imageDir.strip('/')
        self.maskDir = maskDir.strip('/')
        self.resultsPath = "results"
        self.labelsPath = labelsPath
        self.nImages = numImages

        try:
            os.mkdir(self.resultsPath)
            os.mkdir(self.resultsPath+'/Images')
            os.mkdir(self.resultsPath+'/Masks')
        except FileExistsError:
            pass

        copy2(self.labelsPath, self.resultsPath)

        IMG_PATH = os.path.join(self.imageDir, os.listdir(self.imageDir)[0])
        image_size = plt.imread(IMG_PATH).shape[:2]

        self.DATA_PATH = os.path.abspath(os.path.join(self.imageDir, '..')) 
        seed = np.random.randint(100)
        IMGS_OUT = os.path.join(self.resultsPath, 'Images')
        IMGS_SUBDIR = self.imageDir.strip("/").split("/")[-1]

        self.image_generator = self.datagen.flow_from_directory(directory = self.DATA_PATH,
                                                    target_size = image_size,
                                                    save_to_dir = IMGS_OUT,
                                                    classes = [IMGS_SUBDIR],
                                                    class_mode = None,
                                                    save_format = 'jpg',
                                                    seed = seed)

        MASK_OUT = os.path.join(self.resultsPath, 'Masks')
        MASKS_SUBDIR = self.maskDir.strip("/").split("/")[-1]
        self.masks_generator = self.datagen.flow_from_directory(directory = self.DATA_PATH,
                                                    target_size = image_size,
                                                    save_to_dir = MASK_OUT,
                                                    classes = [MASKS_SUBDIR],
                                                    class_mode = None,
                                                    save_format = 'png',
                                                    seed = seed)


        n_iter = int(self.nImages/32)
        if self.nImages % 32:
            n_iter += 1

        print("Copying Original images")
        for img_path, mask_path in zip(os.listdir(self.imageDir), os.listdir(self.maskDir)):
            img_src = os.path.join(self.imageDir, img_path)
            mask_src = os.path.join(self.maskDir, mask_path)
            copy2(img_src, IMGS_OUT)
            copy2(mask_src, MASK_OUT)

        print(f"Generating {n_iter*32} images")
        for i in range(n_iter):
            self.image_generator.next()
            self.masks_generator.next()
            print(i+1,'/',n_iter, 'done')


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    auto = subparser.add_parser('auto')
    auto.add_argument('--imp', type=str, required=True,
                    help="Path where the images are located")
    auto.add_argument('--mp', type=str, required=True,
                    help="Path where the targets are located")
    auto.add_argument('--tp', type=str, required=True,
                    help="Classes path")
    auto.add_argument('--nim', type=int, required=True,
                    help="Number of images")

    arguments = parser.parse_args()

    if arguments.command == 'auto':
        dataAugmentation = DataAugmentation()
        dataAugmentation.DataGenerator(imageDir=arguments.imp, maskDir=arguments.mp, labelsPath=arguments.tp, numImages=arguments.nim)

    else:
        dataAugmentation = DataAugmentation()
        dataAugmentation.DataGenerator(imageDir='./DatasetE2/JPEGImages', maskDir='./DatasetE2/SegmentationClass', labelsPath='./DatasetE2/labelmap.txt', numImages=150)