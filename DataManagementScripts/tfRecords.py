from dataset import parse_labelfile, mask2categorical
import matplotlib.pyplot as plt
from numpy import random
import tensorflow as tf
from os import path
import numpy as np
import argparse
import os


class MakeTfRecords():

    def __init__(self):
        pass

    def bytes_list_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def train_test_split(self, imagesPath, masksPath, val_size):
        """Return 4 lists of paths in the following order:
        - Train images
        - Train masks
        - Validation images
        - Validation masks
        """
        self.trainImagesPath = []
        self.trainMaskPath = []
        self.valImagesPath = []
        self.valMaskPath = []

        indexes = []
        nImages = len(imagesPath)
        n_val_imgs = int(val_size*nImages)
        while len(indexes) < nImages:
            x = random.randint(nImages)
            if x not in indexes:
                indexes.append(x)

        for i in indexes[0:n_val_imgs]:
            self.valImagesPath.append(imagesPath[i])
            self.valMaskPath.append(masksPath[i])

        for i in indexes[n_val_imgs:]:
            self.trainImagesPath.append(imagesPath[i])
            self.trainMaskPath.append(masksPath[i])

        return self.trainImagesPath, self.trainMaskPath, self.valImagesPath, self.valMaskPath

    def create_example(self, img, mask, labels):

        """Creates a tensorflow Example from an image with its mask.
        Parameters
        -----------
        img : str
            path to the image
        mask : str
            path to the mask
        labels : dict
            dict with the corresponding rgb mask values of the labels
        Returns
        --------
        tf.train.Example   
        """

        encoded_img = tf.io.read_file(img)
        encoded_mask = tf.io.read_file(mask)
        decoded_mask = tf.io.decode_image(encoded_mask)
        mask = mask2categorical(decoded_mask, labels)
        mask = tf.expand_dims(mask, axis=-1)

        encoded_mask = tf.io.encode_png(mask) # Re-encoding the mask

        example = tf.train.Example(
                    features=tf.train.Features(feature={
                        'image': self.bytes_list_feature(encoded_img.numpy()),
                        'mask': self.bytes_list_feature(encoded_mask.numpy())
                        }))
        return example


    def main(self, imagePath, maskPath, tfRecordPath, labels, valSize):

        try:
            os.mkdir(tfRecordPath)
        except:
            pass
        
        self.imagePath = imagePath
        self.maskPath = maskPath
        VAL_TFRECORD = path.join(tfRecordPath, "val-data.tfrecord")
        TRAIN_TFRECORD = path.join(tfRecordPath, "train-data.tfrecord")
        LABELS = parse_labelfile(labels)
        val_size = valSize

        # Create 2 lists containing the paths of the images and the masks
        img_path = [path.join(self.imagePath, imgs) for imgs in np.sort(os.listdir(self.imagePath))]
        mask_path = [path.join(self.maskPath, imgs) for imgs in np.sort(os.listdir(self.maskPath))]

        train_img, train_mask, val_img, val_mask = self.train_test_split(img_path, mask_path, val_size)

        # Create and fill the train-data.tfrecord with the examples 
        tf_record_train = tf.io.TFRecordWriter(TRAIN_TFRECORD)
        for img, mask in zip(train_img, train_mask):
            example = self.create_example(img, mask, LABELS)
            tf_record_train.write(example.SerializeToString())
        tf_record_train.close()

        #Create and fill the val-data.tfrecord with the examples
        tf_record_val = tf.io.TFRecordWriter(VAL_TFRECORD)
        for img, mask in zip(val_img, val_mask):
            example = self.create_example(img, mask, LABELS)
            tf_record_val.write(example.SerializeToString())
        tf_record_val.close()

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    auto = subparser.add_parser('auto')
    auto.add_argument('--imp', type=str, required=True,
                    help="Path where the images are located")
    auto.add_argument('--mp', type=str, required=True,
                    help="Path where the targets are located")
    auto.add_argument('--tfp', type=str, required=True,
                    help="TFRecords path")
    auto.add_argument('--labels', type=str, required=True,
                    help="Labels")
    auto.add_argument('--valsize', type=int, required=True,
                    help="Valsize")

    arguments = parser.parse_args()

    if arguments.command == 'auto':

        tfRecords = MakeTfRecords()
        tfRecords.main(imagePath = arguments.imp, maskPath = arguments.mp, tfRecordPath = arguments.tfp, labels = arguments.labels, valSize = arguments.valsize)

    else:

        tfRecords = MakeTfRecords()
        tfRecords.main(imagePath='../../DatasetE2/JPEGImages', maskPath='../../DatasetE2/SegmentationClass', tfRecordPath = './tfrecords/', labels = '../../DatasetE2/labelmap.txt', valSize = 0.2)
