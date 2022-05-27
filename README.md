# Deep Learning for In Vitro Seed Germination Segmentation

![Alt Text](/Static/seeds_growing.gif)

A neural network trained to segmentate 3 different classes:

* Germinated Seeds
* Not Germinated Seeds
* Background

Brief description of the Repository:

1. **App** folder contains: Visualization Application

1. **CaptureScripts**:
    Scripts used to capture Images on Magnetobiology's Lab

1. **DataManagementScripts** folder:
    * Script for Data Augmentation **augmentation.py**
    * Script to load the dataset **dataset.py**
    * Script to make the tf records passed to the network **tfrecords.py**

1. **Available Models**: 
    * `Alexnet.py`
    * `InceptionResnetV2.py`
    * `MobileNetV2.py`
    * `Unet.py`
    * `UnetVgg.py`

1. **Static**:
    README.md images/gifs

1. **Outside of a folder**:
    Scripts used to build, train and evaluate the chosen model.
    * model.py
    * train.py
    * predict.py
    * utils.py

# Usage

## Clone this repository
```shell
!git clone https://github.com/sbuitragoo/Seed_Segmentation.git
```

## Install the requierements with pip

```shell
pip install requirements.txt
```

## (Optional) Data augmentation

It is possible to perform data augmentation before training. It applies some
random transformations to the input images such as rotations, crops and
shiftings.

Run:

```shell
cd Seed_Segmentation/DataManagementScripts
```

```shell
python3 augmentation.py auto --imp ImagesPath --mp MasksPath --tp LabelMapPath --nim NumberOfImages
```

By default `ImagesPath` is set to `./DatasetE2/JPEGImages` , `MasksPath` to
`./DatasetE2/SegmentationClass` , `LabelMapPath` to `./DatasetE2/labelmap.txt` and `NumberOfImages` to `150` 

The script generates a number of images multiple of 32 as closest as possible to `NumberOfImages`.


## Prepare the dataset

Before training, you must first create two tfrecords files using the script `tfrecords.py`:

```shell
python3 tfRecords.py 
```

## Training

```shell
%cd ..
```

```shell
python3 train.py params --i ImagesPath --m MasksPath --model Model --size Size --epochs Epochs --batch BatchSize --mode tfrecord
```

Where **ImagesPath**, **MasksPath**, **Size**, **Epochs**, **BatchSize** should be replaced with your own data, by default is set to:

```shell
python3 train.py params --i ./augmentedDataPath/images --m ./augmentedDataPath/targets --model mobilenetv2 --size 224 --epochs 50 --batch 32 --mode tfrecord
```

## Seed Examples
![Alt Text](/Static/Seeds.png)

## Segmentation Mask
![Alt Text](/Static/MasksExamples.png)

**Note:** Images look blue because were loaded using `opencv` BGR cvtcolor and displayed using `matplotlib.pyplot` RGB.

