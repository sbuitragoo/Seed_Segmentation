import os
import tkinter
import cv2
import numpy as np
import tkinter as tk
import tensorflow as tf
from tkinter.ttk import Label
from tkinter import filedialog
from PIL import ImageTk, Image

root = tk.Tk()
root.title("Seed Segmentation")

def Categorical2Mask(self, X, labels):  
    Y = np.zeros(X.shape[0:2] + [3], dtype="uint8")
    for i, key in enumerate(labels):
        Y[...,0] = np.where(X==i, labels[key][0], Y[...,0])
        Y[...,1] = np.where(X==i, labels[key][1], Y[...,1])
        Y[...,2] = np.where(X==i, labels[key][2], Y[...,2])
    return Y

def parse_labelfile(path):
    with open(path, "r") as FILE:
        lines = FILE.readlines()

    labels = {x.split(":")[0]: x.split(":")[1] for x in lines[1:]}

    for key in labels:
        labels[key] = np.array(labels[key].split(",")).astype("uint8")

    return labels

def VisualizeImage():
    imagePath = filedialog.askopenfilename(initialdir="../", title="Select Image",
                                            filetypes=(('image files', '*.png'), ('image files', '*.jpg'), ('All files', '*.*')))
    image = Image.open(imagePath).resize((360,360), Image.ANTIALIAS)
    displayImage = ImageTk.PhotoImage(image)
    imageLabel = tk.Label(frame, text="Image" ,image=displayImage)
    imageLabel.image = displayImage
    imageLabel.place(x=300,y=200)
    

def VisualizeSegmentation():
    imagePath = filedialog.askopenfilename(initialdir="../", title="Select Image",
                                            filetypes=(('image files', '*.png'), ('image files', '*.jpg'), ('All files', '*.*')))
    image = Image.open(imagePath).resize((360,360), Image.ANTIALIAS)
    displayImage = ImageTk.PhotoImage(image)
    targetLabel = tk.Label(frame, text="Image" ,image=displayImage)
    targetLabel.image = displayImage
    targetLabel.place(x=800,y=200)

#model = tf.keras.models.load_model("../model.h5")
model = 1

def Predict(model=model):
    labels = parse_labelfile("labelmap.txt")
    imagePath = filedialog.askopenfilename(initialdir="../", title="Select Image",
                                            filetypes=(('image files', '*.png'), ('image files', '*.jpg'), ('All files', '*.*')))
    imageToSegment = np.expand_dims(cv2.resize(cv2.imread(imagePath), (224,224)), axis=0)
    predictedMask = model.predict(imageToSegment)
    predictedMask = tf.argmax(predictedMask, axis=-1)
    predictedMask = Categorical2Mask(predictedMask[0], labels)
    predictedMask = cv2.resize(predictedMask, (360, 360), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"Predicted_{imagePath}", predictedMask)

    image = Image.open(imagePath).resize((360,360), Image.ANTIALIAS)
    displayImage = ImageTk.PhotoImage(image)
    imageLabel = tk.Label(frame, text="Image" ,image=displayImage)
    imageLabel.image = displayImage
    imageLabel.place(x=300,y=200)

    mask = Image.open(f"Predicted_{imagePath}")
    displayMask = ImageTk.PhotoImage(mask)
    maskLabel = tk.Label(frame, text="Mask" ,image=displayMask)
    maskLabel.mask = displayMask
    imageLabel.place(x=800,y=200)



canvas = tk.Canvas(root, height=1000, width=2000, bg="black")
canvas.pack()

frame = tk.Frame(root, bg="white")
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

title = Label(frame, text="Seed Segmentation Viewer", font=("Arial", 43), background="white").pack()

openPicture = tk.Button(frame, text="Open Picture", fg="black", bg="white", command=VisualizeImage).place(x=710,y=700)

openMask = tk.Button(frame, text="Open Segmentation", fg="black", bg="white", command=VisualizeSegmentation).place(x=690,y=730)

segmentationButton = tk.Button(frame, text="Segment Image", fg="black", bg="white", command=Predict).place(x=700,y=760)


root.mainloop()
