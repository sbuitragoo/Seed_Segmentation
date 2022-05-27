import os
import tkinter
import cv2
import tkinter as tk
from tkinter.ttk import Label
from tkinter import filedialog
from PIL import ImageTk, Image

root = tk.Tk()
root.title("Seed Segmentation")

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


canvas = tk.Canvas(root, height=1000, width=2000, bg="black")
canvas.pack()

frame = tk.Frame(root, bg="white")
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

title = Label(frame, text="Seed Segmentation Viewer", font=("Arial", 43), background="white").pack()

openPicture = tk.Button(frame, text="Open Picture", fg="white", bg="#263D42", command=VisualizeImage).place(x=710,y=750)

openMask = tk.Button(frame, text="Open Segmentation", fg="white", bg="#263D42", command=VisualizeSegmentation).place(x=690,y=780)


root.mainloop()
