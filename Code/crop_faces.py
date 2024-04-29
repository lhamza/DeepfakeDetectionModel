#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:01:35 2024

@author: apple
"""

import os
import cv2
from skimage import io, color, img_as_ubyte
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches

dataset_path = "/Users/apple/Desktop/uni/year3/term2/PersonalProject/Dataset/Dataset"
new_dataset_path = "/Users/apple/Desktop/uni/year3/term2/PersonalProject/Dataset/Dataset_Cropped"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for root, dirs, files in os.walk(dataset_path):
    for filename in files:
        if filename.endswith(".jpg"):
            
            folder = os.path.basename(root)
            
            img = io.imread(os.path.join(root, filename))
            img_gray = color.rgb2gray(img)
            img_gray = img_as_ubyte(img_gray)
            
            
            faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
         
            #fig, ax = plt.subplots(figsize=(18, 12))
            
            counter = 0
            
            for face in faces:
                
                counter+=1
                new_filename = str(counter)+filename
                
                x, y, w, h = face[0], face[1], face[2], face[3]
                face_box = (x, y, x+w, y+h)
                cropped_img = img[y:y+h, x:x+w]
                face_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                
                cv2.imwrite(os.path.join(new_dataset_path, folder, new_filename), face_img)
                print(folder, new_filename, "copied")
            #ax.imshow(cropped_img), ax.set_axis_off()    
            #fig.tight_layout
            #plt.show()
            
            
            
            
    #crop image
    #if no face detected then leave as it is
    #
            
            
           


