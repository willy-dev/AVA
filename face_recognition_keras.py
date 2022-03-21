#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import requests
import sys
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO

from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img

from tensorflow.keras.applications import imagenet_utils


# In[2]:



    
def get_face():

    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')

    img_counter = 0

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print("[INFO] Found {0} Faces.".format(len(faces)))

        while len(faces)==0:
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            print("[INFO] Found {0} Faces.".format(len(faces)))
            if len(faces)==1:
                break

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            #cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)
            img_counter += 1

        cv2.imshow("test", frame)
        if img_counter==1:
            break

    cam.release()
    cv2.destroyAllWindows()
    roi_color=array_to_img(roi_color).resize((224,224))
    return roi_color

def recognizer(img):
    faces_directory=r'F:\f\face\Mobilenet face recognition\extracted_faces'
    new_model=load_model(f'{faces_directory}/face_classifier.h5')

    img_array=img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    preprocessed_image = tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

    predictions = new_model.predict(preprocessed_image)
    predictions=predictions.tolist()
    max_pred=max(max(predictions))
    print('max pred{}',max_pred)
    if max_pred>=0.9:
        face=np.argmax(predictions)
        return face
    else:
        return float(max_pred)


# In[ ]:




