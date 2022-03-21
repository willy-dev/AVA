#!/usr/bin/env python
# coding: utf-8

# In[1]:


import face_recognition
import os
import cv2


# In[2]:


def known():
    known_faces_dir=r'F:\f\face\face recognition\face_recognition package'
    known_faces=[]
    known_names=[]

    for name in os.listdir(known_faces_dir):
        for filename in os.listdir(f'{known_faces_dir}/{name}'):
            image=face_recognition.load_image_file(f'{known_faces_dir}/{name}/{filename}')
            encoding=face_recognition.face_encodings(image)[0]
            #encoding=pickle.load(open(f'{name}/{filename}','rb'))
            known_faces.append(encoding)
            known_names.append(name)
    return known_faces, known_names


# In[3]:


def get_match():
    known_faces,known_names= known()
    video = cv2.VideoCapture(0)
    TOLERANCE=0.5
    frame_thickness=3
    font_thickness=2
    model='hog'
    img_counter = 0
    while True:
        ret,image=video.read()
        if not ret:
            print("failed to grab frame")
            break
        
        locations=face_recognition.face_locations(image,model=model)
        encodings=face_recognition.face_encodings(image,locations)
        
        for face_encoding ,face_location in zip(encodings,locations):
            results=face_recognition.compare_faces(known_faces,face_encoding,TOLERANCE)
            #print(f'results{results}')
            match=None

            try:
                match=known_names[results.index(True)]
            except ValueError:
                print('Not recognized')
            else:
                print(f"match found:{match}")
                img_counter += 1
        if img_counter==1:
            break 
    video.release()
    return match


# In[ ]:




