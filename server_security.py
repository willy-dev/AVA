#!/usr/bin/env python
# coding: utf-8

# In[2]:



import socket
import bcrypt
import json
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils


# In[ ]:



class OPC_SERVER_SECURITY:
    def __init__(self):
        self.server_socket = socket.socket()
        self.salt = bcrypt.gensalt()

    def init_opc_server_security(self, ip):
        self.server_socket.bind((ip, 5000))
        self.server_socket.listen(1)
    
    def client_authentication(self):
        is_client_authenticated = False
        print("Waiting for clients....")

        conn, address = self.server_socket.accept()
        while True:
            #face=recognizer(img)
            recvd_name=conn.recv(1024).decode()
            recvd_name=float(recvd_name)
            registered_users=[0,1,2,3]
            if any(i in recvd_name for i in registered_users):
                conn.send("Success".encode())
                is_client_authenticated = True
                break
            else:
                conn.send("Failure".encode())
                is_client_authenticated = False
                break
        
        if is_client_authenticated:
            print("Server access granted")
            return True
        else:
            print("Server access denied")
            return False
    

