#!/usr/bin/env python
# coding: utf-8

# In[1]:


import socket
import bcrypt


# In[ ]:



class OPC_CLIENT_SECURITY:
    def __init__(self):
        self.client_socket = socket.socket()

    def init_opc_client_security(self, server_ip):
        self.client_socket.connect((server_ip, 5000))
        
    def get_opc_server_access(self, credentials):
        print("Requesting server access....")
        self.client_socket.send(credentials.encode())
        result = self.client_socket.recv(1024).decode()
        return result

     
