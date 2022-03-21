#!/usr/bin/env python
# coding: utf-8

# In[1]:


from chat_app import*

import sys,time
from socket import socket
import pyttsx3
import datetime
import speech_recognition as sr
from gtts import gTTS # recognize audio
import pyaudio
import regex as re


# In[2]:


from snap7.client import Client as SnapClient
from snap7.types import areas
from snap7.util import *
from opcua import Server, Client, ua, uamethod
from plc_utils import read_data, write_data


# In[3]:


engine = pyttsx3.init()


# In[4]:



global plc
plc=SnapClient()
plc_ip="192.168.0.1"
plc.connect(plc_ip, 0, 1)
plc.get_connected()
plc.get_cpu_state()

engine.say("Client Connected")
engine.runAndWait()
# In[5]:


def voice_data():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print('this is where you start talking...')
        r.adjust_for_ambient_noise(source)
        audio =r.listen(source)

        data=''
    try:
        data=r.recognize_google(audio)
    except sr.UnknownValueError:#check unknown error
        print('i dont understand'
              'google speech recognition does not recognize text')
    except sr.RequestError as e:
        print('SORRY, THE SERVICE IS DOWN'+e)
    return data


# In[6]:



def conversation():
    print("Talk to me.")
    while True:
        keywords=['ph',"setpoint","set","point","value"]
        text=input("You: ")
        #text=voice_data()
        if any(i in text for i in keywords ):
            print("keyword found")
            return text
            break
        print(chatbot_response(text))
        engine.say(chatbot_response(text))
        engine.runAndWait()
        


# In[7]:


def get_ph():
    message=conversation()
    print(message)
    values=re.findall('\d+', message )

    if len(values)==0:    
        while True:
            print("didnt get that.what setpoint do you want?")
            message=conversation()
            values=re.findall('\d+', message ) 
            if len(values)>0:
                ph_value=int(values[0])
                break
    else:
        ph_value=values[0]
        
    return ph_value


# In[8]:


def set_PH():
    #init_plc()
    setpoint=get_ph()
    #write_data(plc, 'DB3.DBD0', setpoint)
    
    print('INFO:setpoint set to {}'.format(setpoint))
    engine.say('setpoint set to {}'.format(setpoint))
    engine.runAndWait()
    
    print('INFO:starting process')
    engine.say('starting process')
    engine.runAndWait()
    write_data(plc, 'M0.3', False)#set stop to False
    write_data(plc, 'M0.2', True)#set start to True


# In[9]:


def stop_process():

    stop_words=['stop','end','halt']
    text=input("You: ")
    #text=voice_data()
    if any(i in text for i in stop_words ):
        print("stop word found")
        write_data(plc, 'M0.3', True)#stop to true
        write_data(plc, 'M0.2', False)#start to false


# In[ ]:






