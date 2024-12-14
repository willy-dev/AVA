# Face Recognition with Liveness Detection Login on Flask Web interface for Siemens PLCs
# Digital twin for Amatrol T554 Siemens Process control station at Dekut Siemens Center

Big thanks to Runo Harris, Margaret Kagwiria, Naomi Nyambura and Evelyne kawara, fellow Mechatronics Engineers and Data Scientist for all the help in building this project, you deserve beers my Gs
## Project Overview
&nbsp;&nbsp;&nbsp;&nbsp;We have implemented Flask web application login page including face verification (1-to-1 to verify whether the person who is logging in is really that person), for security purpose, with liveness detection mechanism (to check whether the person detected on the camera is a REAL person or FAKE (eg. image, video, etc. of that person)), for Anti-Spoofting (Others pretending to be the person), built with Convolutional Neural Network. After the login page, we also provided a webpage placeholder for future use.   
  
**Login component:**  
<img src="./assets/login-page.png" width=374 height=521>  
  
**Loggin in and running inference process:**  
<img src="./assets/short_demo.gif">

# Digital-Triplet (2021)
Virtual assistants for Digital Twin systems  


# Task
Its 2021 and the world is in the fourth insustrial revolution!
Engineering Cyber-Physical systems are evolving from Digital Twin models to Digital Triplet.
At Dedan Kimathi University of Technology (DeKUT) SIEMENS Mechatronics Systems training centre, we are tasked to add Face-Recogniton security access and Voice assistants to the Digital Twin models of the Analytic and Thermal Process Control stations.
This repository outlines the approaches we took to achieve this

# Analytic and Thermal Process Control stations
![image](https://user-images.githubusercontent.com/83555928/116885724-0e3e0600-abdd-11eb-81c9-bcb157bec205.png)
![image](https://user-images.githubusercontent.com/83555928/116886682-39752500-abde-11eb-9f44-d28c2f63cf82.png)

Both of these stations are developed by Amatrol to teach and train on process control which is a vital part of major industries, including: power generation; petrochemicals; food processing and bottling; chemical manufacturing; biotechnology; pharmaceuticals and refineries.

The Analytic process control station monitors and regulates the pH of a process fluid. The station runs on PID Controller Module – Single Loop (T5554-C1-A) or PID Controller Module – Dual Loop (T5554-C2-A)

The Thermal process control station, on the other hand, regulates temperature. The controller for this station is PID Controller Module – Dual Loop (T5553-C2-A).

Here is a link to Amatrol's site if you wish to know more on Process control machines
https://amatrol.com/product/process-control-training/

In 2018, the team of Isaiah Nassiuma, Emanuel Sunguti, Langat KIpkoech Rogers and Timothy Kimari proposed a new control method based on the Siemens S7 1200 PLC PID control functionality.

In 2020, the controller used for both stations was the Siemens S7 1200 PLC (CPU 1214C DC/DC/DC)

# Criteria
1. Digital Twin models for the stations
2. Real time Face recognition
3. Intelligent voice assistant
4. Real time mask detection to ensure the operator always wears his mask correctly

# Tech Stack Summary
We used Siemens NX software to build the digital twin models. With the software, we could design and simulate realizing the best from the digital twins

PLC programs were written in the TIA portal(Totally Integrated Automation)

Face recognition, voice assistant and mask detection programmes were written in Python. We chose Python due to its popularity in Artificial intelligence implementation. our face and voice recognition models had to be intelligent. Secondly, and full disclosure I believe this is one of the major reasons, we didnt have a lot of experience programming. Python has a very huge community, which is only growing, and growing😊 since the introduction of deep learning. We joined the community and keep learining so much! From basic programming, through machine-learning and deep learning fundamentals to deep learning APIs (Keras and a bit of Pytorch)

The mask detection model was created in Tensorflow Keras API

Communication between client machine and server computer was done through OPC UA protocol(Open Platform Communications Unified Architecture)

The Snap7 library was used to interface Python programs with the S7 1200 PLC

# Tech stack
1. Siemens NX
2. Siemens TIA portal
3. Python 
4. Tensorflow Keras API
5. OPC UA protocol
6. Snap7 library
7. Python text to speech
8. Natural Language Processing
9. Speech Recognition
10. Numpy

# Functionality
1. Real time digital twin model
2. Realtime face recognition
3. General conversation with the machine
4. Specify the setpoints using voice in English
5. Control different components (pumps, motors, chiller) of the stations using speech 
6. Real time mask detection for the operator

# Breakdown of approaches taken
The following section is a breakdown of the directories at the top of this repository
## Face-recognition
We did face recognition in two ways;
1. Using Python's face-recognition library 
2. Through transfer learning  
  
I will describe our experiences with both outlining the challenges faced, ways we went around them and possible solutions for those that we didn't
### Using Python's face-recognition library
This was the easiest and most accurate. And why not, its basically recognition through comparison. Locate a face from the camera feed, compare it with saved images, and boom! Match found! or no boom😶if unauthorized
#### Challenges
Every authorized user has a folder with their image. When the program runs, it processes all the images in all the folders, opens the camera and compares the face from the feed with the known faces trying to find a match.  
As the number of authorized users increases, so does the number of images to be processed. This means more processing time and it becomes increasingly difficult for the program to work in real time.
#### Solution
Use a face cascade classifier from open-cv to extract faces from images of known users. Use these extracted faces as reference images since they are smaller in size than selfies and full size images. A smaller image means shorter processing times  
If the program still has challenges working in real time, switch it up, elevate, and use deep learning models like a pro!😎

### Through transfer learning
This is possibly the most powerful way to do face-recognition and other image classification tasks eg mask detection. It involves using pre-trained deep learning models and 'transfering' the knowledge learned on those tasks to similar tasks. Deep learning models can work in realtime even with increased number of authorized users. However, their accuracy is dependent on how well the models learns to generalize during training.  
We transfer-learned Google's MobileNet and ResNet50 a variant of ResNet model
#### MobileNet
MobileNets are a class of light weight deep convolutional neural networks for mobile and embedded vision applications. They can run on mobile devices due to their small sixe. the size of the largest MobileNet currently is about 17 MB. Comparatively, VGG16, a model used in computer vision tasks is 533 MB.  

MobileNets are trained on the ImageNet dataset. The MobileNet in the Keras functional models has 1000 classifications. When we retrained the model to our custom dataset, we changed the classification layers to 4 nodes, each representing an authorized user. We also retrained some of the hidden layers. The only layers we didnt retrain were the top bottleneck layers used for feature extraction.  

The code at the top of this repo has comments to help you follow along. If you would like a detailed tutorial on how to work with MobileNet in Keras, please check out Deeplizard's Keras series. They have both video and text based resources. Tutorials 17 through 19 are on MobileNets. We learnt a lot from them.(Thankyou so much! Mandy and Chris😀)
https://deeplizard.com/learn/video/OO4HD-1wRN8

##### Challenges
These are more of useful insights but can easily challenge how the model learns if not observed.
1. Balanced number of images- sounds pretty obvious right? Have the same number of images for each member(whether in the training set or validatuon set). First time retraining the model, I didnt think  there would be much difference with using 150 images(which are certainly not enough) for one user and 100 for another. But yes, there was. The model would give better predictions on the first user and perform comparatively worse on the other.

2. Variations and number of images- Augment your images. Flip horizontally, rotate, zoom,crop; these variations in the training and validation set help a model to generalize better. There is a program on data augmentation in Keras at the top of this repo. If you don't have access to more images, you can specify the number of images to generate from each image. 

3. Number of hidden layers to retrain- As mentioned earlier, we freeze the top layers, retrain classification and some hidden layers. Experiment with this number(hidden layers). Find the optimum number for your dataset.

4. GPU training- As you may have realized, training for the optimum weights is through trial and error. You end up training your model a number of times before it gives accurate predictions. This training process takes very long on the CPU. If you dont have access to a machine with a NVIDIA GPU, run your code on Google Colab and utilize the GPUs at Google. 

#### Resnet50
ResNet50 is a variant of ResNet model which has 48 Convolution layers along with 1 MaxPool and 1 Average Pool layer. It has 3.8 x 10^9 Floating points operations. It is a widely used ResNet model. Because of the framework that ResNets presented it was made possible to train ultra deep neural networks and by that I mean that the network can contain hundreds or thousands of layers and still achieve great performance.The ResNets were initially applied to the image recognition task but the framework can also be used for non computer vision tasks also to achieve better accuracy. 

##### Challenges
As we know that Deep Convolutional neural networks are really great at identifying low, mid and high level features from the images and stacking more layers generally gives us better accuracy so a question arrises that is getting better model performance as easy as stacking more layers? With this questions arises the problem of vanishing/exploding gradients. These problems were largely handled by many ways and enabled networks with tens of layers to converge but when deep neural networks start to converge we see another problem of the accuracy getting saturated and then degrading rapidly and this was not caused by overfitting as one may guess and adding more layers to a suitable deep model just increased the training error.

##### Solution
This problem was further rectifed by by taking a shallower model and a deep model that was constructed with the layers from the shallow model and and adding identity layers to it and accordingly the deeper model shouldn't have produced any higher training error than its counterpart as the added layers were just the identity layers.
![image](https://user-images.githubusercontent.com/74649440/117497784-97f52700-af81-11eb-94c6-06714038055f.png)

In the Figure above we can see on the left and the right that the deeper model is always producing more error, where in fact it shouldn't have done that.The authors addressed this problem by introducing deep residual learning framework so for this they introduce shortcut connections that simply perform identity mappings

## Mask detection
We accomplished mask detection through transfer learning using Google's MobileNet.

#### Challenges
The challenges of mask detection through transfer learning are similar to those of face recognition. For accurate mask detection, we need to use a good quality camera with high resolution.

## Intelligent voice assistant
The intelligent voice assistant incorporated two types of chatbots;
  1. Speech Recognition
  2. Keyword chatbot
  3. AI chatbot
 
### Speech Recognition
The voice assistant used the python SpeechRecognition library to enable the machine to understand voice commands from the operator. It also uses the try and except functions to avoid unnecessary errors eg; UknownValueError which is caused by the program not understanding your statement and RequestError which happens when the internet is disconnected. Instead of the errors interrupting the program, the voice assistant will inform the operator ie;
  1. UknownValueError - I cant understand your statement
  2. RequestError - Your internet is disconnected

#### Challenge
The SpeechRecognition library is a bit slow and does not facilitate immediate responses. It takes a few seconds to understand a spoken statement; having to run the audio through google text to speech, whose effectiveness largely dependa on your internet speed.

#### Solution
The best way to use the SpeechRecognition library is by using your laptop's inbuilt microphone: if it is functional, otherwise use a good external device with a good microphone eg a usb microphone, usb camera with an inbuilt microphone, headphones or earphones equipped with a good microphone.
Make sure you have a reliable internet connection

### Keyword chatbot
The keyword chatbot used specific keywords to directly write data in TIA portal by using the Snap-7 library in python. This was mainly accomplished by using **if** statements. This enforces the security feature of the machine only if the authorized users are the only ones who have access to the keyworded commands. The snap-7 library writes and reads the required data in the TIA portal program by using the correct PLC addresses in the TIA portal program.

### AI chatbot
 The AI chatbot is mainly used to facilitate conversations about the machine processes with the operator. This was accomplished by first creating a dictionary file with all the possible questions or statements that the operator might use to ask about machine processes and the required answers. With this feature the machine will be able to train a new operator on all the processes that the machine can do. We incorporated natural language processing in creating the AI chatbot model by using nltk(natural language tool kit) and Tensorflow Keras API. The Natural Language Toolkit (NLTK) is a platform used for building Python programs that work with human language data for applying in statistical natural language processing (NLP).It contains text processing libraries for tokenization, parsing, classification, stemming, tagging and semantic reasoning. 
 
# Communication with the PLC
So, you now have a voice assistant. You can communicate in natural language; you can have a general conversation with the assistant, you can specify the setpoint, you can tell the assistant to turn the chiller on, etc. How exactly does this happen? How does the assistant(written in python) communicate with the PLC, a device running on a completely different language?   

Recall from the tech stack summary, we said we use the snap7 library to interface python programs with the PLC. How does snap7 do this? Is it magic? Oh, does snap7 have other special abilities?  

Well, its not magic, and it certainly doesn't have 'special' abilities. The logic behind how it works lies in the memory of the PLC.  
## The memory of the PLC
If you have worked with PLCs, you're probably familiar with the tag table. if you have not, have no worries. We will go through what it is, and how it relates to the memory of a PLC. Below is a snap of the tag table for the analytic station.  

![image](https://user-images.githubusercontent.com/83555928/118378804-eeb6be00-b58a-11eb-8327-bd261914cc81.png)  

All variables in a PLC program can be viewed from the tag table. Every variable has an associated data type and an address.   

Take the agitator motor for instance. The agitator motor is used for stirring acid,water and base in the mixing tank. It is connected to the output module of the PLC. The motor is either on or off. Therefore the associated data type is boolean(Bool). The address is %Q0.5, the 'Q' indicating an output.

If you look at the scaled PH input, the data type is real(floating point) and the address is %MD120. The 'M' stands for memory. The 'D' shows the position in memory where the data lies. 'D' indicates a double word(32 bit). In this PLC's memory, the scaled PH input is stored in the 32 bit position.  

All variables, excluding timers and counters, can be stored as inputs, outputs or memory within the memory of the PLC.  

When we work with PLCs, we access these memory locations, and either write to, or read from them.  

## The communication
To talk python to a PLC using the snap7 library, we need a python program that can find these memory locations, and then read or write to them using methods from util and types classes of the snap7 library.  
Luckily for us, Andrew Kibor has this python script on his github. This script(plc_utils.py) is at the top of this repository. The script has two functions; read_data and write_data. read_data takes two parametes; plc and the key (address tag). The function finds the location in memory of the tag and reads from it.  
write_data takes three parameters; the plc, the key to write to, and the value to write to that tag.

## The Virtual Simulation
We also added a virtual simulation feature to the project. This was achieved by mapping the signals from the input and output devices to the Simulation NX CAD design assembly using the PLC's inbuilt OPCUA server. The names of the mapped signals are defined in the PLC program, collected in the OPCUA python server code and then mapped in the MCD (Mechatronics Concept Designer) signals on NX CAD design.

### Thermal Process Control Simulation

![image](https://user-images.githubusercontent.com/74649440/122034713-7554fa00-cdda-11eb-83cc-718b6a36589b.png)

The signals are collected from the PLC program and are fed into the server code then mapped into the NX design to show the status of the components and processes. To see the complete server code please check the server code in the Thermal Process Control Python Codes folder and open signal_server file.

![image](https://user-images.githubusercontent.com/74649440/122029899-f5c52c00-cdd5-11eb-8d92-543c22837aa1.png)

In the Thermal Process Control project, we used Colour Coding method since there were no moving parts eg motors or rails. Specific colours were used to indicate whether some components were on and the type of process that is taking place ie water cooling or water heating. Consider the example below:

![image](https://user-images.githubusercontent.com/74649440/122030812-c2cf6800-cdd6-11eb-9ddb-82cece7fb29f.png)

This image indicates the Cooling process since:
  1. The Process Tank is blue in colour
  2. The Chiller is blue in colour which indicates it is on
  3. The Chiller heat exchanger is green in colour which indicates it is on
  4. The hot water tank, hot water heat exchanger and hotwater pump are orange in colour which indicates the heater is off and the three-way valve that controls the hot water      flow has been closed completely to prevent any rise in temperature.
  5. The Process Pump is green in colour which indicates that it is on.

This image below indicates the Heating Process

![image](https://user-images.githubusercontent.com/74649440/122114465-fa1c3400-ce2b-11eb-8326-143c149697ca.png)

  1. The Process Tank is red in colour to indicate rise in temperature
  2. The Chiller and Chiller Heat Exchanger are orange in colour to indicate they are off.
  3. The Hot water tank is blue in colour to indicate that the heater is on.
  4. The hot water pump and hot water heat exchanger are green in colour to indicate that they are on and hot water is being pumped through the three-way valve.
  5. The Process Pump is green in colour which indicates that it is on.

## Result
* The face recognition works well detecting face and accurately recognizing the face.
* The liveness detection works well with classifying fake images and videos from smartphone spoofing.
* The liveness detection is also trained to be able to classify solid-printed images (image on papers and cards). But it's trained only with around 10 images, so it doesn't work well everytime (read "Full Workflow usage and Training your own model" section, in case you want to train it yourself with bigger dataset)

## Packages and Tools
Use requirements.txt if you want to install the correct version of packages.
- OpenCV
- TensorFlow 2
- Scikit-learn
- Face_recognition
- Flask
- SQLite
- SQLAlchemy (for Flask)
- OPCUA (for plc communication with the code/ server and client)
- Snap7 (for Python and PLC communication)

## Project structure
### This section will take you through the workings of the app
Basically, the repo indicates how you can leverage AI as a security mechanism in a web app for PLC or Programmable logic controller operations.
This project may be used globaly for any PLC as long as the dependecies are configured correctly
We used Siemens S7 1200 PLC for our process

So basically, the app is deployed to the workstation network, anyone who wants access should be authorized by admin first and have their photo data captured and fed into the model.
The user can now use th facial recognition system to access the process that relays a start and stop function but also includes a voice recognition AI where one specify PH by voice command.

This command is relayed to the program and the command and value are interpreted as data tables or states in the TIA Portal application, they are then written into the system PLC and PID control can now regulate the flow of acid or base depending on the value(e.g. if PH value command is 5, the PH value reading is 7, the acid tank should pour more acid than the base tank until the PH is at 5 then it should stop)
  
<img src = "./assets/project_structure.png" width=446 height=702>  

## Files explanation
&nbsp;&nbsp;&nbsp;&nbsp;In this section, we will show how to use/execute each file in this repo. The full workflow of the repo, from the data collection till running the web application, will be provided step-by-step in the next section.  
*Note: All files have code explanation in the source code themselves*
* In **main** folder (main repo)
  * **`app.py`**: This is the main Flask app. To use it, just normally run this file from your terminal/preferred IDE and the running port for your web app will be shown. **If you want to deploy this app, don't forget to change app.secret_key variable**
  * **`database.sqlite`**: This is the minimum example of database to store user's data and it is used to retrieve and verify while user is logging in.  
* In face_recognition_and_liveness/**face_recognition** folder
  * **`encode_faces.py`**: Detect the face from images, encode, and save encoded version and name (label) to pickle files. **The name/label coms from the name of folder of those input images.**  
    Command line argument:
    * `--dataset (or -i)` Path to input directory of images
    * `--encoding (or -e)` Path/Directory to save encoded images pickle files
    * `--detection-method (or -d)` Face detection model to use: 'hog' or 'cnn' (default is 'cnn')  
    **Example**: `python encode_faces.py -i dataset -e encoded_faces.pickle -d cnn`  
  * **`recognize_faces.py`**: Real-time face recognition on webcam with bounding boxes and names showing on top.  
    Command line argument:
    * `--encoding (or -e)` Path to saved face encodings
    * `--detection-method (or -d)` Face detection model to use: 'hog' or 'cnn' (default is 'cnn')  
    **Example**: `python recognize_faces.py -e encoded_faces.pickle -d cnn`
  * **`dataset`** folder: Example folder and images to store dataset used to do face encoding. There are subfolders in this folder and each subfolder must be named after the owner of image in that subfolder because this subfolder name will be used as a label in face encoding process.
* In face_recognition_and_liveness/**face_liveness_detection** folder
  * **`collect_dataset.py`**: Collect face in each frame from a *video* dataset (real/fake) using face detector model (resnet-10 SSD in this case) and save to a directory (we provided a video example in videos folder, so you can collect the correct video dataset to train the model)  
    Command line argument:
    * `--input (or -i)` Path to input input video
    * `--output (or -o)` Path/Directory to output directory of cropped face images
    * `--detector (or -d)` Path to OpenCV\'s deep learning face detector  
    * `--confidence (or -c)` Confidence of face detector model (default is 0.5 | 50%)
    * `--skip (or -s)` Number of frames to skip before applying face detection and crop (dafault is 16). The main idea for this is the consequence frames usually give the same face to the dataset, so it can easily causes overfitting and is not a useful data for training.  
    **Example**: example for *fake video* dataset -> `python collect_dataset.py -i videos/fake_1.mp4 -o dataset/fake -d face_detector -c 0.5 -s 15` | example for *real video* dataset -> `python collect_dataset.py -i videos/real_1.mp4 -o dataset/real -d face_detector -c 0.5 -s 15`
  * **`face_from_image.py`**: Collect face in each frame from a *image* dataset (real/fake) using face detector model (resnet-10 SSD in this case) and save to a directory (we provided a video example in videos folder, so you can collect the correct video dataset to train the model)  
    Command line argument:
    * `--input (or -i)` Path to input input image (A single image | Since we mainly collect dataset from videos, we use this code only to collect face from those solid-printed picture (picture from paper/card) and we didn't have many of them. So, we make the code just for collect face from 1 image. Feel free to adjust the code if you want to make it able to collect faces from all image in a folder/directory)
    * `--output (or -o)` Path/Directory to output directory of cropped face images
    * `--detector (or -d)` Path to OpenCV\'s deep learning face detector  
    * `--confidence (or -c)` Confidence of face detector model (default is 0.5 | 50%)  
    **Example**: example for *fake image* dataset -> `python face_from_image.py -i images/fakes/2.jpg -o dataset/fake -d face_detector -c 0.5` | example for *real image* dataset -> `python face_from_image.py -i images/reals/1.jpg -o dataset/real -d face_detector -c 0.5`
  * **`livenessnet.py`**: Model architecture for our liveness detection model and build function to build the neural network (there is no command line arguement for this file (no need to do that)). The class *LivenessNet* will be called from the `train_model.py` file in order to build a model and run the training process from train_model.py file.
  * **`train_model.py`**: The code used to train the liveness detection model and output .model, label_encoder.pickle, and plot.png image files.  
    Command line argument:
    * `--dataset (or -d)` Path to input Dataset
    * `--model (or -m)` Path to output trained model
    * `--le (or -l)` Path to output Label Encoder 
    * `--plot (or -p)` Path to output loss/accuracy plot  
    **Example**: `python train_model.py -d dataset -m liveness.model -l label_encoder.pickle -p plot.png`
  * **`liveness_app.py`**: Run face detection, draw bounding box, and run liveness detection model real-time on webcam  
    Command line argument:
    * `--model (or -m)` Path to trained model
    * `--le (or -l)` Path to Label Encoder
    * `--detector (or -d)` Path to OpenCV\'s deep learning face detector  
    * `--confidence (or -c)` Confidence of face detector model (default is 0.5 | 50%)  
    **Example**: `python liveness_app.py -m liveness.model -l label_encoder.pickle -d face_detector -c 0.5`
  * **`face_recognition_liveness_app.py`**: This is the **core** file combined both face recognition and liveness detection together and run them concurrently. The current version of this file is refactored in order to use in the main `app.py` file, so this file **doesn't support command line arguements**. **However**, we have provided the code for command line inside the source code itself and commented those lines. If you really want to run in completely from command line, uncomment those lines and comment out the function structure (header, return, and if \_\_name\_\_ == '\_\_main\_\_') and that's it. In case you want to run from command line, we will provide the command line arguement and example here.  
    **Example if you don't modify the code**: `python face_recognition_liveness_app.py`  
    Command line argument:
    * `--model (or -m)` Path to trained model
    * `--le (or -l)` Path to Label Encoder
    * `--detector (or -d)` Path to OpenCV\'s deep learning face detector  
    * `--confidence (or -c)` Confidence of face detector model (default is 0.5 | 50%)  
    * `--encodings (or -e)` Path to saved face encodings  
    **Example if you modify the code and use command line argument**: `python face_recognition_liveness_app.py -m liveness.model -l label_encoder.pickle -d face_detector -c 0.5 -e ../face_recognition/encoded_faces.pickle`
  * **`dataset`** folder: Example folder and images for training liveness detection model. (These images are outputs of `collect_dataset.py`)
  * **`face_detector`** folder: The folder containing the caffe model files including .prototxt and .caffemodel to use with OpenCV and do face detection
  * **`images`** folder: Example folder and images for inputting to `face_from_image.py`
  * **`videos`** folder: Example folder and videos for inputting to `collect_dataset.py`

## Basic usage
1. Download/Clone this repo
2. Download the dependecies in the file(i'll list them soon)
3. Run app.py
4. That's it!  
  
**Note: Doing only these steps will allow only Liveness detection to work but not Recognition and Full login mechanism (for full workflow and training our own model, please keep reading and follow the next section)**

## Full Workflow usage and Training your own model
**We'll start with face recognition (step 1-5) and we'll do face liveness detection next (step 6-13). Finally, we'll combine everything together (step 14 til the end)**
1. Create 1 folder for 1 person and name after the person's name in *face_recognition/dataset* (you can take a look at this folder in this repo for example)
2. Collect the images showing full face (1 face per 1 image per 1 person). Since we are using **1 shot learning** technique, collect only up to 10 images for each person would be enough.
3. Run `encode_faces.py` like the example above in files explanation section
4. Now you should get encoded faces file ending with .pickle in the path you specify (if you follow the code above, you should see it in the same folder with this file)
5. Run `recognize_faces.py` like the example above in files explanation section and see whether it works well.
6. Collect video of yourself/others in many light condition (the easiest way to do this is to film yourself/others walking around your/others home) and save to *face_liveness_dection/videos* folder. *The length of the video depends on you.* You don't need to name it with the word 'real' or 'fake'. It's just convention that we found helpful when calling from other codes. **Take a look into that folder, we have dropped some example videos there.**
7. Use those recorded videos and play it on your phone. Then, hold your phone and face the phone screen (running those recorded videos) to the webcam and record your PC/laptop screen. By doing this, you are creating the dataset of someone spoofing the person in the video / pretending to be the person in the video. Try to make sure this new spoofing video has the same length (or nearly) as the original one because we need to avoid *unbalanced dataset*. **Take a look into that folder, we have dropped some example videos there.**
8. Run **`collect_dataset.py`** like the example above in files explanation section for every of your video. Make sure you save the output into the right folder (should be in `dataset` folder and in the right label folder `fake` or `real`). Now you must see a lot of images from your video in the output folder.
9. (Optional, but good to do in order to improve model performance) Take a picture of yourself/others from a paper, photo, cards, etc. and save to *face_liveness_detection/images/fakes*. **Take a look into that folder, we have dropped some example videos there.**
10. If you do step 9, please do this step. Otherwise, you can skip this step. Take more pictures of your face / others' face **in the same amount of the fake images you taken in step 8** and save to *face_liveness_detection/images/reals*. Again, by doing this, we can avoid *unbalanced dataset*. **Take a look into that folder, we have dropped some example videos there.**
11. (Skip this step if you didn't do step 9) Run **`face_from_image.py`** like the example above in files explanation section for every of your image in images folder. Make sure you save the output into the right folder (should be in `dataset` folder and in the right label folder `fake` or `real`). *Note: Like we discussed in files explanation section, you have to run this code 1 image at a time. If you have a lot of images, feel free to adjust the code. So you can run only once for every of your image. (But make sure to save outputs to the right folder)*
12. Run **`train_model.py`** like the example above in files explanation section. Now, we should have .model, label encoder file ending with .pickle, and image in the output folder you specify. If you follow exact code above in files explanation, you should see liveness.model, label_encoder.pickle, and plot.png in this exact folder (like in this repo).
13. Run **`liveness_app.py`** like the example above in files explanation section and see whether it works well. If the model always misclassify, go back and see whether you save output images (real/fake) in the right folder. If you are sure that you save everything in the right place, collect more data or better quality data. *This is the common iterative process of training model, don't feel bad if you have this problem.*
14. Run **`face_recognition_liveness_app.py`** like the example above in files explanation section and see whether it works well. The window should show up for a while with bounding box having your name on top and whether real or fake with probability. If your code work properly, this window will be closed automatically after the model can **detect you and you are real** for **10 consequence frames**. By specifying this 10 consequence frames, it can make sure that the person on the screen is really that person who is logging in and real, not just by accident that the model misclassifies for some frames. It also allows some room for model to misclassify. You can adjust this number in **line 176 'sequence_count' variable** in the code, if you want it to be longer or shorter. And right after the window closes, we should see your name and label (real) on the command line/terminal.
15. In **`app.py`**, in the main folder of this project, go down to **line 56** and so on. Uncomment those lines and change the parameters in *Users* object to your username, password, and name. If you want more column go to **line 13** and add more in there.  **IMPORTANT NOTE:** The name you save here in name column in this database **MUST MATCH** the name that you train your face recognition model in steps above (put in simple, must match the folder name inside *face_recognition/dataset*) unless the login mechanism is not going to work.
16. Run `app.py` and go to the port that it's displaying on your command line/terminal. Test everything there, the login mechanism, face recognition, and face liveness detection. It must work fine by now. *Note: Our web page has no return button to go back. So you must change the URL directly if you want to go back*
17. (Optional) If you're capable of writing some basic SQL query, we recommend opening `database.sqlite` file and take a look at *Users* table to verify whether your data is added correctly.
18. **IMPORTANT** After running `app.py` for the first time, go to those adding users to database section in **line 56** and so on that you add in step 15, and comment out all line that you added. The point of doing this is to not adding the same data to the database table which can cause error. (SQL term: the primary key which is username must be unique)
19. Congratulations! You've done everything :D

## Hope you find our project exciting and useful more or less :D
NB: The rest of code and files involve the Amatrol Analytic Process control machine at DKUT, Siemens Center, the AI is used to authenticate access to the machine. It's just a feature built on top of the process.
The feature is not limited to any machine and can have different use cases depending on your needs.

## What can be improved
- Collect more data in many light conditions and from different genders/ethnics to improve the model (it turned out light intensity and condition play a big role here)
- Connect this login web template to the real app.
- Implement more login mechanism
- Add admin profiles to create another layer of security.
- Better frontend UI/UX, am working on it

## Problems we've found
All login mechanism work properly, but sometimes OpenCV camera doesn't show up when calling a function from app.py or when logging in. But restarting app.py once or twice always solves the problem.

### Please remember to download both hardware and software changes when applying the communication settings on your PLC or else you'll encounter errors

## Amazing resources I have learned from
- https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
- https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv/
- https://www.youtube.com/watch?v=2Zz97NVbH0U&t=790s
- https://www.youtube.com/watch?v=4L_xAWDRs7w&list=PLZoTAELRMXVPBaLN3e-uoVRR9hlRFRfUc  
  
**image in the log in page:** https://recfaces.com/wp-content/uploads/2021/01/face_scan_adobe-830x553.jpg

# Thank you for reading til the end. YOU ARE AMAZING!!! :D
<img src="./static/photos/java-python.gif">
