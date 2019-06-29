
# Introduction
Age and gender, two of the key facial attributes, play a very foundational role in social interactions, making age and gender estimation from a single face image an important task in intelligent applications, such as access control, human-computer interaction, law enforcement, marketing intelligence
and visual surveillance, etc.

# Requirement
pip install OpenCV-python

# Steps to Follow
1. Face detection with Haar cascades
2. Gender Recognition with CNN
3. Age Recognition with CNN

## 1. Face detection with Haar cascades
This is a part most of us at least have heard of. OpenCV/JavaCV provide direct methods to import Haar-cascades and use them to detect faces.

## 2. Gender Recognition with CNN
Gender recognition using OpenCV's fisherfaces implementation is quite popular and some of you may have tried or read about it also. But, in this example, I will be using a different approach to recognize gender. This method was introduced by two Israel researchers, Gil Levi and Tal Hassner in 2015. I have used the CNN models trained by them in this example. We are going to use the OpenCV’s dnn package which stands for “Deep Neural Networks”.

## 3. Age Recognition with CNN
This is almost similar to the gender detection part except that the corresponding prototxt file and the caffe model file are “deploy_agenet.prototxt” and “age_net.caffemodel”. Furthermore, the CNN’s output layer (probability layer) in this CNN consists of 8 values for 8 age classes (“0–2”, “4–6”, “8–13”, “15–20”, “25–32”, “38–43”, “48–53” and “60-”)

A caffe model has 2 associated files,

1 .prototxt — The definition of CNN goes in here. This file defines the layers in the neural network, each layer’s inputs, outputs and functionality.

2 .caffemodel — This contains the information of the trained neural network (trained model).

Download .prtotxt and .caffemodel from [Here](https://talhassner.github.io/home/publication/2015_CVPR).

Download haar cascade for face detection from [Here](https://talhassner.github.io/home/publication/2015_CVPR).

# Source Code
```
import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 480) #set width of the frame
video_capture.set(4, 640) #set height of the frame

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']


def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return(age_net, gender_net)

def video_detector(age_net ,  gender_net):
    while True:
        _ , frame = video_capture.read()
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray , 1.3 ,5)
        for(x , y, w ,h) in faces:
            cv2.rectangle(frame , (x,y) , (x+w , y+h) , (255,255,255) , 2)
       
        #Get Face 
        face_img = frame[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        #Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        print("Gender : " + gender)
       
        #Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        print("Age Range: " + age)

        overlay_text = "%s %s" % (gender, age)
        cv2.putText(frame, overlay_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('video' , frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   
    video_capture.realse()
    cv2.destroyWindow()

if __name__ == "__main__":
    age_net, gender_net = load_caffe_models()
video_detector(age_net, gender_net)
```