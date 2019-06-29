
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