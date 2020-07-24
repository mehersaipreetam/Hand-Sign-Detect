#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:01:01 2020

@author: meher
"""

from keras.models import load_model

model = load_model('sign.h5')

import cv2
import numpy as np
import keras
from keras_applications import mobilenet
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

capture = cv2.VideoCapture(0)

numbers = {
        0 : 'Zero',
        1 : 'One',
        2 : 'Two',
        3 : 'Three',
        4 : 'Four',
        5 : 'Five',
        6 : 'Six',
        7 : 'Seven',
        8 : 'Eight',
        9 : 'Nine'
    }


while True:
    ret, frame = capture.read()

    key = cv2.waitKey(1)
    
    w,h = 224,224
    x,y = 50,50
    
    rect = frame[y : y+h , x : x+w]
    
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    
    lower = np.array([0, 10, 60], dtype = "uint8") 
    upper = np.array([20, 150, 255], dtype = "uint8")

    # cv2.imshow('Frame',mask)
    # #back = np.zeros((224,224))

    img_array = rect
    img_array = img_array.reshape(224,224,3)
    img_array_expanded = np.expand_dims(img_array, axis =0)
    rect = keras.applications.mobilenet.preprocess_input(img_array_expanded)
    
    frame_w_rect = cv2.rectangle(frame,(x,y),(x+w,y+h),(100,0,0),2)
    
    y_pred = model.predict(rect, verbose = 0)
    y = y_pred.argmax(axis = 1)
    
    text = numbers[int(y)]
    label_pos = (150,240)
    cv2.putText(frame_w_rect, text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

    cv2.imshow('Frame',frame_w_rect)

    if key == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()