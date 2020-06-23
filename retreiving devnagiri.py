# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 05:29:53 2020

@author: Debatosh
"""

import cv2 as cv
from keras.models import load_model
import numpy as np
from collections import deque


model1 = load_model('devnagiri.h5')
print(model1)
def keras_process_image(img):
    image_x = 32
    image_y = 32
    img = cv.resize(img, (image_x, image_y))
    img = np.array(img, dtype = np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


def keras_predict(model, image):
    processed = keras_process_image(image)
    #print(str(processed.shape))
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


cap = cv.VideoCapture(0)
lower_blue = np.array([56,45,45])
upper_blue = np.array([86,255,255])
'''lower_blue = (35,21,62)
upper_blue = (90,255,255)'''
pred_clas = 0
pts = deque(maxlen = 512)
bl_bd = np.zeros((480,640,3), dtype = np.uint8)
digit = np.zeros((200,200,3), dtype = np.uint8)
while(1):
    _, frame = cap.read()
    frame = cv.flip(frame,1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    blur = cv.medianBlur(mask,9)
    blur = cv.GaussianBlur(blur, (5,5), 0)
    thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,	cv.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    #pts = deque(maxlen = 512)
    if len(cnts) > 0:
       # print(len(cnts)
        contour = max(cnts, key=cv.contourArea)
        if cv.contourArea(contour) > 1500:
            ((x, y), radius) = cv.minEnclosingCircle(contour)
            cv.circle(frame, (int(x), int(y)), int(radius),	(0, 255, 255), 2)
            cv.circle(frame, center, 5, (0, 0, 255), -1)
            M = cv.moments(contour)
            #print(M)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            pts.appendleft(center)
            ##print(len(pts))
            for i in range(1, len(pts)):
                print(len(pts))
                if pts[i - 1] is None or pts[i] is None:
                    continue
                
                cv.line(bl_bd, pts[i - 1], pts[i], (255, 255, 255), 10)
                cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), 5)
                
        else:
            print("zero")            
            pts = deque(maxlen = 512)
            bl_bd = np.zeros((480,640,3), dtype= np.uint8)
            pred_prob, pred_class = keras_predict(model1, digit)
            print(pred_class, pred_prob)
            cv.putText(frame, "conv Network: " + str(pred_class), (10,470), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    elif len(cnts) == 0:
        if(len(pts) != []):
            bl_bd_gray = cv.cvtColor(bl_bd, cv.COLOR_BGR2GRAY)
            blur1 = cv.medianBlur(bl_bd_gray, 15)
            blur1 = cv.GaussianBlur(blur1, (5,5), 0)
            thresh1 = cv.threshold(blur1,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
            bl_bd_cnts = cv.findContours(thresh1.copy(), cv.RETR_EXTERNAL,	cv.CHAIN_APPROX_SIMPLE)[-2]
            if len(bl_bd_cnts) >= 1:
                cnt = max(cnts, key = cv.contourArea)
                print(cv.contourArea(cnt))
                if cv.contourArea(cnt) > 2000:
                    x,y,w,h = cv.boundingRect(cnt)
                    digit = bl_bd_gray[y:y+h, x:x+w]
                    pred_prob, pred_class = keras_predict(model1, digit)
                    print(pred_class, pred_prob)
        print("zero")            
        pts = deque(maxlen = 512)
        bl_bd = np.zeros((480,640,3), dtype= np.uint8)
    #cv.putText(frame, "conv Network: " + pred_class, (10,470), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    #cv.putText(frame, "conv Network: " + str(5), (10,470), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv.imshow("Frame", frame)
    cv.imshow("Contours", thresh)
    cv.imshow("Mask", mask)
    k = cv.waitKey(1) & 0xFF
    if k == ord("q"):
        break
        
cap.release()
cv.destroyAllWindows()