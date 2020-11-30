# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:14:21 2020

@author: nathan barloy
"""
import cv2
import csv
import numpy as np
import serial

import tools


ESCAPE = 27

# Get the communication with the robot
# Warning ! Change 'COM3' according to where the bot is connected on the computer
# (you can find this in the device manager of the computer)
bot = serial.Serial('COM3', 9600)

# dictionnary for the settings of the camera
settings = {"fps":cv2.CAP_PROP_FPS,
            "exposure":cv2.CAP_PROP_EXPOSURE,
            "brightness":cv2.CAP_PROP_BRIGHTNESS,
            "gain":cv2.CAP_PROP_GAIN,
            "gamma":cv2.CAP_PROP_GAMMA}

answer_dict = {"none":'1', "scissors":'4', "paper":'3', "rock":'2'}


# hyperparameters
threshold_value = 220
blur_size = 5
kernel = np.array([[0,0,0,1,0,0,0],
                   [0,0,1,1,1,0,0],
                   [0,1,1,1,1,1,0],
                   [1,1,1,1,1,1,1],
                   [0,1,1,1,1,1,0],
                   [0,0,1,1,1,0,0],
                   [0,0,0,1,0,0,0]], np.uint8)


alpha = np.pi/7
beta = 1.7
gamma = np.pi/3
record_size = 20
empty_treshold = 0.02



# get video capture
cap = cv2.VideoCapture(0)

# setup of the camera
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
with open("properties.csv", 'r') as file :
    reader = csv.reader(file, delimiter=';')
    reader.__next__()
    for row in reader :
        cap.set(settings[row[0]], int(row[1]))
        
"""
# setup of the window        
cv2.namedWindow("Computer vision", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Computer vision",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
"""


last_answer = "none"

# main loop
while True :
    
    # read capture
    ret, img = cap.read(cv2.CV_8UC1)
    if not ret :
        cap = cv2.VideoCapture(0)
        continue
    img = img[:,:,0]
    
    
    # image blur, to remove potential noise
    img = cv2.medianBlur(img, blur_size)
    
    # binarize image
    ret, binar = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # morphological closing, to close potential gaps in the image
    binar = cv2.morphologyEx(binar, cv2.MORPH_CLOSE, kernel)
    
    # recognition of the sign in the image
    answer = "none"
    Ljudge=0
    blocks = []
    try :
        # get the center and the main axis of the image
        center, theta = tools.getCenterAndAngle(binar, empty_treshold)
        
        if center is not None :
            # max and min angles for Ltip and Lmin detection
            amin = theta-alpha
            amax = theta+alpha
            Ltip, Lmin = tools.findTipMin(binar, center, amin, amax)
            
            blocks = []
            if Ltip/Lmin<beta :
                answer = 'rock'
            else :
                Ljudge = (Ltip+Lmin)/2
                blocks = tools.nbBlock(binar, Ljudge, np.pi+theta+gamma, 3*np.pi+theta-gamma, center)
                nb_blocks = len(blocks)
                answer = 'scissors' if nb_blocks<=3 else 'paper'
        
    except (np.linalg.LinAlgError, ZeroDivisionError):
         pass
    
    
    
    # show prediction
    if last_answer!=answer :
        bot.write(answer_dict[answer].encode())
        print(f"sent {answer} to the bot")
        last_answer = answer
    

    # show computer vision
    computer_vision = cv2.cvtColor(binar,cv2.COLOR_GRAY2BGR)
    if answer!='none' and center is not None :
        cv2.line(computer_vision, (center[0]-15,center[1]), (center[0]+15, center[1]), (0,0,255), 2)
        cv2.line(computer_vision, (center[0],center[1]-15), (center[0], center[1]+15), (0,0,255), 2)
        cv2.ellipse(computer_vision, center, (int(Ltip),int(Ltip)), 0, amin*180/np.pi, amax*180/np.pi, (255,0,0), 3)
        cv2.ellipse(computer_vision, center, (int(Lmin),int(Lmin)), 0, amin*180/np.pi, amax*180/np.pi, (0,255,0), 3)
        if answer!='rock' :
            cv2.ellipse(computer_vision, center, (int(Ljudge),int(Ljudge)), 180+theta*180/np.pi, gamma*180/np.pi, 360-gamma*180/np.pi, (0,255,255), 3)
            for b in blocks :
                cv2.circle(computer_vision, b, 10, (255,255,0), -1)
    cv2.imshow("Computer vision", computer_vision)

    
    # input gestion
    key = cv2.waitKey(3)
    if key==ESCAPE :
        break
    
    


# release video capture
cap.release()
cv2.destroyAllWindows()

