# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:14:21 2020

@author: nathan barloy
"""

import cv2
import csv
import numpy as np
import tools
import time
import serial


ESCAPE = 27

bot = serial.Serial('COM3', 9600)

settings = {"fps":cv2.CAP_PROP_FPS,
            "exposure":cv2.CAP_PROP_EXPOSURE,
            "brightness":cv2.CAP_PROP_BRIGHTNESS,
            "gain":cv2.CAP_PROP_GAIN,
            "gamma":cv2.CAP_PROP_GAMMA}


answer_dict = {"none":'1', "scissors":'4', "paper":'3', "rock":'2'}

# hyperparameters
threshold_value = 220
kernel_size = 7
blur_size = 5
kernel = np.array([[0,0,0,1,0,0,0],
                   [0,0,1,1,1,0,0],
                   [0,1,1,1,1,1,0],
                   [1,1,1,1,1,1,1],
                   [0,1,1,1,1,1,0],
                   [0,0,1,1,1,0,0],
                   [0,0,0,1,0,0,0]], np.uint8)
"""
kernerl = np.array([[0,0,0,1,1,1,0,0,0],
                    [0,0,1,1,1,1,1,0,0],
                    [0,1,1,1,1,1,1,1,0],
                    [1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1],
                    [0,1,1,1,1,1,1,1,0],
                    [0,0,1,1,1,1,1,0,0],
                    [0,0,0,1,1,1,0,0,0]], np.uint8)"""

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


# records initialization
record_read = [0]*record_size
record_pretreat = [0]*record_size
record_binar = [0]*record_size
record_morph = [0]*record_size
record_caract = [0]*record_size
record_caract2 = [0]*record_size
record_pred = [0]*record_size
start_rec = time.perf_counter()

last_answer = "none"

# main loop
while True :
    #print(f"last loop : {1000*(time.perf_counter()-start_rec)}")
    #print('')
    start_rec = time.perf_counter()
    
    
    # read capture
    ts = time.perf_counter()
    ret, img = cap.read(cv2.CV_8UC1)
    if not ret :
        cap = cv2.VideoCapture(0)
        continue
    img = img[:,:,0]
    
    te = time.perf_counter()
    record_read.append(te-ts)
    record_read.pop(0)
    
    
    # show image
    #cv2.imshow("Camera Image", img)
    
    
    # image preprocessing
    ts = time.perf_counter()
    img = cv2.medianBlur(img, blur_size)
    #cv2.imshow("Backgroud substaction", img)
    
    te = time.perf_counter()
    record_pretreat.append(te-ts)
    record_pretreat.pop(0)
    
    
    
    # binarize image
    ts = time.perf_counter()
    ret, binar = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    te = time.perf_counter()
    record_binar.append(te-ts)
    record_binar.pop(0)
    
    
    # morphological transformation
    ts = time.perf_counter()
    binar = cv2.morphologyEx(binar, cv2.MORPH_CLOSE, kernel)
    """
    binar = cv2.morphologyEx(binar, cv2.MORPH_OPEN, kernel)
    """
    
    te = time.perf_counter()
    record_morph.append(te-ts)
    record_morph.pop(0)
    
    
    # show binary image
    #cv2.imshow("Binarized Image", binar)
    
    
    
    answer = "none"
    Ljudge=0
    blocks = []
    try :
        ts = time.perf_counter()
        # get the center and the main axis of the image
        center, theta = tools.getCenterAndAngle(binar, empty_treshold)
        te = time.perf_counter()
        record_caract.append(te-ts)
        record_caract.pop(0)
        
        if center is not None :
            ts = time.perf_counter()
            # max and min angles for Ltip and Lmin detection
            amin = theta-alpha
            amax = theta+alpha
            Ltip, Lmin = tools.findTipMin(binar, center, amin, amax)
            te = time.perf_counter()
            record_caract2.append(te-ts)
            record_caract2.pop(0)
            
            blocks = []
            ts = time.perf_counter()
            if Ltip/Lmin<beta :
                answer = 'rock'
            else :
                Ljudge = (Ltip+Lmin)/2
                blocks = tools.nbBlock(binar, Ljudge, np.pi+theta+gamma, 3*np.pi+theta-gamma, center)
                nb_blocks = len(blocks)
                answer = 'scissors' if nb_blocks<=2 else 'paper'
                
            te = time.perf_counter()
            record_pred.append(te-ts)
            record_pred.pop(0)    
            
        else :
            record_caract2.append(0)
            record_caract2.pop(0)
            record_pred.append(0)
            record_pred.pop(0) 
        
    except (np.linalg.LinAlgError, ZeroDivisionError):
         pass
    
    
    
    # show prediction
    #print(answer)
    if last_answer!=answer :
        bot.write(answer_dict[answer].encode())
        last_answer = answer
    
    tot_time = time.perf_counter()-start_rec
    #print(f"total time : {1000*tot_time}")
    

    
    notes = cv2.cvtColor(binar,cv2.COLOR_GRAY2BGR)
    if answer!='none' and center is not None :
        cv2.line(notes, (center[0]-15,center[1]), (center[0]+15, center[1]), (0,0,255), 2)
        cv2.line(notes, (center[0],center[1]-15), (center[0], center[1]+15), (0,0,255), 2)
        cv2.ellipse(notes, center, (int(Ltip),int(Ltip)), 0, amin*180/np.pi, amax*180/np.pi, (255,0,0), 3)
        cv2.ellipse(notes, center, (int(Lmin),int(Lmin)), 0, amin*180/np.pi, amax*180/np.pi, (0,255,0), 3)
        if answer!='rock' :
            cv2.ellipse(notes, center, (int(Ljudge),int(Ljudge)), 180+theta*180/np.pi, gamma*180/np.pi, 360-gamma*180/np.pi, (0,255,255), 3)
            for b in blocks :
                cv2.circle(notes, b, 10, (255,255,0), -1)
    
    cv2.imshow("Notes", notes)
    
    """
    # print performance
    print(f"get image : {1000*sum(record_read)/record_size}")
    print(f"pretreat : {1000*sum(record_pretreat)/record_size}")
    print(f"binarization : {1000*sum(record_binar)/record_size}")
    print(f"morphology : {1000*sum(record_morph)/record_size}")
    print(f"center, angle : {1000*sum(record_caract)/record_size}")
    print(f"Ltip, Lmin : {1000*sum(record_caract2)/record_size}")
    print(f"prediction : {1000*sum(record_pred)/record_size}")
    """
    
    
    # input gestion
    key = cv2.waitKey(3)
    if key==ESCAPE :
        break
    
    


# release video capture
cap.release()
cv2.destroyAllWindows()

