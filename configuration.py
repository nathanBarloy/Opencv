# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:34:47 2020

@author: nathan barloy
"""

import cv2
import csv


ESCAPE = 27
ENTER = 13


############################
#
# format of the settings
#
# property_name:(prop_id, min_value, max_value, step, key+, key-)
#
############################

settings = {"fps":(cv2.CAP_PROP_FPS, 10, 120, 10, ord('a'), ord('q')),
            "exposure":(cv2.CAP_PROP_EXPOSURE, -12, -5, 1, ord('z'), ord('s')),
            "brightness":(cv2.CAP_PROP_BRIGHTNESS, 0, 120, 10, ord('e'), ord('d')),
            "gain":(cv2.CAP_PROP_GAIN, 0, 100, 10, ord('r'), ord('f')),
            "gamma":(cv2.CAP_PROP_GAMMA, 0, 200, 10, ord('t'), ord('g'))}




cap = cv2.VideoCapture(0)

properties = {}

# get the current properties of the camera
with open("properties.csv", 'r') as file :
    reader = csv.reader(file, delimiter=';')
    reader.__next__()
    for row in reader :
        properties[row[0]] = int(row[1])

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
for k in settings.keys() :
    cap.set(settings[k][0], properties[k])



def change_prop(prop_name, up) :
    global properties
    if up :
        properties[prop_name] = min(properties[prop_name]+settings[prop_name][3], settings[prop_name][2])
    else :
        properties[prop_name] = max(properties[prop_name]-settings[prop_name][3], settings[prop_name][1])
    cap.set(settings[prop_name][0], properties[prop_name])
    print(f"new {prop_name} value : {properties[prop_name]}")


save = False

while True:
    
    ret, frame = cap.read()
    
    if not ret :
        continue
    
    
    cv2.imshow("camera", frame)
    
    key = cv2.waitKey(5)
    if key==ESCAPE :
        break
    if key==ENTER :
        save = True
        break
    
    if key>=0 :
        for k,v in settings.items() :
            if key==v[4] :
                change_prop(k,True)
                break
            if key==v[5] :
                change_prop(k,False)
                break



if save :
    # set the wanted properties of the camera
    with open("properties.csv", 'w', newline='') as file :
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["property", "value"])
        for elem in properties.items() :
            writer.writerow(elem)

    

cap.release()
cv2.destroyAllWindows()
