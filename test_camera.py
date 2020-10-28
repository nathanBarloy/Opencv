# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:16:57 2020

@author: SAUL-GGM11
"""



#%% Récupération des caractéristiques de la caméra
import cv2

name_list = ["position (ms)", "index frame", "relative position", "width", "height", "fps", "fourcc", "frame count", "format of Mat", "capture mode", "brightness",
             "contrast", "saturation", "hue", "gain", "exposure", "convert RGB", "white balance blue", "rectification", "monochrome", "sharpness", "auto-exposure",
             "gamma", "temperature", "trigger", "trigger delay", "xhite balance red", "zoom", "focus", "guid", "iso speed", "?????", "backlight", "pan", "tilt", "roll",
             "iris", "settings", "buffersize", "auto-focus", "sar num", "sar den", "backend", "channel", "auto-wb", "wb temperature", "codec pixel format", "bitrate",
             "orientation meta", "orientation auto"]

cap = cv2.VideoCapture(0)

for i in range(len(name_list)) :
    val = cap.get(i)
    if val==0 : val="None"
    if val==-1 : val="Neg"
    print(i, name_list[i], ":", val)
    
print(cap.getBackendName())

"""
La caméra ASI178MM utilise le backend DirectShow, et possède les propriété suivantes (avec les valeurs par défaut) :
    width : 640
    height : 480
    brightness : 10
    gain : 256 -> garde la valeur précedemment utilisée (même sur un autre logiciel)
    exposure : -10 (?) -> garde la valeur précédente
    gamma : 50
    fps : 0 -> prend une valeur par défaut qui dépend de la taille de la fenêtre
"""



#%% boucle d'affichage

sizeValues = [(640,480), (1024,768), (1280,720)]
brightnessValues = [1,2,10,20,50,100,150,200,250,300,400,500]
gainValues = [0,5,20,50,100,170,256,350,400]
gammaValues = [0,1,10,25,50, 75, 100]
exposureValues = [-6,-7,-8,-9,-10,-11,-12]
fpsValues = [2,5,10,20,30,45,60,80,100,120,150,200]




while True :
    ret, frame = cap.read()
    if not ret :
        cap = cv2.VideoCapture(0)
        continue
    
    cv2.imshow('img',frame)
    
    key = cv2.waitKey(1)
    if key > 0 :
        
        if key==ord('q') :
            break
        
        if key==ord('a') :
            try :
                i = sizeValues.index((cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                i+=1
                if i==len(sizeValues) : i=0
            except ValueError :
                i=0
            print(f"Changing width to {sizeValues[i][0]}.", end=' ')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, sizeValues[i][0])
            print(f"New width : {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            print(f"Changing height to {sizeValues[i][1]}.", end=' ')
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, sizeValues[i][1])
            print(f"New height : {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
            
        if key==ord('z') :
            try :
                i = brightnessValues.index(cap.get(cv2.CAP_PROP_BRIGHTNESS))
                i+=1
                if i==len(brightnessValues) : i=0
            except ValueError :
                i=0
            print(f"Changing brightness to {brightnessValues[i]}.", end=' ')
            cap.set(cv2.CAP_PROP_BRIGHTNESS, brightnessValues[i])
            print(f"New brightness : {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
            
        if key==ord('e') :
            try :
                i = gainValues.index(cap.get(cv2.CAP_PROP_GAIN))
                i+=1
                if i==len(gainValues) : i=0
            except ValueError :
                i=0
            print(f"Changing gain to {gainValues[i]}.", end=' ')
            cap.set(cv2.CAP_PROP_GAIN, gainValues[i])
            print(f"New gain : {cap.get(cv2.CAP_PROP_GAIN)}")
            
        if key==ord('r') :
            try :
                i = gammaValues.index(cap.get(cv2.CAP_PROP_GAMMA))
                i+=1
                if i==len(gammaValues) : i=0
            except ValueError :
                i=0
            print(f"Changing gamma to {gammaValues[i]}.", end=' ')
            cap.set(cv2.CAP_PROP_GAMMA, gammaValues[i])
            print(f"New gamma : {cap.get(cv2.CAP_PROP_GAMMA)}")
            
        if key==ord('t') :
            try :
                i = exposureValues.index(cap.get(cv2.CAP_PROP_EXPOSURE))
                i+=1
                if i==len(exposureValues) : i=0
            except ValueError :
                i=0
            print(f"Changing exposure to {exposureValues[i]}.", end=' ')
            cap.set(cv2.CAP_PROP_EXPOSURE, exposureValues[i])
            print(f"New exposure : {cap.get(cv2.CAP_PROP_EXPOSURE)}")
        
        if key==ord('y') :
            try :
                i = fpsValues.index(round(cap.get(cv2.CAP_PROP_FPS)))
                i+=1
                if i==len(fpsValues) : i=0
            except ValueError :
                i=0
            print(f"Changing fps to {fpsValues[i]}.", end=' ')
            cap.set(cv2.CAP_PROP_FPS, fpsValues[i])
            print(f"New fps : {cap.get(cv2.CAP_PROP_FPS)}")

    
cap.release()