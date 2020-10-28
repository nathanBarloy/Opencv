# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:58:07 2020

@author: SAUL-GGM11
"""
import tools
import numpy as np
import cv2

img = cv2.imread('image4.jpg', cv2.IMREAD_GRAYSCALE)
back = cv2.imread('back.jpg', cv2.IMREAD_GRAYSCALE)

adiff = cv2.absdiff(img, back)

_,thr = cv2.threshold(adiff, 25, 255, cv2.THRESH_BINARY)


(x,y), theta = tools.getCenterAndAngle(thr)
print(x,y)

d = 10
cv2.line(thr, (x-d,y), (x+d,y), (0,0,255), 2)
cv2.line(thr, (x,y-d), (x,y+d), (0,0,255), 2)

x2 = x + 5*d*np.cos(theta)
y2 = y + 5*d*np.sin(theta)
x2 = int(x2)
y2 = int(y2)

cv2.line(thr, (x,y), (x2,y2), (0,0,255), 3)
an = np.pi/8

Ltip, Lmin = tools.findTipMin(thr, (x,y), theta-an, theta+an)
print(Ltip, Lmin)

cv2.imshow('lol', thr)
cv2.waitKey(0)

cv2.destroyAllWindows()