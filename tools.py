# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:39:34 2020

@author: Nathan Barloy
"""

from typing import List, Tuple
import math

import numpy as np
import cv2


def generate_arc(R:float, amin:float, amax:float) -> List[Tuple[int,int]] :
    """
    Generates a list of pixels that represent a circular arc.
    
    Parameters
    ----------
    R : float
        Radius of the arc.
    amin : float
        Start angle for the arc.
    amax : float
        End angle for the arc

    Returns
    -------
    List[Tuple[int,int]] 
        List of pixels.

    """
    
    angle = amin
    res = [(int(R*np.cos(angle)), int(R*np.sin(angle)))]
    da = 1/R
    while angle <= amax :
        x = int(R*np.cos(angle))
        y =int( R*np.sin(angle))
        if (x,y)!=res[-1] :
            res.append((x,y))
        angle += da
    return res




def getCenterAndAngle(src:np.ndarray, empty_threshold:float) -> Tuple[Tuple[int,int], float] :
    """
    Get the center and the angle of the main axis of a binary image.
    None if the image is considered empty.

    Parameters
    ----------
    src : np.ndarray
        The binary image to be analysed.
    empty_threshold : float
        The maximum proportion of white pixels for the image to be considered empty.

    Returns
    -------
    Tuple[Tuple[int,int], float] 
        The center pixel and the angle of the main axis.

    """
    
    y, x = np.nonzero(src)
    if len(x)<empty_threshold*len(src)*len(src[0]) :
        return None, None
    mx, my = np.mean(x), np.mean(y)
    x = x - mx
    y = y - my
    coords = np.vstack([x, y])
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    theta = math.atan2(y_v1, x_v1)
    return ((int(mx), int(my)), theta)
    



def isBlack(src:np.ndarray, R:float, amin:float, amax:float, center:Tuple[int,int]) -> bool :
    """
    Check wether the pixels on the circular arc defined by the center 'center',
    the radius 'R' and the angles 'amin' and 'amax' are all black.

    Parameters
    ----------
    src : np.ndarray
        The binary image to be analysed.
    R : float
        The radius of the circular arc.
    amin : float
        The starting angle of the circular arc.
    amax : float
        The ending angle of the circular arc.
    center : Tuple[int,int]
        The center of the object in the image.

    Returns
    -------
    bool 
        The result of the analysis.

    """
    
    intmx, intmy = center
    try :
        for coord in generate_arc2(R, amin, amax) :
            if src[intmy+coord[1], intmx+coord[0]]!=0 :
                return False
        return True
    except IndexError :
        return True




def isWhite(src:np.ndarray, R:float, amin:float, amax:float, center:Tuple[int,int]) -> bool :
    """
    Check wether the pixels on the circular arc defined by the center 'center', the radius 'R' and 
    the angles 'amin' and 'amax' are all white. It can begin and end with a black block, but there 
    can't be a black pixel in the middle of white pixels.

    Parameters
    ----------
    src : np.ndarray
        The binary image to be analysed.
    R : float
        The radius of the circular arc.
    amin : float
        The starting angle of the circular arc.
    amax : float
        The ending angle of the circular arc.
    center : Tuple[int,int]
        The center of the object in the image.

    Returns
    -------
    bool 
        The result of the analysis.

    """
    
    intmx, intmy = center
    prec = 0
    seen = False
    for coord in generate_arc2(R, amin, amax) :
        try :
            new = src[intmy+coord[1], intmx+coord[0]]
        except IndexError:
            continue
        if prec==0 and new!=0 :
            if not seen :
                seen = True
            else :
                return False
        prec = new
    return True




def nbBlock(src:np.ndarray, R:float, amin:float, amax:float, center:Tuple[int,int]) \
            -> List[Tuple[int,int]] :
    """
    get the first pixel of the white blocks on the circular arc defined by the center 'center',
    the radius 'R' and the angles 'amin' and 'amax'.

    Parameters
    ----------
    src : np.ndarray
        The binary image to be analysed.
    R : float
        The radius of the circular arc.
    amin : float
        The starting angle of the circular arc.
    amax : float
        The ending angle of the circular arc.
    center : Tuple[int,int]
        The center of the object in the image.

    Returns
    -------
    int 
        The numumber of white blocks.

    """
    intmx, intmy = center
    prec = 0
    res = []
    for coord in generate_arc2(R, amin, amax) :
        try :
            new =src[intmy+coord[1], intmx+coord[0]]
        except IndexError :
            new = 0
        if prec==0 and new!=0 :
            res.append((intmx+coord[0], intmy+coord[1]))
        prec = new
    return res




def findTipMin(src:np.ndarray, center:Tuple[int,int], amin:float, amax:float) \
                -> Tuple[float, float] :
    """
    Find two things :
        - The distance between the center of the object, and the tip of the fingers,
            only checking between 'amin' and 'amax'.
        - The distance between the center of the object, and the base of the fingers,
            only checking between 'amin' and 'amax'.

    Parameters
    ----------
    src : np.ndarray
        The binary image to be analysed.
    center : Tuple[int,int]
        The center of the object in the image.
    amin, amax : float
        The borns of the range where we are checking.

    Returns
    -------
    Tuple[float, float] 
        The Ltip and the Lmin

    """
    #find the tip
    Rmax = max(len(src), len(src[0]))
    Rmin = 0
    while Rmax-Rmin>1 :
        Rmed = (Rmax+Rmin)/2
        if isBlack(src, Rmed, amin, amax, center) :
            Rmax = Rmed
        else :
            Rmin = Rmed
    Ltip = Rmax
    
    #find the min
    Rmax = Ltip
    Rmin = 0
    while Rmax-Rmin>1 :
        Rmed = (Rmax+Rmin)/2
        if isWhite(src, Rmed, amin, amax, center) :
            Rmin = Rmed
        else :
            Rmax = Rmed
    Lmin = Rmin
    
    return Ltip, Lmin




def generate_circle(R:int) -> List[Tuple[int,int]] :
    """
    Build the pixels of a circle of radius 'R'

    Parameters
    ----------
    R : int
        radius of the wanted circle.

    Returns
    -------
    List[Tuple[int,int]] 
        List of pixels.

    """
    
    octantNE = []
    octantEN = []
    octantNO = []
    octantON = []
    octantSE = []
    octantES = []
    octantSO = []
    octantOS = []
    x = 0
    y = R
    m = 5-4*R
    while x<=y :
        octantNE.insert(0,(x,y))
        octantEN.append((y,x))
        octantNO.append((-x,y))
        octantON.insert(0,(-y,x))
        octantES.insert(0,(y,-x))
        octantSE.append((x,-y))
        octantSO.insert(0,(-x,-y))
        octantOS.append((-y, -x))
        if m>0 :
            y -= 1
            m -= 8*y
        x += 1
        m += 8*x + 4
    res = octantOS
    if res[-1]==octantSO[0] :
        res.pop()
    res += octantSO
    if res[-1]==octantSE[0] :
        res.pop()
    res += octantSE
    if res[-1]==octantES[0] :
        res.pop()
    res += octantES
    if res[-1]==octantEN[0] :
        res.pop()
    res += octantEN
    if res[-1]==octantNE[0] :
        res.pop()
    res += octantNE
    if res[-1]==octantNO[0] :
        res.pop()
    res += octantNO
    if res[-1]==octantON[0] :
        res.pop()
    res += octantON
    if res[0]==res[-1] :
        res.pop()
    return res




def generate_arc2(R:float, amin:float, amax:float) -> List[Tuple[int,int]] :
    """
    Generates a list of pixels that represent a circular arc.
    
    Parameters
    ----------
    R : float
        Radius of the arc.
    amin : float
        Start angle for the arc.
    amax : float
        End angle for the arc.

    Returns
    -------
    List[Tuple[int,int]] 
        List of pixels.

    """
    
    pmin = (amin/(2*np.pi) + 0.5)%1
    pmax = (amax/(2*np.pi) + 0.5)%1

    circle  = generate_circle(int(R))
    n = len(circle)
    
    if pmin < pmax :
        return circle[int(n*pmin):int(n*pmax)]
    else :
        return circle[int(n*pmin):]+circle[:int(n*pmax)]
      