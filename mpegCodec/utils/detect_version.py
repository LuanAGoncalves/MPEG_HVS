# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 07:43:46 2015

@author: uluyac
"""

def opencv_version(lib=None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        import cv2 as lib
        
    version = lib.__version__.split(".")
        
    # return whether or not the current OpenCV version matches the
    # major version number
    return int(version[0])
