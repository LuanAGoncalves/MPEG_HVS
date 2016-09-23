# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:41:38 2015

@author: uluyac
"""

import cv2
import os

from Tkinter import Tk
from tkSimpleDialog import askstring

def sequence_iterator(sequence):
    
    nFrames = len(sequence)
    wName = 'Video'
    cv2.namedWindow(wName)
    
    print '\n### Sequence Iterator ###'
    print 'Number of frames: ' + str(nFrames)
    for i in range(nFrames):
        cv2.imshow('Video', sequence[i])
        cv2.waitKey(0)
        
    cv2.destroyAllWindows()
    
    
def write_sequence_frames(sequence, mode, hvsqm, quality, video_name = ''):
    
    root = Tk()
    root.withdraw()
    
    nFrames = len(sequence)
    wName = 'Video'
    cv2.namedWindow(wName)
    
    dirName = ''
    if video_name == '':
        dirName = askstring("Directory Name", "Enter with the directory output name").__str__()
    else:
        dirName = video_name.split('/')
        dirName = dirName[-1]
        dirName = dirName.split('.')[0]
        hvs = dirName.split('_')[-1]
        
#    dirName = ''.join(e for e in dirName if e.isalnum())
    if not os.path.exists('./frames_output/'):
        os.makedirs('./frames_output/')
    if hvs == 'hvs':
        if mode == '444':
            directory = './frames_output/hvs/444/' + '%d'%quality  + '/' + dirName + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)
        elif mode == '420':
            directory = './frames_output/hvs/420/'+ '%d'%quality  + '/' + dirName + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)
    else:
        if mode == '444':
            directory = './frames_output/normal/444/'+ '%d'%quality  + '/' + dirName + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)
        elif mode == '420':
            directory = './frames_output/normal/420/'+ '%d'%quality  + '/' + dirName + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)
    extension = '.png'
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    print '\n### Writing frames ###'
    print 'Number of frames: ' + str(nFrames)
    for i in range(nFrames):
        imName = directory + str(i) + extension
        print('Saving ' + imName)
        cv2.imwrite(imName, sequence[i])