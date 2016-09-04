# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:59:12 2016

@author: luan
"""

from mpegCodec import codec
from mpegCodec.utils import frame_utils as futils
import matplotlib.pylab as plt
from Tkinter import Tk
from tkFileDialog import askopenfilename
import sys
import numpy as np
import os

root = Tk()
root.withdraw()

if __name__ == "__main__":
    fileName = askopenfilename(parent=root, title="Enter with a file name.").__str__()
    fileName2 = fileName
    videoName = fileName
    if fileName == '': 
        sys.exit('Filename empty!')
    print("\nFile: " + fileName)
    name = fileName.split('/')
    print name
    name = name[-1]
    print name
    name = name.split('.')[-1]
    print name
    
    # In order to run the encoder just enter with a video file.
    # In order to run the decoder just enter with a output file (in the output directory).
    
    quality = range (0,110,10)    # Compression quality.
    sspace = 15    # Search space.
    search = 1        # 0 - Full search; 1 - Parallel hierarchical.
    flat = 10.0    # Qflat value.
    p_value = 2**np.arange(-5.,5.5,0.5)        # Parameter p.
    mode = '420'       # 444 or 420
    hvsqm = [0, 1]       # Normal or HVS based method
    kbps = np.zeros((len(p_value)+1,len(quality)))
    mssim = np.zeros((len(p_value)+1,len(quality)))
#    mpeg_avgBits, mpeg_CRate, mpeg_redundancy, mpeg_kbps = [], [], [], []
#    mpeg_hvs_avgBits, mpeg_hvs_CRate, mpeg_hvs_redundancy, mpeg_hvs_kbps = [], [], [], []
#    seq, meanPsnrValues, meanMssimValues, avgBits = [], [], [], []
#    seq_hvs, meanPsnrValues_hvs, meanMssimValues_hvs, avgBits_hvs = [], [], [], []
    
    
    if name == 'mp4' or name == 'MP4' or name == 'mpg'or name == 'avi' or name == 'AVI':
        count = 0
        for i in quality:
            mpeg = codec.Encoder(fileName2, i, sspace, mode, search, 0, flat, 2)
            mpeg.run()
            kbps[0,count] = np.mean(mpeg.kbps)
            mpeg = codec.Decoder(mpeg.output_name, videoName)
            a, b, c, d = mpeg.run()
            mssim[0,count] = np.mean(np.array(c))
            count2 = 1
            for p in p_value:
                mpeg = codec.Encoder(fileName2, i, sspace, mode, search, 1, flat, p)
                mpeg.run()
                kbps[count2,count] = np.mean(mpeg.kbps)
                mpeg = codec.Decoder(mpeg.output_name, videoName)
                a, b, c, d = mpeg.run()
                mssim[count2,count] = np.mean(np.array(c))
                count2 += 1
            count += 1
        
        fig2 = plt.figure()
        ax1 = fig2.add_subplot(11)
        r, c = mssim.shape
        count = 0
        for i in range (r):
            if i == 0:
                ax1.plot(kbps[i], mssim[i], label="Standard")
            else:
                ax1.plot(kbps[i], mssim[i], label="p = %f" % (p_value[count]))
                count += 1
        legend1 = ax1.legend(loc='best', shadow=True)
        ax1.set_xlabel("Taxa (kbps)")
        ax1.set_ylabel("mssim")
        ax1.set_title("MSSIM")
        ax1.grid()
                