# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:04:07 2015

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
    flat = 16.0    # Qflat value.
    p = 1.0        # Parameter p.
    mode = '420'       # 444 or 420
    hvsqm = [0, 1]       # Normal or HVS based method
    mpeg_avgBits, mpeg_CRate, mpeg_redundancy, mpeg_kbps = [], [], [], []
    mpeg_hvs_avgBits, mpeg_hvs_CRate, mpeg_hvs_redundancy, mpeg_hvs_kbps = [], [], [], []
    seq, meanPsnrValues, meanMssimValues, avgBits = [], [], [], []
    seq_hvs, meanPsnrValues_hvs, meanMssimValues_hvs, avgBits_hvs = [], [], [], []
    
    
    if name == 'mp4' or name == 'MP4' or name == 'mpg'or name == 'avi' or name == 'AVI':
        for i in quality:
            mpeg = codec.Encoder(fileName2, i, sspace, mode, search, hvsqm[0], flat, p)
            mpeg.run()
            mpeg_avgBits.append(np.mean(mpeg.avgBits))
            mpeg_CRate.append(np.mean(mpeg.CRate))
            mpeg_redundancy.append(np.mean(mpeg.redundancy))
            mpeg_kbps.append(np.mean(mpeg.kbps))
            mpeg_hvs = codec.Encoder(fileName2, i, sspace, mode, search, hvsqm[1], flat, p)
            mpeg_hvs.run()
            mpeg_hvs_avgBits.append(np.mean(mpeg_hvs.avgBits))
            mpeg_hvs_CRate.append(np.mean(mpeg_hvs.CRate))
            mpeg_hvs_redundancy.append(np.mean(mpeg_hvs.redundancy))
            mpeg_hvs_kbps.append(np.mean(mpeg_hvs.kbps))
            fileName = [mpeg.output_name, mpeg_hvs.output_name]
            
            path = fileName[0].split('/')
            directory = './outputs/resultados/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            figurename = directory + path[-1].split('.')[0] + '_vs_hvs' +'.png'
            print "Figure name: " + figurename
            mpeg = codec.Decoder(fileName[0], videoName)
            seq = 0
            a, b, c, d = mpeg.run()
            seq = a
            PsnrValues = b
            meanPsnrValues.append(np.mean(np.array(b)))
            MssimValues = c
            meanMssimValues.append(np.mean(np.array(c)))
            avgBits += d
            futils.write_sequence_frames(seq, mpeg.mode, mpeg.hvsqm, i, fileName[0])
            path = fileName[1].split('/')
            mpeg_hvs = codec.Decoder(fileName[1], videoName)
            seq_hvs = 0
            a, b, c, d = mpeg_hvs.run()
            seq_hvs = a
            PsnrValues_hvs = b
            meanPsnrValues_hvs.append(np.mean(np.array(b)))
            MssimValues_hvs =c
            meanMssimValues_hvs.append(np.mean(np.array(c)))
            avgBits_hvs += d
            futils.write_sequence_frames(seq_hvs, mpeg_hvs.mode, mpeg_hvs.hvsqm, i, fileName[1])
        fig2 = plt.figure()
        ax1 = fig2.add_subplot(211)
        ax1.plot(mpeg_kbps, meanPsnrValues, '-*', color = "blue",label="MPEG-1")
        ax1.plot(mpeg_hvs_kbps, meanPsnrValues_hvs, '-o', color="red",label="MPEG-HVS")
        legend1 = ax1.legend(loc='best', shadow=True)
        ax1.set_xlabel("Taxa (kbps)")
        ax1.set_ylabel("psnr")
        ax1.set_title("PSNR")
        ax1.grid()
        
        ax2 = fig2.add_subplot(212)
        ax2.plot(mpeg_kbps,meanMssimValues, '-*', color="blue", label="MPEG-1")
        ax2.plot(mpeg_hvs_kbps,meanMssimValues_hvs, '-o', color="red", label="MPEG-HVS")
        legend2 = ax2.legend(loc='best', shadow=True)
        ax2.set_xlabel("Taxa (kbps)")
        ax2.set_ylabel("mssim")
        ax2.set_title("MSSIM")
        ax2.grid()
        plt.subplots_adjust(hspace=0.4)
        plt.savefig(figurename, format='png')
        
##        print "%s Measures %s" % (5*'#', 5*'#')
#        print "%s MPEG %d %s" % (5*'#', quality, 5*'#')
#        print "Average MSSIM = %f" % (msimMean)
#        print "Average PSNR = %f" % (psnrMean)
##        print "%s Measures %s" % (5*'#', 5*'#')
#        print "%s MPEG HVS %s" % (5*'#', 5*'#')
#        print "Average MSSIM = %f" % (msimMean_hvs)
#        print "Average PSNR = %f" % (psnrMean_hvs)
        
#                ### PLAYER ###
#        print "\n### [PLAYER] > ###\n Press\n - \'p\' to play\n - \'f\' to step-by-step\n - \'b\' to backstep\n - \'q\' to quit"
#        play=True
#        f = -1
#        fator = 0
#        press=False
#        videoname = path[-1].split('.')[0]
#        while play:
#            if f == -1 or f >= mpeg.nframes:
#                cv2.imshow(videoname,np.zeros((mpeg.shape)))
#            else:
#                cv2.imshow(videoname, seq[f])
#            k = cv2.waitKey(fator) & 0xFF
#                
#            if k==-1 or k == 255:
#                f += 1
#            elif k==ord('q'):
#                play=False
#                cv2.destroyAllWindows()
#            elif k==ord('p') or k==ord(' '):
#                if press == False:
#                    fator = int((1./mpeg.fps)*1000)
#                    press=True
#                elif press==True:
#                    fator = 0
#                    press = False   #            f += 1
#            elif k==ord('f'):
#                f += 1
#                fator = 0
#            elif k==ord('b'):
#                f -= 1
#                fator = 0
#            elif k==ord('r'):
#                fator = int((1./mpeg.fps)*1000)
#                press = False
#                f = -1
#            if f < -1:
#                f = -1
#                fator = 0
#            elif f >= mpeg.nframes:
#                f = mpeg.nframes
#                fator = 0
#    ### END PLAYER ###    
    else:
        print('Invalid filename!!!!')