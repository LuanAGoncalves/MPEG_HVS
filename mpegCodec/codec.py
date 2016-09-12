# -*- coding: utf-8 -*-
"""
Created on Sat May 23 14:37:54 2015

@author: luan
"""
import cv2
import numpy as np
from frames import mpeg
from frames import MJPEGcodec as jpeg
from frames import MJPEGhvs as jpeghvs
from math import sqrt, log, exp
from utils import detect_version as dtv
from utils.image_quality_assessment import metrics
import os

class Encoder:
    def __init__(self, videoName, quality = 50, sspace = 15, mode = '420', search = 0, hvsqm = 0, flat = 10.0, p = 2.0):
        '''
        # MPEG Encoder: \n
        Method: Constructor. \n
        About: This class runs the algorithm of the mpeg encoder. \n
        Parameters:  \n
            1) videoName: Video's name. \n
            2) quality: Quality of the image (default is 50). \n
            3) sspace: Search space (default is 15). \n
            4) mode: Down sample mode (default is 420). \n
            5) search: Search method. Use 0 self.outputr fullsearch and 1 self.outputr parallel hierarchical (default is 0). \n
            6) hvsqm: perceptual quantization flag (0 -> normal and 1 -> perceptual quantization ). \n
            7) flat = 10.0
            8) p = 2.0
        '''
        self.hvsqm = hvsqm    # Perceptual quantization flag.
        self.avgBits = []    # Average amount of bits in a pixel per frame.
        self.CRate = []     # Compression rate per frame.
        self.redundancy = []    # Redundancy per frame.
        self.hvstables = None   # Stores perceptual quantization tables if hvsqm = 1.
        self.p = p              # Used in the function genHVStables.
        self.flat = flat        # Coefficient's magnitudes of the flat quantization matrix.
        self.oR, self.oC, self.oD = [0, 0, 0]   # Used in the function readVideo.
        self.mbr, self.mbc = [16, 16]   # Macroblock's shape.
        self.mode = mode    # 444 or 420 (420 for mpeg-1).
        self.nframes = 0    # Number of frames.
        self.fps = 0        # Video rate.
        self.video = self.readVideo(videoName)
        self.output_name = ''
        if self.hvsqm == 0:
            if mode == '444':
                directory = './outputs/normal/444/'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                self.output_name = './outputs/normal/444/'+videoName.split('/')[-1].split('.')[0]+ '.txt'
                self.output = open(self.output_name, 'w')
            else:
                directory = './outputs/normal/420/'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                self.output_name = './outputs/normal/420/'+videoName.split('/')[-1].split('.')[0]+ '.txt'
                self.output = open(self.output_name, 'w')
                                                                
        else:
            if mode == '444':
                directory = './outputs/hvs/444/'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                self.output_name = './outputs/hvs/444/'+videoName.split('/')[-1].split('.')[0]+ '_hvs'+ '.txt'
                self.output = open(self.output_name, 'w')
            else:
                directory = './outputs/hvs/420/'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                self.output_name = './outputs/hvs/420/'+videoName.split('/')[-1].split('.')[0]+ '_hvs'+ '.txt'
                self.output = open(self.output_name, 'w')
        self.sspace = sspace    # Search space.
        self.search = search    # 0 - Full search; 1 - Parallel hierarchical.
        self.quality = quality  # Video's quality.
        self.hufftables = self.acdctables()
        self.Z = self.genQntb(self.quality)
        self.numBits = 0
        self.kbps =  0
        if self.hvsqm == 1:
            self.genHVStables()
            
        # Huffman doces for the motion vectors.
        self.vlc_mv_enc = {  (-30):('0010'),
                    (-29):('00010'),
                    (-28):('0000110'),
                    (-27):('00001010'),
                    (-26):('00001000'),
                    (-25):('00000110'),
                    (-24):('0000010110'),
                    (-23):('0000010100'),
                    (-22):('0000010010'),
                    (-21):('00000100010'),
                    (-20):('00000100000'),
                    (-19):('00000011110'),
                    (-18):('00000011100'),
                    (-17):('00000011010'),
                    (-16):('00000011001'),
                    (-15):('00000011011'),
                    (-14):('00000011101'),
                    (-13):('00000011111'),
                    (-12):('00000100001'),
                    (-11):('00000100011'),
                    (-10):('0000010011'),
                    (-9):('0000010101'),
                    (-8):('0000010111'),
                    (-7):('00000111'),
                    (-6):('00001001'),
                    (-5):('00001011'),
                    (-4):('0000111'),
                    (-3):('00011'),
                    (-2):('0011'),
                    (-1):('011'),
                    (0):('1'),
                    (1):('010'),
                    (2):('0010'),
                    (3):('00010'),
                    (4):('0000110'),
                    (5):('00001010'),
                    (6):('00001000'),
                    (7):('00000110'),
                    (8):('0000010110'),
                    (9):('0000010100'),
                    (10):('0000010010'),
                    (11):('00000100010'),
                    (12):('00000100000'),
                    (13):('00000011110'),
                    (14):('00000011100'),
                    (15):('00000011010'),
                    (16):('00000011001'),
                    (17):('00000011011'),
                    (18):('00000011101'),
                    (19):('00000011111'),
                    (20):('00000100001'),
                    (21):('00000100011'),
                    (22):('0000010011'),
                    (23):('0000010101'),
                    (24):('0000010111'),
                    (25):('00000111'),
                    (26):('00001001'),
                    (27):('00001011'),
                    (28):('0000111'),
                    (29):('00011'),
                    (30):('0011')}
                    
        self.vlc_mv_dec = {('00000011001'):(-16,16),
                    ('00000011011'):(-15,17),
                    ('00000011101'):(-14,18),
                    ('00000011111'):(-13,19),
                    ('00000100001'):(-12,20),
                    ('00000100011'):(-11,21),
                    ('0000010011'):(-10,22),
                    ('0000010101'):(-9,23),
                    ('0000010111'):(-8,24),
                    ('00000111'):(-7,25),
                    ('00001001'):(-6,26),
                    ('00001011'):(-5,27),
                    ('0000111'):(-4,28),
                    ('00011'):(-3,29),
                    ('0011'):(-2,30),
                    ('011'):(-1,-1),
                    ('1'):(0,0),
                    ('010'):(1,1),
                    ('0010'):(2,-30),
                    ('00010'):(3,-29),
                    ('0000110'):(4,-28),
                    ('00001010'):(5,-27),
                    ('00001000'):(6,-26),
                    ('00000110'):(7,-25),
                    ('0000010110'):(8,-24),
                    ('0000010100'):(9,-23),
                    ('0000010010'):(10,-22),
                    ('00000100010'):(11,-21),
                    ('00000100000'):(12,-20),
                    ('00000011110'):(13,-19),
                    ('00000011100'):(14,-18),
                    ('00000011010'):(15,-17),
                    ('00000011001'):(16,-16),
                    ('00000011011'):(17,-15)}
                    
    def vec2bin (self, MV, frame_type, nCol):
        """
        # MPEG Encoder:  \n
        Method: vec2bin(self, MV, frame_type, nCol)-> vec_seq \n
        About: Encodes the motion vectors. \n
        """
        c = nCol
        vec_seq = ''
        MV_dif = []
        
        if frame_type == '01':    #B frame
            for i in range(len(MV)):
                if MV[i][1] == 'f':
                    if i%c == 0:
                        MV_dif.append((MV[i][1],MV[i][2],MV[i][3]))
                    else:
                        if MV[i-1][1] == 'i':
                            MV_dif.append((MV[i][1], MV[i][2]-MV[i-1][2], MV[i][3]-MV[i-1][3]))
                        else:
                            MV_dif.append((MV[i][1], MV[i][2]-MV[i-1][2], MV[i][3]-MV[i-1][3]))
                elif MV[i][1] == 'b':
                    if i%c == 0:
                        MV_dif.append((MV[i][1],MV[i][2],MV[i][3]))
                    else:
                        if MV[i-1][1] == 'i':
                            MV_dif.append((MV[i][1], MV[i][2]-MV[i-1][2], MV[i][3]-MV[i-1][3]))
                        else:
                            MV_dif.append((MV[i][1], MV[i][2]-MV[i-1][2], MV[i][3]-MV[i-1][3]))
                elif MV[i][1] == 'i':
                    if i%c == 0:
                        MV_dif.append((MV[i][1],MV[i][2],MV[i][3],MV[i][4],MV[i][5]))
                    else:
                        if MV[i-1][1] == 'i':
                            MV_dif.append((MV[i][1], MV[i][2]-MV[i-1][2], MV[i][3]-MV[i-1][3], MV[i][4]-MV[i-1][4], MV[i][5]-MV[i-1][5]))
                        else:
                            MV_dif.append((MV[i][1], MV[i][2]-MV[i-1][2], MV[i][3]-MV[i-1][3], MV[i][4]-MV[i-1][2], MV[i][5]-MV[i-1][3]))
    
        else:                       #P frame
            for i in range(len(MV)):
                if i%c == 0:
                    MV_dif.append((MV[i][0],MV[i][1]))
                else:
                    MV_dif.append((MV[i][0]-MV[i-1][0], MV[i][1]-MV[i-1][1]))
                    
        mb_type = {('f'):('00'), ('b'):('01'), ('i'):('10')}
        
        for i in range (len(MV_dif)):
            if frame_type == '01':
                vec_seq += mb_type[MV_dif[i][0]]
                for j in range (1,len(MV_dif[i])):
                    vec_seq += str(self.vlc_mv_dec[self.vlc_mv_enc[MV_dif[i][j]]].index(MV_dif[i][j]))+self.vlc_mv_enc[MV_dif[i][j]]
            else:
                for j in range (len(MV_dif[i])):
                    vec_seq += str(self.vlc_mv_dec[self.vlc_mv_enc[MV_dif[i][j]]].index(MV_dif[i][j]))+self.vlc_mv_enc[MV_dif[i][j]]
                
        return vec_seq
    
    def readVideo(self, videoName):
        '''
        # MPEG Encoder: \n
        Method: readVideo(self, videoName)-> sequence \n
        About: This method store a video in a list. \n
        '''
        video = cv2.VideoCapture(videoName)
        if(dtv.opencv_version() >= 3):
            self.fps = video.get(cv2.CAP_PROP_FPS)
        else:
            self.fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        ret, fr = video.read()
        sequence = []
        sequence.append(self.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)))
        self.oR, self.oC, self.oD = fr.shape
        self.oR, self.oC, self.oD = fr.shape

        while ret:
            ret, fr = video.read()
            if ret != False:
                sequence.append(self.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)))
        video.release()
        
        self.nframes = len(sequence)
        
        if (len(sequence)-1)%6 != 0:
            for i in range(6-(len(sequence)-1)%6):
                sequence.append(sequence[-1])
        
        return sequence
    
    def resize (self, frame):
        '''
        # MPEG Encoder: \n
        Method: resize(self, frame)-> auxImage \n
        About: This method adjusts the shape of a given frame. \n
        '''
        rR, rC, rD = frame.shape
        aR = aC = 0

        if rR % self.mbr != 0:
            aR = rR + (self.mbr - (rR%self.mbr))
        else:
            aR = rR

        if rC % self.mbc != 0:
            aC = rC + (self.mbc - (rC%self.mbc))
        else:
            aC = rC

        for i in range (0,rR,2):
            for j in range (0,rC,2):
                frame[i+1,j,1] = frame[i,j+1,1] = frame[i+1,j+1,1] = frame[i,j,1]
                frame[i+1,j,2] = frame[i,j+1,2] = frame[i+1,j+1,2] = frame[i,j,2]

        auxImage = np.zeros((aR, aC, rD), np.float32)
        auxImage[:rR,:rC] = frame

        return auxImage
        
#    def genHVStables (self):
#        """
#        # MPEG Encoder: \n
#        Method: genHVStables(self)-> tables \n
#        About: Generates matrices of perceptual quantization. \n
#        """
#        tables = [[0 for x in range (self.sspace+1)] for x in range (self.sspace+1)]
#        g = [[0 for x in range (self.sspace+1)] for x in range (self.sspace+1)]
#        qflat = self.flat*np.ones((8,8), float)
#        for mh in range (self.sspace+1):
#            for mt in range (self.sspace+1):
#                vh = float(mh*self.fps)/float(self.oR)
#                vt = float(mt*self.fps)/float(self.oC)
#                v = sqrt(vh**2+vt**2)
#                gaux = np.zeros((8,8), np.float32)
#                const1 = float(mh*self.oR)/float(self.mbr*self.mbc)
#                const2 = float(mt*self.oC)/float(self.mbr*self.mbc)
#                if v != 0:
#                    for i in range (8):
#                        for j in range (8):
#                            ai = const1*0.5*i
#                            aj = const2*0.5*j
#                            aij = ai + aj
#                            gaux[i,j] = (6.1+7.3*abs(log(v/3.0))**3.0)*(v*(aij**2.0))*exp(-2.0*aij*(v+2.0)/45.9)
#                    g[mh][mt] = gaux
#                    
#                else:
#                    g[mh][mt] = gaux
#                    
#        g = np.array(g)
#        gmax = np.max(g)
#        for mh in range (self.sspace+1):
#            for mt in range (self.sspace+1):
#                qhvs = np.zeros((8,8), np.float32)
#                for i in range (8):
#                    for j in range (8):
#                        qhvs[i,j] = ((mh+mt)/float(self.p))*(1.-(g[mh,mt,i,j]/gmax))
#
#                tables[mh][mt] = qflat + qhvs
#                
#        self.hvstables = tables
        
    def genHVStables (self):
        """
        # MPEG Encoder: \n
        Method: genHVStables(self)-> tables \n
        About: Generates matrices of perceptual quantization. \n
        """
        if self.quality <= 70:
            max_value = self.Z.max()
        else:
            max_value = 119.9
        tables = [[0 for x in range (self.sspace+1)] for x in range (self.sspace+1)]
        g = [[0 for x in range (self.sspace+1)] for x in range (self.sspace+1)]
        min_value = 10.6577596664+234.260803223*0.859902977943**(self.quality)
        qflat = min_value*np.ones((8,8), float)
        for mh in range (self.sspace+1):
            for mt in range (self.sspace+1):
                vh = float(mh*self.fps)/float(self.oR)
                vt = float(mt*self.fps)/float(self.oC)
                v = sqrt(vh**2+vt**2)
                gaux = np.zeros((8,8), np.float32)
                const1 = float(mh*self.oR)/float(self.mbr*self.mbc)
                const2 = float(mt*self.oC)/float(self.mbr*self.mbc)
                if v != 0:
                    for i in range (8):
                        for j in range (8):
                            ai = const1*0.5*i
                            aj = const2*0.5*j
                            aij = ai + aj
                            gaux[i,j] = (6.1+7.3*abs(log(v/3.0))**3.0)*(v*(aij**2.0))*exp(-2.0*aij*(v+2.0)/45.9)
                    g[mh][mt] = gaux
                    
                else:
                    g[mh][mt] = gaux
                    
        g = np.array(g)
        gmax = np.max(g)
        for mh in range (self.sspace+1):
            for mt in range (self.sspace+1):
                qhvs = np.zeros((8,8), np.float32)
                for i in range (8):
                    for j in range (8):
                        qhvs[i,j] = ((mh+mt)/float(self.p))*(1.-(g[mh,mt,i,j]/gmax))
                q = (qflat + qhvs)
                tables[mh][mt] = ((q/np.linalg.norm(q))*(max_value-min_value))+min_value
#                tables[mh][mt] = q
                
        self.hvstables = tables
        
    def run (self):
        '''
        # MPEG Encoder: \n
        Method: run(self) \n
        About: This method runs the algorithm of the MPEG encoder. \n
        '''
        self.output.write(str(self.oR)+','+str(self.oC)+','+str(self.oD)+' '+str(self.quality)+ ' '+str(self.nframes)+' '+ self.mode + ' ' + str(self.sspace) + ' ' + str(self.hvsqm) +'\n')
        if self.hvsqm:
            self.output.write(str(self.flat)+' '+str(self.fps)+' '+str(self.p)+'\n')
        count = 0
        total = len(self.video)
        nBfr = 2
        gopsize = 6
        fP = 0
        
        print '###################### Starting MPEG Encoder ######################'
    
        for f in range(0,len(self.video)):
            # I B B P B B I B B P B B I B B P
            # 0 1 2 3 4 5 6 7 8 9
            print 'Frame ', f
            # I-frame
            if f%(gopsize) == 0:
                frame_type = '00'
                encoder = jpeg.Encoder( self.video[f], self.quality, self.hufftables, self.Z, self.mode)
                self.avgBits.append(encoder.avgBits)
                self.CRate.append(encoder.CRate)
                self.redundancy.append(1.-(1./encoder.CRate))
                self.numBits += encoder.NumBits
                self.output.write(frame_type+'\n'+str(encoder.seqhuff[0])+'\n'+str(encoder.seqhuff[1])+'\n'+str(encoder.seqhuff[2])+'\n')
                fP = f
                lP = f + (nBfr+1)
                count += 1
                print 'I - Progress', count, total
                
            # B-frame
            elif f%(nBfr+1) != 0:
                frame_type = '01'
                bframe = mpeg.Bframe(self.video[fP], self.video[f], self.video[lP], self.sspace, self.search)
                self.output.write(frame_type+'\n')
                vecsz = len(bframe.motionVec)
                MV = list(np.zeros(vecsz))
                for j in range (vecsz):
                    if bframe.motionVec[j][1] == 'i':
                        if self.hvsqm == 1:
                            MV[j] = (abs(bframe.motionVec[j][2]), abs(bframe.motionVec[j][3]), abs(bframe.motionVec[j][4]), abs(bframe.motionVec[j][5]))
                    else:
                        if self.hvsqm == 1:
                            MV[j] = ( abs(bframe.motionVec[j][2]), abs(bframe.motionVec[j][3]) )
                
                if self.hvsqm == 1:
                    ZtabVec = [self.hvstables, MV, self.sspace]
                    encoder = jpeghvs.Encoder(bframe.bframe, self.quality, self.hufftables, ZtabVec, self.hvsqm, self.mode)
                    self.avgBits.append(encoder.avgBits)
                    self.CRate.append(encoder.CRate)
                    self.redundancy.append(1.-(1./encoder.CRate))
                    self.numBits += encoder.NumBits
                else:
                    encoder = jpeg.Encoder(bframe.bframe, self.quality, self.hufftables, self.Z, self.mode)
                    self.avgBits.append(encoder.avgBits)
                    self.CRate.append(encoder.CRate)
                    self.redundancy.append(1.-(1./encoder.CRate))
                    self.numBits += encoder.NumBits
                
                mv_seq = self.vec2bin(bframe.motionVec, frame_type, self.video[0].shape[1]/self.mbc)
                self.output.write(mv_seq+'\n'+str(encoder.seqhuff[0])+'\n'+str(encoder.seqhuff[1])+'\n'+str(encoder.seqhuff[2])+'\n')
                count += 1
                print 'B - Progress', count, total
                
            # P-frame
            else:
                frame_type = '10'
                pframe = mpeg.Pframe(self.video[fP],self.video[f],self.sspace,self.search)    # P-frame
                self.output.write(frame_type+'\n')
                vecsz = len(pframe.motionVec)
                MV = list(np.zeros(vecsz))
                for j in range(vecsz):
                    MV[j] = ( abs(pframe.motionVec[j][0]), abs(pframe.motionVec[j][1]) )
                
                if self.hvsqm==1:
                    ZtabVec = [self.hvstables, MV, self.sspace]
                    encoder = jpeghvs.Encoder(pframe.pframe, self.quality, self.hufftables, ZtabVec, self.hvsqm, self.mode)
                    self.avgBits.append(encoder.avgBits)
                    self.CRate.append(encoder.CRate)
                    self.redundancy.append(1.-(1./encoder.CRate))
                    self.numBits += encoder.NumBits
                else:
                    encoder = jpeg.Encoder(pframe.pframe, self.quality, self.hufftables, self.Z, self.mode)
                    self.avgBits.append(encoder.avgBits)
                    self.CRate.append(encoder.CRate)
                    self.redundancy.append(1.-(1./encoder.CRate))
                    self.numBits += encoder.NumBits
                
                mv_seq = self.vec2bin(pframe.motionVec, frame_type, self.video[0].shape[1]/self.mbc)
                self.output.write(mv_seq + '\n' +str(encoder.seqhuff[0])+'\n'+str(encoder.seqhuff[1])+'\n'+str(encoder.seqhuff[2])+'\n')
                count += 1
                print 'P - Progress', count, total
                fP = f          #First P-frame
                lP = f+(nBfr+1) #Last P-frame
        self.kbps = (float(self.numBits*self.fps)/float(self.nframes))/1000.
    
        self.output.close()
        
    def acdctables(self):
        """
        # MPEG Encoder: \n
        Method: acdctables (self)-> (dcLumaTB, dcChroTB, acLumaTB, acChrmTB) \n
        About: Generates the Huffman code Tables for AC and DC coefficient differences.
        """
        dcLumaTB = { 0:(2,'00'),     1:(3,'010'),      2:(3,'011'),       3:(3,'100'),
                4:(3,'101'),    5:(3,'110'),      6:(4,'1110'),      7:(5,'11110'),
                8:(6,'111110'), 9:(7,'1111110'), 10:(8,'11111110'), 11:(9,'111111110')}
    
        dcChroTB = { 0:(2,'00'),       1:(2,'01'),         2:( 2,'10'),          3:( 3,'110'),
                4:(4,'1110'),     5:(5,'11110'),      6:( 6,'111110'),      7:( 7,'1111110'),
                8:(8,'11111110'), 9:(9,'111111110'), 10:(10,'1111111110'), 11:(11,'11111111110')}
                     
        #Table for luminance DC coefficient differences
        #       [(run,category) : (size, 'codeword')]
        acLumaTB = {( 0, 0):( 4,'1010'), #EOB
                ( 0, 1):( 2,'00'),               ( 0, 2):( 2,'01'),
                ( 0, 3):( 3,'100'),              ( 0, 4):( 4,'1011'),
                ( 0, 5):( 5,'11010'),            ( 0, 6):( 7,'1111000'),
                ( 0, 7):( 8,'11111000'),         ( 0, 8):(10,'1111110110'),
                ( 0, 9):(16,'1111111110000010'), ( 0,10):(16,'1111111110000011'),
                ( 1, 1):( 4,'1100'),             ( 1, 2):( 5,'11011'),
                ( 1, 3):( 7,'1111001'),          ( 1, 4):( 9,'111110110'),
                ( 1, 5):(11,'11111110110'),      ( 1, 6):(16,'1111111110000100'),
                ( 1, 7):(16,'1111111110000101'), ( 1, 8):(16,'1111111110000110'),
                ( 1, 9):(16,'1111111110000111'), ( 1,10):(16,'1111111110001000'),
                ( 2, 1):( 5,'11100'),            ( 2, 2):( 8,'11111001'),
                ( 2, 3):(10,'1111110111'),       ( 2, 4):(12,'111111110100'),
                ( 2, 5):(16,'1111111110001001'), ( 2, 6):(16,'1111111110001010'),
                ( 2, 7):(16,'1111111110001011'), ( 2, 8):(16,'1111111110001100'),
                ( 2, 9):(16,'1111111110001101'), ( 2,10):(16,'1111111110001110'),
                ( 3, 1):( 6,'111010'),           ( 3, 2):( 9,'111110111'),
                ( 3, 3):(12,'111111110101'),     ( 3, 4):(16,'1111111110001111'),
                ( 3, 5):(16,'1111111110010000'), ( 3, 6):(16,'1111111110010001'),
                ( 3, 7):(16,'1111111110010010'), ( 3, 8):(16,'1111111110010011'),
                ( 3, 9):(16,'1111111110010100'), ( 3,10):(16,'1111111110010101'),
                ( 4, 1):( 6,'111011'),           ( 4, 2):(10,'1111111000'),
                ( 4, 3):(16,'1111111110010110'), ( 4, 4):(16,'1111111110010111'),
                ( 4, 5):(16,'1111111110011000'), ( 4, 6):(16,'1111111110011001'),
                ( 4, 7):(16,'1111111110011010'), ( 4, 8):(16,'1111111110011011'),
                ( 4, 9):(16,'1111111110011100'), ( 4,10):(16,'1111111110011101'),
                ( 5, 1):( 7,'1111010'),          ( 5, 2):(11,'11111110111'),
                ( 5, 3):(16,'1111111110011110'), ( 5, 4):(16,'1111111110011111'),
                ( 5, 5):(16,'1111111110100000'), ( 5, 6):(16,'1111111110100001'),
                ( 5, 7):(16,'1111111110100010'), ( 5, 8):(16,'1111111110100011'),
                ( 5, 9):(16,'1111111110100100'), ( 5,10):(16,'1111111110100101'),
                ( 6, 1):( 7,'1111011'),          ( 6, 2):(12,'111111110110'),
                ( 6, 3):(16,'1111111110100110'), ( 6, 4):(16,'1111111110100111'),
                ( 6, 5):(16,'1111111110101000'), ( 6, 6):(16,'1111111110101001'),
                ( 6, 7):(16,'1111111110101010'), ( 6, 8):(16,'1111111110101011'),
                ( 6, 9):(16,'1111111110101100'), ( 6,10):(16,'1111111110101101'),
                ( 7, 1):( 8,'11111010'),         ( 7, 2):(12,'111111110111'),
                ( 7, 3):(16,'1111111110101110'), ( 7, 4):(16,'1111111110101111'),
                ( 7, 5):(16,'1111111110110000'), ( 7, 6):(16,'1111111110110001'),
                ( 7, 7):(16,'1111111110110010'), ( 7, 8):(16,'1111111110110011'),
                ( 7, 9):(16,'1111111110110100'), ( 7,10):(16,'1111111110110101'),
                ( 8, 1):( 9,'111111000'),        ( 8, 2):(15,'111111111000000'),
                ( 8, 3):(16,'1111111110110110'), ( 8, 4):(16,'1111111110110111'),
                ( 8, 5):(16,'1111111110111000'), ( 8, 6):(16,'1111111110111001'),
                ( 8, 7):(16,'1111111110111010'), ( 8, 8):(16,'1111111110111011'),
                ( 8, 9):(16,'1111111110111100'), ( 8,10):(16,'1111111110111101'),
                ( 9, 1):( 9,'111111001'),        ( 9, 2):(16,'1111111110111110'),
                ( 9, 3):(16,'1111111110111111'), ( 9, 4):(16,'1111111111000000'),
                ( 9, 5):(16,'1111111111000001'), ( 9, 6):(16,'1111111111000010'),
                ( 9, 7):(16,'1111111111000011'), ( 9, 8):(16,'1111111111000100'),
                ( 9, 9):(16,'1111111111000101'), ( 9,10):(16,'1111111111000110'),
                (10, 1):( 9,'111111010'),        (10, 2):(16,'1111111111000111'),
                (10, 3):(16,'1111111111001000'), (10, 4):(16,'1111111111001001'),
                (10, 5):(16,'1111111111001010'), (10, 6):(16,'1111111111001011'),
                (10, 7):(16,'1111111111001100'), (10, 8):(16,'1111111111001101'),
                (10, 9):(16,'1111111111001110'), (10,10):(16,'1111111111001111'),
                (11, 1):(10,'1111111001'),       (11, 2):(16,'1111111111010000'),
                (11, 3):(16,'1111111111010001'), (11, 4):(16,'1111111111010010'),
                (11, 5):(16,'1111111111010011'), (11, 6):(16,'1111111111010100'),
                (11, 7):(16,'1111111111010101'), (11, 8):(16,'1111111111010110'),
                (11, 9):(16,'1111111111010111'), (11,10):(16,'1111111111011000'),
                (12, 1):(10,'1111111010'),       (12, 2):(16,'1111111111011001'),
                (12, 3):(16,'1111111111011010'), (12, 4):(16,'1111111111011011'),
                (12, 5):(16,'1111111111011100'), (12, 6):(16,'1111111111011101'),
                (12, 7):(16,'1111111111011110'), (12, 8):(16,'1111111111011111'),
                (12, 9):(16,'1111111111100000'), (12,10):(16,'1111111111100001'),
                (13, 1):(11,'11111111000'),      (13, 2):(16,'1111111111100010'),
                (13, 3):(16,'1111111111100011'), (13, 4):(16,'1111111111100100'),
                (13, 5):(16,'1111111111100101'), (13, 6):(16,'1111111111100110'),
                (13, 7):(16,'1111111111100111'), (13, 8):(16,'1111111111101000'),
                (13, 9):(16,'1111111111101001'), (13,10):(16,'1111111111101010'),
                (14, 1):(16,'1111111111101011'), (14, 2):(16,'1111111111101100'),
                (14, 3):(16,'1111111111101101'), (14, 4):(16,'1111111111101110'),
                (14, 5):(16,'1111111111101111'), (14, 6):(16,'1111111111110000'),
                (14, 7):(16,'1111111111110001'), (14, 8):(16,'1111111111110010'),
                (14, 9):(16,'1111111111110011'), (14,10):(16,'1111111111110100'),
                (15, 0):(11,'11111111001'),     #(ZRL)
                (15, 1):(16,'1111111111110101'), (15, 2):(16,'1111111111110110'),
                (15, 3):(16,'1111111111110111'), (15, 4):(16,'1111111111111000'),
                (15, 5):(16,'1111111111111001'), (15, 6):(16,'1111111111111010'),
                (15, 7):(16,'1111111111111011'), (15, 8):(16,'1111111111111100'),
                (15, 9):(16,'1111111111111101'), (15,10):(16,'1111111111111110')}
                
        #Table for chrominance AC coefficients
        acChrmTB = {( 0, 0):( 2,'00'), #EOB
                ( 0, 1):( 2,'01'),               ( 0, 2):( 3,'100'),
                ( 0, 3):( 4,'1010'),             ( 0, 4):( 5,'11000'),
                ( 0, 5):( 5,'11001'),            ( 0, 6):( 6,'111000'),
                ( 0, 7):( 7,'1111000'),          ( 0, 8):( 9,'111110100'),
                ( 0, 9):(10,'1111110110'),       ( 0,10):(12,'111111110100'),
                ( 1, 1):( 4,'1011'),             ( 1, 2):( 6,'111001'),
                ( 1, 3):( 8,'11110110'),         ( 1, 4):( 9,'111110101'),
                ( 1, 5):(11,'11111110110'),      ( 1, 6):(12,'111111110101'),
                ( 1, 7):(16,'1111111110001000'), ( 1, 8):(16,'1111111110001001'),
                ( 1, 9):(16,'1111111110001010'), ( 1,10):(16,'1111111110001011'),
                ( 2, 1):( 5,'11010'),            ( 2, 2):( 8,'11110111'),
                ( 2, 3):(10,'1111110111'),       ( 2, 4):(12,'111111110110'),
                ( 2, 5):(15,'111111111000010'),  ( 2, 6):(16,'1111111110001100'),
                ( 2, 7):(16,'1111111110001101'), ( 2, 8):(16,'1111111110001110'),
                ( 2, 9):(16,'1111111110001111'), ( 2,10):(16,'1111111110010000'),
                ( 3, 1):( 5,'11011'),            ( 3, 2):( 8,'11111000'),
                ( 3, 3):(10,'1111111000'),       ( 3, 4):(12,'111111110111'),
                ( 3, 5):(16,'1111111110010001'), ( 3, 6):(16,'1111111110010010'),
                ( 3, 7):(16,'1111111110010011'), ( 3, 8):(16,'1111111110010100'),
                ( 3, 9):(16,'1111111110010101'), ( 3,10):(16,'1111111110010110'),
                ( 4, 1):( 6,'111010'),           ( 4, 2):( 9,'111110110'),
                ( 4, 3):(16,'1111111110010111'), ( 4, 4):(16,'1111111110011000'),
                ( 4, 5):(16,'1111111110011001'), ( 4, 6):(16,'1111111110011010'),
                ( 4, 7):(16,'1111111110011011'), ( 4, 8):(16,'1111111110011100'),
                ( 4, 9):(16,'1111111110011101'), ( 4,10):(16,'1111111110011110'),
                ( 5, 1):( 6,'111011'),           ( 5, 2):(10,'1111111001'),
                ( 5, 3):(16,'1111111110011111'), ( 5, 4):(16,'1111111110100000'),
                ( 5, 5):(16,'1111111110100001'), ( 5, 6):(16,'1111111110100010'),
                ( 5, 7):(16,'1111111110100011'), ( 5, 8):(16,'1111111110100100'),
                ( 5, 9):(16,'1111111110100101'), ( 5,10):(16,'1111111110100110'),
                ( 6, 1):( 7,'1111001'),          ( 6, 2):(11,'11111110111'),
                ( 6, 3):(16,'1111111110100111'), ( 6, 4):(16,'1111111110101000'),
                ( 6, 5):(16,'1111111110101001'), ( 6, 6):(16,'1111111110101010'),
                ( 6, 7):(16,'1111111110101011'), ( 6, 8):(16,'1111111110101100'),
                ( 6, 9):(16,'1111111110101101'), ( 6,10):(16,'1111111110101110'),
                ( 7, 1):( 7,'1111010'),          ( 7, 2):(11,'11111111000'),
                ( 7, 3):(16,'1111111110101111'), ( 7, 4):(16,'1111111110110000'),
                ( 7, 5):(16,'1111111110110001'), ( 7, 6):(16,'1111111110110010'),
                ( 7, 7):(16,'1111111110110011'), ( 7, 8):(16,'1111111110110100'),
                ( 7, 9):(16,'1111111110110101'), ( 7,10):(16,'1111111110110110'),
                ( 8, 1):( 8,'11111001'),         ( 8, 2):(16,'1111111110110111'),
                ( 8, 3):(16,'1111111110111000'), ( 8, 4):(16,'1111111110111001'),
                ( 8, 5):(16,'1111111110111010'), ( 8, 6):(16,'1111111110111011'),
                ( 8, 7):(16,'1111111110111100'), ( 8, 8):(16,'1111111110111101'),
                ( 8, 9):(16,'1111111110111110'), ( 8,10):(16,'1111111110111111'),
                ( 9, 1):( 9,'111110111'),        ( 9, 2):(16,'1111111111000000'),
                ( 9, 3):(16,'1111111111000001'), ( 9, 4):(16,'1111111111000010'),
                ( 9, 5):(16,'1111111111000011'), ( 9, 6):(16,'1111111111000100'),
                ( 9, 7):(16,'1111111111000101'), ( 9, 8):(16,'1111111111000110'),
                ( 9, 9):(16,'1111111111000111'), ( 9,10):(16,'1111111111001000'),
                (10, 1):( 9,'111111000'),        (10, 2):(16,'1111111111001001'),
                (10, 3):(16,'1111111111001010'), (10, 4):(16,'1111111111001011'),
                (10, 5):(16,'1111111111001100'), (10, 6):(16,'1111111111001101'),
                (10, 7):(16,'1111111111001110'), (10, 8):(16,'1111111111001111'),
                (10, 9):(16,'1111111111010000'), (10,10):(16,'1111111111010001'),
                (11, 1):( 9,'111111001'),        (11, 2):(16,'1111111111010010'),
                (11, 3):(16,'1111111111010011'), (11, 4):(16,'1111111111010100'),
                (11, 5):(16,'1111111111010101'), (11, 6):(16,'1111111111010110'),
                (11, 7):(16,'1111111111010111'), (11, 8):(16,'1111111111011000'),
                (11, 9):(16,'1111111111011001'), (11,10):(16,'1111111111011010'),
                (12, 1):( 9,'111111010'),        (12, 2):(16,'1111111111011011'),
                (12, 3):(16,'1111111111011100'), (12, 4):(16,'1111111111011101'),
                (12, 5):(16,'1111111111011110'), (12, 6):(16,'1111111111011111'),
                (12, 7):(16,'1111111111100000'), (12, 8):(16,'1111111111100001'),
                (12, 9):(16,'1111111111100010'), (12,10):(16,'1111111111100011'),
                (13, 1):(11,'11111111001'),      (13, 2):(16,'1111111111100100'),
                (13, 3):(16,'1111111111100101'), (13, 4):(16,'1111111111100110'),
                (13, 5):(16,'1111111111100111'), (13, 6):(16,'1111111111101000'),
                (13, 7):(16,'1111111111101001'), (13, 8):(16,'1111111111101010'),
                (13, 9):(16,'1111111111101011'), (13,10):(16,'1111111111101100'),
                (14, 1):(14,'11111111100000'),   (14, 2):(16,'1111111111101101'),
                (14, 3):(16,'1111111111101110'), (14, 4):(16,'1111111111101111'),
                (14, 5):(16,'1111111111110000'), (14, 6):(16,'1111111111110001'),
                (14, 7):(16,'1111111111110010'), (14, 8):(16,'1111111111110011'),
                (14, 9):(16,'1111111111110100'), (14,10):(16,'1111111111110101'),
                (15, 0):(10,'1111111010'),       #(ZRL)
                (15, 1):(15,'111111111000011'),  (15, 2):(16,'1111111111110110'),
                (15, 3):(16,'1111111111110111'), (15, 4):(16,'1111111111111000'),
                (15, 5):(16,'1111111111111001'), (15, 6):(16,'1111111111111010'),
                (15, 7):(16,'1111111111111011'), (15, 8):(16,'1111111111111100'),
                (15, 9):(16,'1111111111111101'), (15,10):(16,'1111111111111110')}
                    
        return (dcLumaTB, dcChroTB, acLumaTB, acChrmTB)
        
    def genQntb(self, qualy):
        
        '''
        # MPEG Encoder: \n
        Method: genQntb (self, qualy) -> qz \n
        About: Generates the standard quantization table. \n
        '''
    
        fact = qualy
        Z = np.array([[[16., 17., 17.], [11., 18., 18.], [10., 24., 24.], [16., 47., 47.], [124., 99., 99.], [140., 99., 99.], [151., 99., 99.], [161., 99., 99.]],
                  [[12., 18., 18.], [12., 21., 21.], [14., 26., 26.], [19., 66., 66.], [ 26., 99., 99.], [158., 99., 99.], [160., 99., 99.], [155., 99., 99.]],
                  [[14., 24., 24.], [13., 26., 26.], [16., 56., 56.], [24., 99., 99.], [ 40., 99., 99.], [157., 99., 99.], [169., 99., 99.], [156., 99., 99.]],
                  [[14., 47., 47.], [17., 66., 66.], [22., 99., 99.], [29., 99., 99.], [ 51., 99., 99.], [187., 99., 99.], [180., 99., 99.], [162., 99., 99.]],
                  [[18., 99., 99.], [22., 99., 99.], [37., 99., 99.], [56., 99., 99.], [ 68., 99., 99.], [109., 99., 99.], [103., 99., 99.], [177., 99., 99.]],
                  [[24., 99., 99.], [35., 99., 99.], [55., 99., 99.], [64., 99., 99.], [ 81., 99., 99.], [104., 99., 99.], [113., 99., 99.], [192., 99., 99.]],
                  [[49., 99., 99.], [64., 99., 99.], [78., 99., 99.], [87., 99., 99.], [103., 99., 99.], [121., 99., 99.], [120., 99., 99.], [101., 99., 99.]],
                  [[72., 99., 99.], [92., 99., 99.], [95., 99., 99.], [98., 99., 99.], [112., 99., 99.], [100., 99., 99.], [103., 99., 99.], [199., 99., 99.]]])
                  
        if qualy < 1 : fact = 1
        if qualy > 99: fact = 99
        if qualy < 50:
            qualy = 5000 / fact
        else:
            qualy = 200 - 2*fact
        
        qZ = ((Z*qualy) + 50)/100
        qZ[qZ<1] = 1
        qZ[qZ>255] = 255
    
        return qZ
            
class Decoder:
    def __init__(self, flName, videoName):
        '''
        # MPEG Decoder: \n
        Method: Constructor. \n
        About: This class runs the algorithm of the mpeg decoder. \n
        Parameters:  \n
            1) flname: File name. \n
        '''
        self.MBR, self.MBC = [16, 16]
        self.avgBits = []
        self.originalVideo = self.readVideo(videoName)
        self.psnrValues = []
        self.mssimValues = []
        self.input = open(flName,'r')
        self.hvstables = None
        self.nframes = 0
        self.quality = 0
        self.mode = ''
        self.sspace = 0
        self.shape = []
        self.hufftables = self.acdctables()
        self.Z = []
        self.hvsqm = None
        self.mbr, self.mbc = [8, 8]
        self.vlc_mv_dec = {('00000011001'):(-16,16),
                    ('00000011011'):(-15,17),
                    ('00000011101'):(-14,18),
                    ('00000011111'):(-13,19),
                    ('00000100001'):(-12,20),
                    ('00000100011'):(-11,21),
                    ('0000010011'):(-10,22),
                    ('0000010101'):(-9,23),
                    ('0000010111'):(-8,24),
                    ('00000111'):(-7,25),
                    ('00001001'):(-6,26),
                    ('00001011'):(-5,27),
                    ('0000111'):(-4,28),
                    ('00011'):(-3,29),
                    ('0011'):(-2,30),
                    ('011'):(-1,-1),
                    ('1'):(0,0),
                    ('010'):(1,1),
                    ('0010'):(2,-30),
                    ('00010'):(3,-29),
                    ('0000110'):(4,-28),
                    ('00001010'):(5,-27),
                    ('00001000'):(6,-26),
                    ('00000110'):(7,-25),
                    ('0000010110'):(8,-24),
                    ('0000010100'):(9,-23),
                    ('0000010010'):(10,-22),
                    ('00000100010'):(11,-21),
                    ('00000100000'):(12,-20),
                    ('00000011110'):(13,-19),
                    ('00000011100'):(14,-18),
                    ('00000011010'):(15,-17),
                    ('00000011001'):(16,-16),
                    ('00000011011'):(17,-15)}
                    
    def bin2vec (self, vec_seq, nCol, p_type):
        """
        # MPEG Decoder: \n
        Method: bin2vec(self, vec_seq, nCol, ptype) -> resp \n
        About: Decodes the motions vectors. \n
        """
        c = nCol
        MV = []
        resp = []
        if p_type == '01':   # B frame
            string_aux = ''
            signal = ''
            MC_type = ''
            aux_vec = []
            control = 0
            mc_type = {('00'):('f'), ('01'):('b'), ('10'):('i')}
            for i in range(len(vec_seq)):
                if control == 1:
                    control = 0
                    continue
                
                elif len(MC_type) == 0:
                    MC_type += vec_seq [i:i+2]
                    aux_vec.append(mc_type[MC_type])
                    control = 1
                else:
                    if mc_type[MC_type] == 'i':
                        if len(signal) == 0:
                            signal = vec_seq[i]
                        elif len(signal) != 0:
                            string_aux += vec_seq[i]
                            if string_aux in self.vlc_mv_dec:
                                aux_vec.append(self.vlc_mv_dec[string_aux][int(signal)])
                                if len(aux_vec) == 5:
                                    MV.append((aux_vec[0],aux_vec[1],aux_vec[2],aux_vec[3],aux_vec[4]))
                                    string_aux = ''
                                    signal = ''
                                    MC_type = ''
                                    aux_vec = []
                                else:
                                    string_aux = ''
                                    signal = ''
                            else:
                                pass
                    else:
                        if len(signal) == 0:
                            signal = vec_seq[i]
                        elif len(signal) != 0:
                            string_aux += vec_seq[i]
                            if string_aux in self.vlc_mv_dec:
                                aux_vec.append(self.vlc_mv_dec[string_aux][int(signal)])
                                if len(aux_vec) == 3:
                                    MV.append((aux_vec[0],aux_vec[1],aux_vec[2]))
                                    string_aux = ''
                                    signal = ''
                                    MC_type = ''
                                    aux_vec = []
                                else:
                                    string_aux = ''
                                    signal = ''
                            else:
                                pass
                                    
            for i in range (len(MV)):
                if i%c == 0:
                    if MV[i][0] != 'i':
                        resp.append((MV[i][0],MV[i][1],MV[i][2]))
                    else:
                        resp.append((MV[i][0],MV[i][1],MV[i][2],MV[i][3],MV[i][4]))
                                
                else:
                    if MV[i][0] != 'i':
                        if resp[-1][0] != 'i':
                            resp.append((MV[i][0],MV[i][1]+resp[i-1][1],MV[i][2]+resp[i-1][2]))
                        else:
                            resp.append((MV[i][0],MV[i][1]+resp[i-1][1],MV[i][2]+resp[i-1][2]))
                    else:
                        if resp[-1][0] != 'i':
                            resp.append((MV[i][0],MV[i][1]+resp[i-1][1],MV[i][2]+resp[i-1][2],MV[i][3]+resp[i-1][1],MV[i][4]+resp[i-1][2]))
                        else:
                            resp.append((MV[i][0],MV[i][1]+resp[i-1][1],MV[i][2]+resp[i-1][2],MV[i][3]+resp[i-1][3],MV[i][4]+resp[i-1][4]))
       
        
        else:                       # P frame
            string_aux = ''
            aux_vec = []
            signal = ''
            for i in range(len(vec_seq)):
                if len(signal) == 0:
                    signal = vec_seq[i]
                else:
                    string_aux += vec_seq[i]
                    if string_aux in self.vlc_mv_dec:
                        aux_vec.append(self.vlc_mv_dec[string_aux][int(signal)])
                        if len(aux_vec) == 2:
                            MV.append((aux_vec[0],aux_vec[1]))
                            string_aux = ''
                            signal = ''
                            aux_vec = []
                        else:
                            string_aux = ''
                            signal = ''
                    else:
                        pass
                    
            for i in range(len(MV)):
                if i%c == 0:
                    resp.append((MV[i][0],MV[i][1]))
                else:
                    resp.append((MV[i][0]+resp[i-1][0], MV[i][1]+resp[i-1][1]))
        return resp
        
    def readVideo(self, videoName):
        '''
        # MPEG Encoder: \n
        Method: readVideo(self, videoName)-> sequence \n
        About: This method store a video in a list. \n
        '''
        video = cv2.VideoCapture(videoName)
        if(dtv.opencv_version() >= 3):
            self.fps = video.get(cv2.CAP_PROP_FPS)
        else:
            self.fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        ret, fr = video.read()
        sequence = []
        sequence.append(self.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)))

        while ret:
            ret, fr = video.read()
            if ret != False:
                sequence.append(self.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)))
        video.release()
        
        self.nframes = len(sequence)
        
        if (len(sequence)-1)%6 != 0:
            for i in range(6-(len(sequence)-1)%6):
                sequence.append(sequence[-1])
        
        return sequence

    def resize (self, frame):
        '''
        # MPEG Encoder: \n
        Method: resize(self, frame)-> auxImage \n
        About: This method adjusts the shape of a given frame. \n
        '''
        rR, rC, rD = frame.shape
        aR = aC = 0

        if rR % self.MBR != 0:
            aR = rR + (self.MBR - (rR%self.MBR))
        else:
            aR = rR

        if rC % self.MBC != 0:
            aC = rC + (self.MBC - (rC%self.MBC))
        else:
            aC = rC

        for i in range (0,rR,2):
            for j in range (0,rC,2):
                frame[i+1,j,1] = frame[i,j+1,1] = frame[i+1,j+1,1] = frame[i,j,1]
                frame[i+1,j,2] = frame[i,j+1,2] = frame[i+1,j+1,2] = frame[i,j,2]

        auxImage = np.zeros((aR, aC, rD), np.float32)
        auxImage[:rR,:rC] = frame

        return auxImage
        
#    def genHVStables (self):
#        """
#        # MPEG Decoder: \n
#        Method: genHVStables(self)-> tables \n
#        About: Generates matrices of perceptual quantization. \n
#        """
#        tables = [[0 for x in range (int(self.sspace)+1)] for x in range (int(self.sspace)+1)]
#        g = [[0 for x in range (int(self.sspace)+1)] for x in range (int(self.sspace)+1)]
#        qflat = float(self.flat)*np.ones((8,8), np.float32)
#        for mh in range (int(self.sspace)+1):
#            for mt in range (int(self.sspace)+1):
#                
#                vh = float(mh*float(self.fps))/float(self.shape[0])
#                vt = float(mt*float(self.fps))/float(self.shape[1])
#                v = sqrt(vh**2+vt**2)
#                qhvs = np.zeros((8,8), float)
#                gaux = np.zeros((8,8), float)
#                const1 = float(mh*int(self.shape[0]))/float(int(self.MBR)*int(self.MBC))
#                const2 = float(mt*int(self.shape[1]))/float(int(self.MBR)*int(self.MBC))
#                if v != 0:
#                    for i in range (8):
#                        for j in range (8):
#                            ai = const1*0.5*i
#                            aj = const2*0.5*j
#                            aij = ai + aj
#                            gaux[i,j] = (6.1+7.3*abs(log(v/3.0))**3.0)*(v*(aij**2.0))*exp(-2.0*aij*(v+2.0)/45.9)
#                    g[mh][mt] = gaux
#                    
#                else:
#                    g[mh][mt] = gaux
#                    
#        g = np.array(g)
#        gmax = np.max(g)
#        for mh in range (int(self.sspace)+1):
#            for mt in range (int(self.sspace)+1):
#                qhvs = np.zeros((8,8), float)
#                for i in range (8):
#                    for j in range (8):
#                        qhvs[i,j] = ((mh+mt)/float(self.p))*(1.-(g[mh,mt,i,j]/gmax))
#                tables[mh][mt] = qflat + qhvs
#                
#        self.hvstables = tables
        
    def genHVStables (self):
        """
        # MPEG Decoder: \n
        Method: genHVStables(self)-> tables \n
        About: Generates matrices of perceptual quantization. \n
        """
        if self.quality <= 70:
            max_value = self.Z.max()
        else:
            max_value = 119.9
        tables = [[0 for x in range (int(self.sspace)+1)] for x in range (int(self.sspace)+1)]
        g = [[0 for x in range (int(self.sspace)+1)] for x in range (int(self.sspace)+1)]
        min_value = 10.6577596664+234.260803223*0.859902977943**(self.quality)
        qflat = min_value*np.ones((8,8), float)
#        print qflat
        for mh in range (int(self.sspace)+1):
            for mt in range (int(self.sspace)+1):
                
                vh = float(mh*float(self.fps))/float(self.shape[0])
                vt = float(mt*float(self.fps))/float(self.shape[1])
                v = sqrt(vh**2+vt**2)
                qhvs = np.zeros((8,8), float)
                gaux = np.zeros((8,8), float)
                const1 = float(mh*int(self.shape[0]))/float(int(self.MBR)*int(self.MBC))
                const2 = float(mt*int(self.shape[1]))/float(int(self.MBR)*int(self.MBC))
                if v != 0:
                    for i in range (8):
                        for j in range (8):
                            ai = const1*0.5*i
                            aj = const2*0.5*j
                            aij = ai + aj
                            gaux[i,j] = (6.1+7.3*abs(log(v/3.0))**3.0)*(v*(aij**2.0))*exp(-2.0*aij*(v+2.0)/45.9)
                    g[mh][mt] = gaux
                    
                else:
                    g[mh][mt] = gaux
                    
        g = np.array(g)
        gmax = np.max(g)
        for mh in range (int(self.sspace)+1):
            for mt in range (int(self.sspace)+1):
                qhvs = np.zeros((8,8), float)
                for i in range (8):
                    for j in range (8):
                        qhvs[i,j] = ((mh+mt)/float(self.p))*(1.-(g[mh,mt,i,j]/gmax))
                q = (qflat + qhvs)
                tables[mh][mt] = ((q/np.linalg.norm(q))*(max_value-min_value))+min_value
#                tables[mh][mt] = q
                
        self.hvstables = tables
        
    def precover (self, pastfr, currentfr, motionVecs, sspace):
        '''
        # MPEG Decoder: \n
        Method: precover(self, pastfr, currentfr, motionVecs, sspace)-> result \n
        About: This method revocers a P frame. \n
        Parameters:  \n
            1) pastfr: Past frame. \n
            2) currentfr: Current frame. \n
            3) motionVecs: Motion vectors. \n
            4) sspace: Search space. \n
        '''
        result = np.zeros(pastfr.shape, np.float32)
        count = 0
        
#        print motionVecs
            
        for i in range(0,currentfr.shape[0],16):
            for j in range(0,currentfr.shape[1],16):
#                print motionVecs[count]
                a, b = motionVecs[count]
                aR, aC, aD = currentfr.shape
                backgroundImgPast = np.zeros((aR+2*sspace, aC+2*sspace, aD), np.float32)
                backgroundImgPast[sspace:sspace+aR, sspace:sspace+aC] = pastfr
                result[i:i+16, j:j+16] = backgroundImgPast[i+a+sspace:i+a+sspace+16, j+b+sspace:j+b+sspace+16] + currentfr[i:i+16, j:j+16]
                count += 1
        return result
        
    def brecover (self, pastfr, currentfr, postfr, motionVecs, sspace):
        '''
        # MPEG Decoder: \n
        Method: precover(self, pastfr, currentfr, motionVecs, sspace)-> result \n
        About: This method revocers a B frame. \n
        Parameters:  \n
            1) pastfr: Past frame. \n
            2) currentfr: Current frame. \n
            3) postfr: Post frame. \n
            4) motionVecs: Motion vectors. \n
            5) sspace: Search space. \n
        '''
        result = np.zeros(pastfr.shape, np.float32)
        count = 0
        
        aR, aC, aD = currentfr.shape
        backgroundImgPast = np.zeros((aR+2*sspace, aC+2*sspace, aD), np.float32)
        backgroundImgPast[sspace:sspace+aR, sspace:sspace+aC] = pastfr
        backgroundImgPost= np.zeros((aR+2*sspace, aC+2*sspace, aD), np.float32)
        backgroundImgPost[sspace:sspace+aR, sspace:sspace+aC] = postfr
    
        for i in range(0,currentfr.shape[0],16):
            for j in range(0,currentfr.shape[1],16):
                if motionVecs[count][0] == 'i':
                    a, b, c, d = motionVecs[count][1:]
                    result[i:i+16, j:j+16] = (backgroundImgPast[i+a+sspace:i+a+sspace+16, j+b+sspace:j+b+sspace+16] + 2.0*currentfr[i:i+16, j:j+16] + backgroundImgPost[i+c+sspace:i+c+sspace+16, j+d+sspace:j+d+sspace+16])/2.0
                    
                if motionVecs[count][0] == 'b':
                    a, b = motionVecs[count][1:]
                    result[i:i+16, j:j+16] = backgroundImgPost[i+a+sspace:i+a+sspace+16, j+b+sspace:j+b+sspace+16] + currentfr[i:i+16, j:j+16]
                    
                elif motionVecs[count][0] == 'f':
                    a, b = motionVecs[count][1:]
                    result[i:i+16, j:j+16] = backgroundImgPast[i+a+sspace:i+a+sspace+16, j+b+sspace:j+b+sspace+16] + currentfr[i:i+16, j:j+16]
                    
                count += 1
        return result
        
    def run (self):
        '''
        # MPEG Decoder: \n
        Method: run(self) \n
        About: This method runs the algorithm of the MPEG decoder. \n
        '''
        fo = self.input.read()
        fo = fo.split('\n')
        
        self.shape, self.quality, self.nframes, self.mode, self.sspace, self.hvsqm = fo[0].split(' ')
        self.shape = np.array(self.shape.split(','), int)
        self.quality = int(self.quality)
        self.nframes = int(self.nframes)
        self.hvsqm   = int(self.hvsqm)
        self.Z = self.genQntb(self.quality)
        self.hvstables = None
        count = 1
        if self.hvsqm == 1:
            fo[count].split(' ')
            self.flat, self.fps, self.p = fo[count].split(' ')
            self.genHVStables()
            count += 1
            
        sequence = []
        nauxfr = self.nframes+(6-((self.nframes-1)%6))
        self.sspace = int(self.sspace)
        countfr = 0
        
        print '\n### Starting MPEG Decoder ###'
        print '1) JPEG:'
        mb_type = {('00'):('I'), ('01'):('B'), ('10'):('P')}
        nl = self.shape[0] if self.shape[0]%16 == 0 else self.shape[0]+(self.shape[0]%16 - 16)
        ml = self.shape[1] if self.shape[1]%16 == 0 else self.shape[1]+(self.shape[1]%16 - 16)
        while countfr < nauxfr:
            aux = fo[count]
            if mb_type[aux] == 'I':
                ch = []
                aux = 0
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                self.avgBits.append(float(aux)/float(nl*ml))
                sequence.append([countfr, jpeg.Decoder(ch, self.hufftables, self.Z, [self.shape, self.quality, self.mode])._run_(), None])
                count += 1
                countfr += 1
            
            elif mb_type[aux] == 'P':
                count += 1
#                vecSTR = aux[1][1:].split(':')
                motionVec = []
#                for i in range(len(vecSTR)):
#                    motionVec.append(tuple(np.array(vecSTR[i].split(','), int)))
                if self.shape[1]%self.MBC == 0:
                    motionVec = self.bin2vec(fo[count], self.shape[1]/self.MBC, aux)
                else:
                    motionVec = self.bin2vec(fo[count], (self.shape[1]+(self.MBC-(self.shape[1]%self.MBC)))/self.MBC, aux)
                ch = []
                aux = 0
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                self.avgBits.append(float(aux)/float(nl*ml))
                if self.hvsqm == 1:
                    sequence.append([countfr, jpeghvs.Decoder(ch, self.hufftables, self.hvstables, [self.shape, self.quality, self.mode, motionVec])._run_(), motionVec])
                else:
                    sequence.append([countfr, jpeg.Decoder(ch, self.hufftables, self.Z, [self.shape, self.quality, self.mode])._run_(), motionVec])
                count += 1
                countfr += 1
            
            elif mb_type[aux] == 'B':
                count += 1
#                vecSTR = aux[1][1:].split(':')
                motionVec = []
#                for i in range(len(vecSTR)):
#                    if vecSTR[i].split(',')[0] == 'i':
#                        motionVec.append((vecSTR[i].split(',')[0], int(vecSTR[i].split(',')[1]), int(vecSTR[i].split(',')[2]), int(vecSTR[i].split(',')[3]), int(vecSTR[i].split(',')[4])))
#                    else:
#                        motionVec.append((vecSTR[i].split(',')[0], int(vecSTR[i].split(',')[1]), int(vecSTR[i].split(',')[2])))
                if self.shape[1]%self.MBC == 0:
                    motionVec = self.bin2vec(fo[count], self.shape[1]/self.MBC, aux)
                else:
                    motionVec = self.bin2vec(fo[count], (self.shape[1]+(self.MBC-(self.shape[1]%self.MBC)))/self.MBC, aux)
                ch = []
                aux = 0
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                self.avgBits.append(float(aux)/float(nl*ml))
                if self.hvsqm == 1:
                    sequence.append([countfr, jpeghvs.Decoder(ch, self.hufftables, self.hvstables, [self.shape, self.quality, self.mode, motionVec])._run_(), motionVec])
                else:
                    sequence.append([countfr, jpeg.Decoder(ch, self.hufftables, self.Z, [self.shape, self.quality, self.mode])._run_(), motionVec])
                count += 1
                countfr += 1
            print 'Progress: %d/%d' % (countfr,nauxfr)
        
        count = 0
        sequence.sort(key=lambda tup: tup[0])
        self.input.close()
        print '2) MPEG:'
        for i in range (0,len(sequence)-1,6):
            sequence[i+3][1] = self.precover(sequence[i][1],sequence[i+3][1], sequence[i+3][2], self.sspace)
            sequence[i+1][1] = self.brecover(sequence[i][1], sequence[i+1][1], sequence[i+3][1], sequence[i+1][2], self.sspace)
            sequence[i+2][1] = self.brecover(sequence[i][1], sequence[i+2][1], sequence[i+3][1], sequence[i+2][2], self.sspace)
            sequence[i+4][1] = self.brecover(sequence[i+3][1], sequence[i+4][1], sequence[i+6][1], sequence[i+4][2], self.sspace)
            sequence[i+5][1] = self.brecover(sequence[i+3][1], sequence[i+5][1], sequence[i+6][1], sequence[i+5][2], self.sspace)
        
        output = []
        print "%s Computing visual metrics. %s\nPlease wait..." % ("#"*4,"#"*4)
        for i in range (self.nframes+1):
            sequence[i][1][sequence[i][1]>255.0] = 255.0
            sequence[i][1][sequence[i][1]<0.0] = 0.0
            output.append(cv2.cvtColor(np.uint8(sequence[i][1]), cv2.COLOR_YCR_CB2BGR))
            self.mssimValues.append(metrics.msim(self.originalVideo[i], output[-1]))
            self.psnrValues.append(metrics.psnr(self.originalVideo[i], output[-1]))
        print "Thanks!"
        
        return [output, self.psnrValues, self.mssimValues, self.avgBits]
            
    def acdctables(self):
        """
        # MPEG Decoder: \n
        Method: acdctables (self)-> (dcLumaTB, dcChroTB, acLumaTB, acChrmTB) \n
        About: Generates the Huffman code Tables for AC and DC coefficient differences.
        """
        dcLumaTB = { 0:(2,'00'),     1:(3,'010'),      2:(3,'011'),       3:(3,'100'),
                4:(3,'101'),    5:(3,'110'),      6:(4,'1110'),      7:(5,'11110'),
                8:(6,'111110'), 9:(7,'1111110'), 10:(8,'11111110'), 11:(9,'111111110')}
    
        dcChroTB = { 0:(2,'00'),       1:(2,'01'),         2:( 2,'10'),          3:( 3,'110'),
                4:(4,'1110'),     5:(5,'11110'),      6:( 6,'111110'),      7:( 7,'1111110'),
                8:(8,'11111110'), 9:(9,'111111110'), 10:(10,'1111111110'), 11:(11,'11111111110')}
                     
        #Table for luminance DC coefficient differences
        #       [(run,category) : (size, 'codeword')]
        acLumaTB = {( 0, 0):( 4,'1010'), #EOB
                ( 0, 1):( 2,'00'),               ( 0, 2):( 2,'01'),
                ( 0, 3):( 3,'100'),              ( 0, 4):( 4,'1011'),
                ( 0, 5):( 5,'11010'),            ( 0, 6):( 7,'1111000'),
                ( 0, 7):( 8,'11111000'),         ( 0, 8):(10,'1111110110'),
                ( 0, 9):(16,'1111111110000010'), ( 0,10):(16,'1111111110000011'),
                ( 1, 1):( 4,'1100'),             ( 1, 2):( 5,'11011'),
                ( 1, 3):( 7,'1111001'),          ( 1, 4):( 9,'111110110'),
                ( 1, 5):(11,'11111110110'),      ( 1, 6):(16,'1111111110000100'),
                ( 1, 7):(16,'1111111110000101'), ( 1, 8):(16,'1111111110000110'),
                ( 1, 9):(16,'1111111110000111'), ( 1,10):(16,'1111111110001000'),
                ( 2, 1):( 5,'11100'),            ( 2, 2):( 8,'11111001'),
                ( 2, 3):(10,'1111110111'),       ( 2, 4):(12,'111111110100'),
                ( 2, 5):(16,'1111111110001001'), ( 2, 6):(16,'1111111110001010'),
                ( 2, 7):(16,'1111111110001011'), ( 2, 8):(16,'1111111110001100'),
                ( 2, 9):(16,'1111111110001101'), ( 2,10):(16,'1111111110001110'),
                ( 3, 1):( 6,'111010'),           ( 3, 2):( 9,'111110111'),
                ( 3, 3):(12,'111111110101'),     ( 3, 4):(16,'1111111110001111'),
                ( 3, 5):(16,'1111111110010000'), ( 3, 6):(16,'1111111110010001'),
                ( 3, 7):(16,'1111111110010010'), ( 3, 8):(16,'1111111110010011'),
                ( 3, 9):(16,'1111111110010100'), ( 3,10):(16,'1111111110010101'),
                ( 4, 1):( 6,'111011'),           ( 4, 2):(10,'1111111000'),
                ( 4, 3):(16,'1111111110010110'), ( 4, 4):(16,'1111111110010111'),
                ( 4, 5):(16,'1111111110011000'), ( 4, 6):(16,'1111111110011001'),
                ( 4, 7):(16,'1111111110011010'), ( 4, 8):(16,'1111111110011011'),
                ( 4, 9):(16,'1111111110011100'), ( 4,10):(16,'1111111110011101'),
                ( 5, 1):( 7,'1111010'),          ( 5, 2):(11,'11111110111'),
                ( 5, 3):(16,'1111111110011110'), ( 5, 4):(16,'1111111110011111'),
                ( 5, 5):(16,'1111111110100000'), ( 5, 6):(16,'1111111110100001'),
                ( 5, 7):(16,'1111111110100010'), ( 5, 8):(16,'1111111110100011'),
                ( 5, 9):(16,'1111111110100100'), ( 5,10):(16,'1111111110100101'),
                ( 6, 1):( 7,'1111011'),          ( 6, 2):(12,'111111110110'),
                ( 6, 3):(16,'1111111110100110'), ( 6, 4):(16,'1111111110100111'),
                ( 6, 5):(16,'1111111110101000'), ( 6, 6):(16,'1111111110101001'),
                ( 6, 7):(16,'1111111110101010'), ( 6, 8):(16,'1111111110101011'),
                ( 6, 9):(16,'1111111110101100'), ( 6,10):(16,'1111111110101101'),
                ( 7, 1):( 8,'11111010'),         ( 7, 2):(12,'111111110111'),
                ( 7, 3):(16,'1111111110101110'), ( 7, 4):(16,'1111111110101111'),
                ( 7, 5):(16,'1111111110110000'), ( 7, 6):(16,'1111111110110001'),
                ( 7, 7):(16,'1111111110110010'), ( 7, 8):(16,'1111111110110011'),
                ( 7, 9):(16,'1111111110110100'), ( 7,10):(16,'1111111110110101'),
                ( 8, 1):( 9,'111111000'),        ( 8, 2):(15,'111111111000000'),
                ( 8, 3):(16,'1111111110110110'), ( 8, 4):(16,'1111111110110111'),
                ( 8, 5):(16,'1111111110111000'), ( 8, 6):(16,'1111111110111001'),
                ( 8, 7):(16,'1111111110111010'), ( 8, 8):(16,'1111111110111011'),
                ( 8, 9):(16,'1111111110111100'), ( 8,10):(16,'1111111110111101'),
                ( 9, 1):( 9,'111111001'),        ( 9, 2):(16,'1111111110111110'),
                ( 9, 3):(16,'1111111110111111'), ( 9, 4):(16,'1111111111000000'),
                ( 9, 5):(16,'1111111111000001'), ( 9, 6):(16,'1111111111000010'),
                ( 9, 7):(16,'1111111111000011'), ( 9, 8):(16,'1111111111000100'),
                ( 9, 9):(16,'1111111111000101'), ( 9,10):(16,'1111111111000110'),
                (10, 1):( 9,'111111010'),        (10, 2):(16,'1111111111000111'),
                (10, 3):(16,'1111111111001000'), (10, 4):(16,'1111111111001001'),
                (10, 5):(16,'1111111111001010'), (10, 6):(16,'1111111111001011'),
                (10, 7):(16,'1111111111001100'), (10, 8):(16,'1111111111001101'),
                (10, 9):(16,'1111111111001110'), (10,10):(16,'1111111111001111'),
                (11, 1):(10,'1111111001'),       (11, 2):(16,'1111111111010000'),
                (11, 3):(16,'1111111111010001'), (11, 4):(16,'1111111111010010'),
                (11, 5):(16,'1111111111010011'), (11, 6):(16,'1111111111010100'),
                (11, 7):(16,'1111111111010101'), (11, 8):(16,'1111111111010110'),
                (11, 9):(16,'1111111111010111'), (11,10):(16,'1111111111011000'),
                (12, 1):(10,'1111111010'),       (12, 2):(16,'1111111111011001'),
                (12, 3):(16,'1111111111011010'), (12, 4):(16,'1111111111011011'),
                (12, 5):(16,'1111111111011100'), (12, 6):(16,'1111111111011101'),
                (12, 7):(16,'1111111111011110'), (12, 8):(16,'1111111111011111'),
                (12, 9):(16,'1111111111100000'), (12,10):(16,'1111111111100001'),
                (13, 1):(11,'11111111000'),      (13, 2):(16,'1111111111100010'),
                (13, 3):(16,'1111111111100011'), (13, 4):(16,'1111111111100100'),
                (13, 5):(16,'1111111111100101'), (13, 6):(16,'1111111111100110'),
                (13, 7):(16,'1111111111100111'), (13, 8):(16,'1111111111101000'),
                (13, 9):(16,'1111111111101001'), (13,10):(16,'1111111111101010'),
                (14, 1):(16,'1111111111101011'), (14, 2):(16,'1111111111101100'),
                (14, 3):(16,'1111111111101101'), (14, 4):(16,'1111111111101110'),
                (14, 5):(16,'1111111111101111'), (14, 6):(16,'1111111111110000'),
                (14, 7):(16,'1111111111110001'), (14, 8):(16,'1111111111110010'),
                (14, 9):(16,'1111111111110011'), (14,10):(16,'1111111111110100'),
                (15, 0):(11,'11111111001'),     #(ZRL)
                (15, 1):(16,'1111111111110101'), (15, 2):(16,'1111111111110110'),
                (15, 3):(16,'1111111111110111'), (15, 4):(16,'1111111111111000'),
                (15, 5):(16,'1111111111111001'), (15, 6):(16,'1111111111111010'),
                (15, 7):(16,'1111111111111011'), (15, 8):(16,'1111111111111100'),
                (15, 9):(16,'1111111111111101'), (15,10):(16,'1111111111111110')}
                
        #Table for chrominance AC coefficients
        acChrmTB = {( 0, 0):( 2,'00'), #EOB
                ( 0, 1):( 2,'01'),               ( 0, 2):( 3,'100'),
                ( 0, 3):( 4,'1010'),             ( 0, 4):( 5,'11000'),
                ( 0, 5):( 5,'11001'),            ( 0, 6):( 6,'111000'),
                ( 0, 7):( 7,'1111000'),          ( 0, 8):( 9,'111110100'),
                ( 0, 9):(10,'1111110110'),       ( 0,10):(12,'111111110100'),
                ( 1, 1):( 4,'1011'),             ( 1, 2):( 6,'111001'),
                ( 1, 3):( 8,'11110110'),         ( 1, 4):( 9,'111110101'),
                ( 1, 5):(11,'11111110110'),      ( 1, 6):(12,'111111110101'),
                ( 1, 7):(16,'1111111110001000'), ( 1, 8):(16,'1111111110001001'),
                ( 1, 9):(16,'1111111110001010'), ( 1,10):(16,'1111111110001011'),
                ( 2, 1):( 5,'11010'),            ( 2, 2):( 8,'11110111'),
                ( 2, 3):(10,'1111110111'),       ( 2, 4):(12,'111111110110'),
                ( 2, 5):(15,'111111111000010'),  ( 2, 6):(16,'1111111110001100'),
                ( 2, 7):(16,'1111111110001101'), ( 2, 8):(16,'1111111110001110'),
                ( 2, 9):(16,'1111111110001111'), ( 2,10):(16,'1111111110010000'),
                ( 3, 1):( 5,'11011'),            ( 3, 2):( 8,'11111000'),
                ( 3, 3):(10,'1111111000'),       ( 3, 4):(12,'111111110111'),
                ( 3, 5):(16,'1111111110010001'), ( 3, 6):(16,'1111111110010010'),
                ( 3, 7):(16,'1111111110010011'), ( 3, 8):(16,'1111111110010100'),
                ( 3, 9):(16,'1111111110010101'), ( 3,10):(16,'1111111110010110'),
                ( 4, 1):( 6,'111010'),           ( 4, 2):( 9,'111110110'),
                ( 4, 3):(16,'1111111110010111'), ( 4, 4):(16,'1111111110011000'),
                ( 4, 5):(16,'1111111110011001'), ( 4, 6):(16,'1111111110011010'),
                ( 4, 7):(16,'1111111110011011'), ( 4, 8):(16,'1111111110011100'),
                ( 4, 9):(16,'1111111110011101'), ( 4,10):(16,'1111111110011110'),
                ( 5, 1):( 6,'111011'),           ( 5, 2):(10,'1111111001'),
                ( 5, 3):(16,'1111111110011111'), ( 5, 4):(16,'1111111110100000'),
                ( 5, 5):(16,'1111111110100001'), ( 5, 6):(16,'1111111110100010'),
                ( 5, 7):(16,'1111111110100011'), ( 5, 8):(16,'1111111110100100'),
                ( 5, 9):(16,'1111111110100101'), ( 5,10):(16,'1111111110100110'),
                ( 6, 1):( 7,'1111001'),          ( 6, 2):(11,'11111110111'),
                ( 6, 3):(16,'1111111110100111'), ( 6, 4):(16,'1111111110101000'),
                ( 6, 5):(16,'1111111110101001'), ( 6, 6):(16,'1111111110101010'),
                ( 6, 7):(16,'1111111110101011'), ( 6, 8):(16,'1111111110101100'),
                ( 6, 9):(16,'1111111110101101'), ( 6,10):(16,'1111111110101110'),
                ( 7, 1):( 7,'1111010'),          ( 7, 2):(11,'11111111000'),
                ( 7, 3):(16,'1111111110101111'), ( 7, 4):(16,'1111111110110000'),
                ( 7, 5):(16,'1111111110110001'), ( 7, 6):(16,'1111111110110010'),
                ( 7, 7):(16,'1111111110110011'), ( 7, 8):(16,'1111111110110100'),
                ( 7, 9):(16,'1111111110110101'), ( 7,10):(16,'1111111110110110'),
                ( 8, 1):( 8,'11111001'),         ( 8, 2):(16,'1111111110110111'),
                ( 8, 3):(16,'1111111110111000'), ( 8, 4):(16,'1111111110111001'),
                ( 8, 5):(16,'1111111110111010'), ( 8, 6):(16,'1111111110111011'),
                ( 8, 7):(16,'1111111110111100'), ( 8, 8):(16,'1111111110111101'),
                ( 8, 9):(16,'1111111110111110'), ( 8,10):(16,'1111111110111111'),
                ( 9, 1):( 9,'111110111'),        ( 9, 2):(16,'1111111111000000'),
                ( 9, 3):(16,'1111111111000001'), ( 9, 4):(16,'1111111111000010'),
                ( 9, 5):(16,'1111111111000011'), ( 9, 6):(16,'1111111111000100'),
                ( 9, 7):(16,'1111111111000101'), ( 9, 8):(16,'1111111111000110'),
                ( 9, 9):(16,'1111111111000111'), ( 9,10):(16,'1111111111001000'),
                (10, 1):( 9,'111111000'),        (10, 2):(16,'1111111111001001'),
                (10, 3):(16,'1111111111001010'), (10, 4):(16,'1111111111001011'),
                (10, 5):(16,'1111111111001100'), (10, 6):(16,'1111111111001101'),
                (10, 7):(16,'1111111111001110'), (10, 8):(16,'1111111111001111'),
                (10, 9):(16,'1111111111010000'), (10,10):(16,'1111111111010001'),
                (11, 1):( 9,'111111001'),        (11, 2):(16,'1111111111010010'),
                (11, 3):(16,'1111111111010011'), (11, 4):(16,'1111111111010100'),
                (11, 5):(16,'1111111111010101'), (11, 6):(16,'1111111111010110'),
                (11, 7):(16,'1111111111010111'), (11, 8):(16,'1111111111011000'),
                (11, 9):(16,'1111111111011001'), (11,10):(16,'1111111111011010'),
                (12, 1):( 9,'111111010'),        (12, 2):(16,'1111111111011011'),
                (12, 3):(16,'1111111111011100'), (12, 4):(16,'1111111111011101'),
                (12, 5):(16,'1111111111011110'), (12, 6):(16,'1111111111011111'),
                (12, 7):(16,'1111111111100000'), (12, 8):(16,'1111111111100001'),
                (12, 9):(16,'1111111111100010'), (12,10):(16,'1111111111100011'),
                (13, 1):(11,'11111111001'),      (13, 2):(16,'1111111111100100'),
                (13, 3):(16,'1111111111100101'), (13, 4):(16,'1111111111100110'),
                (13, 5):(16,'1111111111100111'), (13, 6):(16,'1111111111101000'),
                (13, 7):(16,'1111111111101001'), (13, 8):(16,'1111111111101010'),
                (13, 9):(16,'1111111111101011'), (13,10):(16,'1111111111101100'),
                (14, 1):(14,'11111111100000'),   (14, 2):(16,'1111111111101101'),
                (14, 3):(16,'1111111111101110'), (14, 4):(16,'1111111111101111'),
                (14, 5):(16,'1111111111110000'), (14, 6):(16,'1111111111110001'),
                (14, 7):(16,'1111111111110010'), (14, 8):(16,'1111111111110011'),
                (14, 9):(16,'1111111111110100'), (14,10):(16,'1111111111110101'),
                (15, 0):(10,'1111111010'),       #(ZRL)
                (15, 1):(15,'111111111000011'),  (15, 2):(16,'1111111111110110'),
                (15, 3):(16,'1111111111110111'), (15, 4):(16,'1111111111111000'),
                (15, 5):(16,'1111111111111001'), (15, 6):(16,'1111111111111010'),
                (15, 7):(16,'1111111111111011'), (15, 8):(16,'1111111111111100'),
                (15, 9):(16,'1111111111111101'), (15,10):(16,'1111111111111110')}
                    
        return (dcLumaTB, dcChroTB, acLumaTB, acChrmTB)
        
    def genQntb(self, qualy):
        
        '''
        # MPEG Decoder: \n
        Method: genQntb (self, qualy) -> qz \n
        About: Generates the standard quantization table. \n
        '''
    
        fact = qualy
        Z = np.array([[[16., 17., 17.], [11., 18., 18.], [10., 24., 24.], [16., 47., 47.], [124., 99., 99.], [140., 99., 99.], [151., 99., 99.], [161., 99., 99.]],
                  [[12., 18., 18.], [12., 21., 21.], [14., 26., 26.], [19., 66., 66.], [ 26., 99., 99.], [158., 99., 99.], [160., 99., 99.], [155., 99., 99.]],
                  [[14., 24., 24.], [13., 26., 26.], [16., 56., 56.], [24., 99., 99.], [ 40., 99., 99.], [157., 99., 99.], [169., 99., 99.], [156., 99., 99.]],
                  [[14., 47., 47.], [17., 66., 66.], [22., 99., 99.], [29., 99., 99.], [ 51., 99., 99.], [187., 99., 99.], [180., 99., 99.], [162., 99., 99.]],
                  [[18., 99., 99.], [22., 99., 99.], [37., 99., 99.], [56., 99., 99.], [ 68., 99., 99.], [109., 99., 99.], [103., 99., 99.], [177., 99., 99.]],
                  [[24., 99., 99.], [35., 99., 99.], [55., 99., 99.], [64., 99., 99.], [ 81., 99., 99.], [104., 99., 99.], [113., 99., 99.], [192., 99., 99.]],
                  [[49., 99., 99.], [64., 99., 99.], [78., 99., 99.], [87., 99., 99.], [103., 99., 99.], [121., 99., 99.], [120., 99., 99.], [101., 99., 99.]],
                  [[72., 99., 99.], [92., 99., 99.], [95., 99., 99.], [98., 99., 99.], [112., 99., 99.], [100., 99., 99.], [103., 99., 99.], [199., 99., 99.]]])
                  
        if qualy < 1 : fact = 1
        if qualy > 99: fact = 99
        if qualy < 50:
            qualy = 5000 / fact
        else:
            qualy = 200 - 2*fact
        
        qZ = ((Z*qualy) + 50)/100
        qZ[qZ<1] = 1
        qZ[qZ>255] = 255
    
        return qZ