# -*- coding: utf-8 -*-
"""
Created on Thu Apr 02 19:28:28 2015

@author: Navegantes
"""

import huffcoder as h
import numpy as np
import cv2

class Encoder:
    def __init__(self, frame, qually, hufftables, Z, mode='444'):
        '''
        '''
        
        #TRATA AS DIMENSOES DA IMAGEM
        (Madj, Nadj, Dadj), self.img = h.adjImg(np.float32(frame))         #        imOrig = frame  #cv2.imread(filepath,1)        #self.filepath = filepath
        self.mode = mode
        self.hufftables = hufftables
        self.CRate = 0; self.Redunc = 0                #Taxa de compressão e Redundancia
        self.avgBits = 0; self.NumBits = 0             #Media de bits e numero de bits
        self.qually = qually                           #Qualidade
        self.M, self.N, self.D = frame.shape           #imOrig.shape          #Dimensões da imagem original
        self.r, self.c = [8, 8]                        #DIMENSAO DOS BLOCOS
        
        #NUMERO DE BLOCOS NA VERTICAL E HORIZONTAL
        self.nBlkRows = int(np.floor(Madj/self.r))
        self.nBlkCols = int(np.floor(Nadj/self.c))
        
        #GERA TABELA DE QUANTIZAÇÃO
        self.Z = Z
        #TRANSFORMA DE RGB PARA YCbCr
        self.Ymg = self.img #cv2.cvtColor(self.img, cv2.COLOR_BGR2YCR_CB)
        
        if self.D == 2:
            self.NCHNL = 1
        elif self.D == 3:
            self.NCHNL = 3
            
        self.seqhuff = self._run_()
                                
    def _run_(self):
        '''
        '''
        
        hf = h.HuffCoDec(self.hufftables)        #flnm = self.filepath.split('/')[-1:][0].split('.')[0] + '.huff'        #fo = open(flnm,'w')        #fo.write(str(self.Mo) + ',' + str(self.No) + ',' + str(self.Do) + ',' +         #         str(self.qually) + ',' + self.mode + '\n')
        outseq = []
        
#        dYmg = self.Ymg - 128
        dYmg = self.Ymg - 128
        r, c, chnl = self.r, self.c, self.NCHNL
        coefs = np.zeros((r, c, chnl))

        
        if self.mode == '444':
            for ch in range(chnl):
                DCant = 0
                
                seqhuff = ''        #nbits = self.NumBits
                
                for i in range(self.nBlkRows):
                    temp_seq=''
                    for j in range(self.nBlkCols):
                        
                        sbimg = dYmg[r*i:r*i+r, c*j:c*j+c, ch]     #Subimagens nxn
                #    TRANSFORMADA - Aplica DCT
                        coefs = cv2.dct(sbimg)
                #    QUANTIZAÇÃO/LIMIARIZAÇÃO
                        zcoefs = np.round( coefs/self.Z[:,:,ch] )      #Coeficientes normalizados - ^T(u,v)=arred{T(u,v)/Z(u,v)}
                #    CODIFICAÇÃO - Codigos de Huffman
                #  - FOWARD HUFF
                        seq = h.zigzag(zcoefs)                     #Gera Sequencia de coeficientes 1-D
                        hfcd = hf.fwdhuff(DCant, seq, ch)          #Gera o codigo huffman da subimagem
                        DCant = seq[0]
                        self.NumBits += hfcd[0]
                        temp_seq += hfcd[1]
                    seqhuff += temp_seq
                        
                #Salvar os codigos em arquivo
                #fo.write(seqhuff+'\n')
                outseq.append(seqhuff)
                
        elif self.mode == '420':
            
            if chnl == 1:
                Ymg = dYmg
            else:
                Y = dYmg[:,:,0]
                dims, CrCb = h.adjImg(downsample(dYmg[:,:,1:3], self.mode)[1])
                Ymg = [ Y, CrCb[:,:,0], CrCb[:,:,1] ]
                self.lYmg = Ymg
            for ch in range(chnl):
                DCant = 0
                
                seqhuff = ''        #nbits = self.NumBits
                
                if ch == 0: #LUMINANCIA
                    rBLK = self.nBlkRows
                    cBLK = self.nBlkCols
                else:       #CROMINANCIA
                    rBLK, cBLK = int(np.floor(dims[0]/self.r)), int(np.floor(dims[1]/self.c))
                
                for i in range(rBLK):
                    for j in range(cBLK):
                        sbimg = Ymg[ch][r*i:r*i+r, c*j:c*j+c]     #Subimagens nxn
                #    TRANSFORMADA - Aplica DCT
                        coefs = cv2.dct(sbimg)
                #    QUANTIZAÇÃO/LIMIARIZAÇÃO
                        zcoefs = np.round( coefs/self.Z[:,:,ch] )      #Coeficientes normalizados - ^T(u,v)=arred{T(u,v)/Z(u,v)}
                #    CODIFICAÇÃO - Codigos de Huffman - FOWARD HUFF
                        seq = h.zigzag(zcoefs)                     #Gera Sequencia de coeficientes 1-D
                        hfcd = hf.fwdhuff(DCant, seq, ch)          #Gera o codigo huffman da subimagem
                        DCant = seq[0]
                        self.NumBits += hfcd[0]
                        seqhuff += hfcd[1]          
                #Salvar os codigos em arquivo
                #fo.write(seqhuff + '\n')
                outseq.append(seqhuff)
                
        #fo.close()
        self.avgBits = (float(self.NumBits)/float(self.M*self.N))
        self.CRate = 24./self.avgBits
        self.Redunc = 1.-(1./self.CRate)
        #print '- Encoder Complete...'
        #return (self.CRate, self.Redunc, self.NumBits)
        return outseq
        
        
    def Outcomes(self):
        '''
        '''
        
        print '    :: Taxa de Compressao: %2.3f'%(self.CRate)
        print '    :: Redundancia de Dados: %2.3f' %(self.Redunc)
        print '    :: Numero total de bits: ', self.NumBits
        print '    :: Media de bits/Pixel: %2.3f' %(self.avgBits)
#End class Encoder
        
class Decoder:
    '''
    '''
    
    def __init__(self, huffcode, hufftables, Z, args):  #filename):
        '''
        '''
        
        #self.fl = open(filename,'r')        #header = hdr #self.fl.readline().split(',')                  #Lê cabeçalho
        self.SHAPE, self.qually, self.mode = args[0], args[1], args[2] #int(header[0]), int(header[1]), int(header[2]), int(header[3]), header[4][:-1]
        self.huffcodes = huffcode        #self.SHAPE = conf[0] #(self.Mo, self.No, self.Do)
        (self.M, self.N, self.D), self.imRaw = h.adjImg( np.zeros(self.SHAPE) )
        #NUMERO DE BLOCOS NA VERTICAL E HORIZONTAL
        self.R, self.C = [8,8]
        #NUMERO DE BLOCOS NA VERTICAL E HORIZONTAL
        self.nBlkRows = int(np.floor(self.M/self.R))
        self.nBlkCols = int(np.floor(self.N/self.C))
        #Gera Tabela de Qunatizaçao
        self.Z = Z
        self.hufftables = hufftables
        
        if self.D == 2:
            self.NCHNL = 1
        elif self.D == 3:
            self.NCHNL = 3
            
    def _run_(self):
        '''
        '''
        #print '- Running Mjpeg Decoder...'
        hf = h.HuffCoDec(self.hufftables)
        r, c, chnl = self.R, self.C, self.NCHNL
        Z = self.Z
        
        #hufcd = self.huffcodes#self.fl.readline()[:-1]
        if self.mode == '444':
            for ch in range(chnl):                #hufcd = self.fl.readline()[:-1]            #    print hufcd[0:20]
                nblk, seqrec = hf.invhuff(self.huffcodes[ch], ch)
                for i in range(self.nBlkRows):
                    for j in range(self.nBlkCols):
#                        print("sec " + str(len(seqrec)))
#                        print("index " + str(i*self.nBlkCols + j))
                        blk = h.zagzig(seqrec[i*self.nBlkCols + j])
                        self.imRaw[r*i:r*i+r, c*j:c*j+c, ch] = np.round_( cv2.idct( blk*Z[:,:,ch] ))
                        
        elif self.mode == '420':
            #import math as m
            if chnl == 1:
                rYmg = self.imRaw
            else:                #Y = self.imRaw[:,:,0]
                Y = np.zeros( (self.M, self.N) )
                dims, CrCb = h.adjImg( downsample(np.zeros( (self.M, self.N, 2) ), self.mode)[1] )
                rYmg = [ Y, CrCb[:,:,0], CrCb[:,:,1] ]
                
            for ch in range(chnl):
                #hufcd = self.fl.readline()[:-1]
                if ch == 0:
                    rBLK = self.nBlkRows
                    cBLK = self.nBlkCols
                else:
                    rBLK, cBLK = int(np.floor(dims[0]/self.R)), int(np.floor(dims[1]/self.C))
            #    print hufcd[0:20]
                nblk, self.seqrec = hf.invhuff(self.huffcodes[ch], ch)
                for i in range(rBLK):
                    for j in range(cBLK):
                        blk = h.zagzig(self.seqrec[i*cBLK + j])
                        #print rYmg[ch][r*i:r*i+r, c*j:c*j+c].shape, ch, i, j
                        rYmg[ch][r*i:r*i+r, c*j:c*j+c] = np.round_( cv2.idct( blk*Z[:,:,ch] ))
            # UPSAMPLE
            if chnl == 1:
                self.imRaw = rYmg #[:self.Mo, : self.No]
            else:
                self.imRaw[:,:,0] = rYmg[0]
                self.imRaw[:,:,1] = upsample(rYmg[1], self.mode)[:self.M, :self.N]
                self.imRaw[:,:,2] = upsample(rYmg[2], self.mode)[:self.M, :self.N]
        
        #self.fl.close()
        
#        imrec = cv2.cvtColor((self.imRaw[:self.Mo, :self.No]+128), cv2.COLOR_YCR_CB2BGR)
#        imrec = self.imRaw[:self.Mo, :self.No]+128
        imrec = self.imRaw+128.0
#        imrec[imrec>255.0]=255.0
#        imrec[imrec<0.0]=0.0
                                
        #print 'Mjpeg Decoder Complete...'
        
        return imrec

def downsample(mat, mode):
    '''
    '''
        
    import math as m
    M, N, D = mat.shape
    #M, N = mat.shape
    #D = mat[0,0].shape[0]
    ndims = ( int(m.ceil(M/2)), int(m.ceil(N/2)))
    newmat = np.zeros((ndims[0], ndims[1]))
    #aux = np.zeros((m.ceil(M/2), N, D))
    
    if mode == '420':
        newmat = mat[::2,::2]
    elif mode == '422':
        pass
        
    #dims, newmat = h.adjImg(newmat)
    return ndims, newmat
    #return h.adjImg(newmat)
    
def upsample(mat, mode):
    '''
    '''
    
    M, N = mat.shape
    newmat = np.zeros((M*2, N*2))
    
    if mode == '420':
        newmat[::2, ::2] = mat
        newmat[::2, 1::2] = mat
        newmat[1::2, :] = newmat[::2, :]
#        newmat[1::2, ::2] = mat
#        newmat[1::2, 1::2] = mat
    elif mode == '422':
        pass
    
    return newmat
    
    
