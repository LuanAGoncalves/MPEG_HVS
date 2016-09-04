# -*- coding: utf-8 -*-
"""
Created on Thu Apr 02 19:28:28 2015

@author: Navegantes
"""

import huffcoder as h
import numpy as np
import cv2

class Encoder:
    
    def __init__(self, filepath, qually, mode='444'):
        '''
        '''
        imOrig = cv2.imread(filepath,1)
        self.filepath = filepath
        self.mode = mode
        #Taxa de compressão e Redundancia
        self.CRate = 0; self.Redunc = 0
        self.avgBits = 0
        #Qualidade
        self.qually = qually
        #Dimensões da imagem original
        self.Mo, self.No, self.Do = imOrig.shape
        self.r, self.c = [8, 8]       #DIMENSAO DOS BLOCOS
        #TRATA AS DIMENSOES DA IMAGEM
        (self.M, self.N, self.D), self.img = h.adjImg(imOrig)
        #NUMERO DE BLOCOS NA VERTICAL E HORIZONTAL
        self.nBlkRows = int(np.floor(self.M/self.r))
        self.nBlkCols = int(np.floor(self.N/self.c))
        #Gera Tabela de Qunatizaçao
        self.Z = h.genQntb(self.qually)
        #TRANSFORMA DE RGB PARA YCbCr
        self.Ymg = cv2.cvtColor(self.img, cv2.COLOR_BGR2YCR_CB)
        self.NumBits = 0
        if self.Do == 2:
            self.NCHNL = 1
        elif self.Do == 3:
            self.NCHNL = 3
            
#        self.OUTCOMES = self._run_()
        self._run_()

    def _run_(self):
        '''
        '''
        
        print '- Running Encoder...'
        hf = h.HuffCoDec()
        flnm = self.filepath.split('/')[-1:][0].split('.')[0] + '.huff'
        fo = open(flnm,'w')
        fo.write(str(self.Mo) + ',' + str(self.No) + ',' + str(self.Do) + ',' + 
                 str(self.qually) + ',' + self.mode + '\n')
        
        dYmg = self.Ymg - 128
        r, c, chnl = self.r, self.c, self.NCHNL
        coefs = np.zeros((r, c, chnl))
        seqhuff = ''
        #nbits = self.NumBits
        if self.mode == '444':
            for ch in range(chnl):
                DCant = 0
                for i in range(self.nBlkRows):
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
                        seqhuff += hfcd[1]          
                #Salvar os codigos em arquivo
                fo.write(seqhuff+'\n')
                seqhuff = ''
                
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
                fo.write(seqhuff + '\n')
                seqhuff = ''
        
        fo.close()

        self.avgBits = (float(self.NumBits)/float(self.Mo*self.No))
        self.CRate = 24./self.avgBits
        self.Redunc = 1.-(1./self.CRate)
        print '- Encoder Complete...'
        #return (self.CRate, self.Redunc, self.NumBits)
        
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
    
    def __init__(self, filename):
        '''
        '''
        
        self.fl = open(filename,'r')        
        header = self.fl.readline().split(',')                  #Lê cabeçalho
        self.Mo, self.No, self.Do, self.qually, self.mode = int(header[0]), int(header[1]), int(header[2]), int(header[3]), header[4][:-1]
        self.SHAPE = (self.Mo, self.No, self.Do)
        (self.M, self.N, self.D), self.imRaw = h.adjImg( np.zeros(self.SHAPE) )
        #NUMERO DE BLOCOS NA VERTICAL E HORIZONTAL
        self.R, self.C = [8,8]
        #NUMERO DE BLOCOS NA VERTICAL E HORIZONTAL
        self.nBlkRows = int(np.floor(self.M/self.R))
        self.nBlkCols = int(np.floor(self.N/self.C))
        #Gera Tabela de Qunatizaçao
        self.Z = h.genQntb(self.qually)
        
        if self.Do == 2:
            self.NCHNL = 1
        elif self.Do == 3:
            self.NCHNL = 3
            
    def _run_(self):
        '''
        '''
        print '- Running Decoder...'
        hf = h.HuffCoDec()
        r, c, chnl = self.R, self.C, self.NCHNL
        Z = self.Z
        
        if self.mode == '444':
            for ch in range(chnl):
                hufcd = self.fl.readline()[:-1]
            #    print hufcd[0:20]
                nblk, seqrec = hf.invhuff(hufcd, ch)
                for i in range(self.nBlkRows):
                    for j in range(self.nBlkCols):
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
                hufcd = self.fl.readline()[:-1]
                if ch == 0:
                    rBLK = self.nBlkRows
                    cBLK = self.nBlkCols
                else:
                    rBLK, cBLK = int(np.floor(dims[0]/self.R)), int(np.floor(dims[1]/self.C))
            #    print hufcd[0:20]
                nblk, self.seqrec = hf.invhuff(hufcd, ch)
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
        
        self.fl.close()
        
        imrec = cv2.cvtColor((self.imRaw[:self.Mo, :self.No]+128), cv2.COLOR_YCR_CB2BGR)
        imrec[imrec>255]=255
        imrec[imrec<0]=0
        
        print 'Decoder Complete...'
        
        return np.uint8(imrec)

def downsample(mat, mode):
    '''
    '''
        
    import math as m
    M, N, D = mat.shape
    #M, N = mat.shape
    #D = mat[0,0].shape[0]
    ndims = ( m.ceil(M/2), m.ceil(N/2) )
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