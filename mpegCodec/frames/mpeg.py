# -*- coding: utf-8 -*-
"""
Created on Sat May 16 20:08:00 2015

@author: luan
"""

import numpy as np
import cv2
from math import log

class Pframe:
	def __init__ (self, pastfr, currentfr, sspace, search = 0):
		self.mbr, self.mbc = [16, 16]
		self.pastfr = pastfr
		self.currentfr = currentfr
		self.sspace = sspace
		self.search = search
		self.motionVec, self.pframe = self.forewardPrediction(self.pastfr, self.currentfr)
		
	def resize (self, frame):
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

	def forewardPrediction (self, pastfr, currentfr):
		rR, rC, rD = currentfr.shape
		pastfrBackG = np.zeros((rR+2*self.sspace,rC+2*self.sspace, rD), np.float32)
		pastfrBackG[self.sspace:self.sspace+rR,self.sspace:self.sspace+rC] = pastfr
		pastfr = pastfrBackG
		result = np.zeros(currentfr.shape, np.float32)
		motioVec = []
		
		if self.search == 0:	# Full search.
			for i in range(0, rR, self.mbr):
				for j in range (0, rC, self.mbc):
					mad = 256.0
					x = y = 0
					for a in range (i-self.sspace, i+self.sspace):
						for b in range (j-self.sspace, j+self.sspace):
							mae = (1.0/float(self.mbr*self.mbc))*(np.sum(cv2.absdiff(self.currentfr[i:i+self.mbr,j:j+self.mbc, 0], pastfr[self.sspace+a:self.sspace+a+self.mbr, self.sspace+b:self.sspace+b+self.mbc,0])))
							if mad >= mae:
								mad = mae
								x = a-i
								y = b-j
					motioVec.append((int(x), int(y)))
					result[i:i+self.mbr, j:j+self.mbc] = currentfr[i:i+self.mbr, j:j+self.mbc] - pastfr[self.sspace+i+x:self.sspace+i+x+self.mbr, self.sspace+j+y:self.sspace+j+y+self.mbc]
		
		elif self.search == 1:	# Parallel.
			for i in range(0, rR, self.mbr):
				for j in range (0, rC, self.mbc):
					mad = 256.0
					x = y = 0
					s = int(np.floor(2**(log(self.sspace,2))))
     
					while s >= 1:
						for a in [i-s, i, i+s]:
							for b in [j-s, j, j+s]:
								mae = (1.0/float(self.mbr*self.mbc))*(np.sum(cv2.absdiff(self.currentfr[i:i+self.mbr,j:j+self.mbc, 0], pastfr[self.sspace+a:self.sspace+a+self.mbr, self.sspace+b:self.sspace+b+self.mbc,0])))
								if mad >= mae:
									mad = mae
									x = a-i
									y = b-j
						s = s/2
					motioVec.append((int(x), int(y)))
					result[i:i+self.mbr, j:j+self.mbc] = currentfr[i:i+self.mbr, j:j+self.mbc] - pastfr[self.sspace+i+x:self.sspace+i+x+self.mbr, self.sspace+j+y:self.sspace+j+y+self.mbc]
					
		return [motioVec, result]
		
class Bframe:
	def __init__(self, pastfr, currentfr, postfr, sspace, search = 0):
		self.mbr, self.mbc = [16, 16]
		self.search = search
		self.sspace = sspace
		self.pastfr = pastfr
		self.currentfr = currentfr
		self.postfr = postfr
		self.motionVec, self.bframe = self.bidirectionalPrediction (self.pastfr, self.currentfr, self.postfr)
		
	def bidirectionalPrediction (self, pastfr, currentfr, postfr):
		rR, rC, rD = currentfr.shape
		interpolDif = np.zeros(currentfr.shape, np.float32)
		interpolativeMVec = []
		
		aux = Pframe(postfr, currentfr, self.sspace, self.search)
		backwardMVec = aux.motionVec
		backwardDif = aux.pframe
		
		aux = Pframe(pastfr, currentfr, self.sspace, self.search)
		forewardMVec = aux.motionVec
		forewardDif = aux.pframe
		
		result = np.zeros(currentfr.shape, np.float32)
		motionVec = []
		
		count = 0
#		interpolDif = (backwardDif+forewardDif)/2.0
		for i in range (0, rR, self.mbr):
			for j in range (0, rC, self.mbc):
				b0, b1 = backwardMVec[count]
				f0, f1 = forewardMVec[count]
				interpolDif[i:i+self.mbr, j:j+self.mbc] = (forewardDif[i:i+self.mbr, j:j+self.mbc]+backwardDif[i:i+self.mbr, j:j+self.mbc])/2.0
				interpolativeMVec.append((int(f0),int(f1),int(b0),int(b1)))
#				count += 1
				
#		count = 0
				
#		for i in range (0, rR, self.mbr):
#			for j in range (0, rC, self.mbc):
#				b0, b1 = backwardMVec[count]
#				f0, f1 = forewardMVec[count]
				Eb = self.entropy(backwardDif[i:i+self.mbr,j:j+self.mbc,0])
				Ei = self.entropy(interpolDif[i:i+self.mbr,j:j+self.mbc,0])
				Ef = self.entropy(forewardDif[i:i+self.mbr,j:j+self.mbc,0])

				aux = [Eb, Ei ,Ef]
				w = aux.index(min(aux))
				if w == 0: 		#b
					result[i:i+self.mbr, j:j+self.mbc] = backwardDif[i:i+self.mbr,j:j+self.mbc]
					motionVec.append((count,'b',backwardMVec[count][0],backwardMVec[count][1]))
					
				elif w == 1:	# i
					result[i:i+self.mbr, j:j+self.mbc] = interpolDif[i:i+self.mbr,j:j+self.mbc]
					motionVec.append((count,'i',interpolativeMVec[count][0],interpolativeMVec[count][1],interpolativeMVec[count][2],interpolativeMVec[count][3]))
						
				elif w == 2:	#f
					result[i:i+self.mbr, j:j+self.mbc] = forewardDif[i:i+self.mbr,j:j+self.mbc]
					motionVec.append((count,'f',forewardMVec[count][0],forewardMVec[count][1]))
				count += 1
				motionVec.sort(key=lambda tup: tup[0])
				
		return [motionVec, result]
				
	def entropy(self, img):

#		M, N = img.shape
		h = np.array( self.histo(img) )
		Pr = h/np.float(self.mbr*self.mbc)
		H = 0
		for p in Pr:
			if p != 0:
				H += p*np.log2(p)
	    
		return -1*H

	def histo(self, img):
		'''
		Gera o Histograma de uma imagem.
		plot=True para plotar o histograma
		'''
	    
		img = np.uint8(img)
		h = np.zeros((256))
		for tom in img.flatten():
			h[tom]=h[tom]+1

		return h