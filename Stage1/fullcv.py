import csv
import numpy as np
import cv2

class cv:
	def __init__(self):
		pass
	
	
	def acv(F, I):
		if isinstance(I, list):
			I = np.array(I)
		if isinstance(F, list):
			F = np.array(F)
		
		(h, w) = I.shape[:2]
		(hF, wF) = F.shape[:2]
		
		A = []
		N = []
		for i in range(h + hF - 1):
			AR = []
			NR = []
			for j in range(w + wF - 1):
				AR.append(0)
				NR.append(0)
			A.append(AR)
			N.append(NR)
		
		areaF = 0
		for i in range(hF):
			for j in range(wF):
				areaF += F[i,j]
		AVG = np.divide(I, areaF)
		hS = hF // 2
		wS = wF // 2
		
		for i in range(h):
			for j in range(w):
				Ival = AVG[i,j]
				for k in range(hF):
					for l in range(wF):
						r = float(Ival)
						
						hP = i + k
						wP = j + l
						N[hP][wP] += 1
						A[hP][wP] += r


		R = np.divide(A, N)
		RA = R.tolist()
		RSP = RA[(hS):(h + hS)][(wS):(w + wS)]
		
		return RSP
