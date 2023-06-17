import cv2
import numpy as np
from algorithm import algorithm
from techniques import techniques
from SNN import SNN
from network import network

class modifications:
	def __init__(self):
		pass
	
	def nesterovMomentum(delta_prev, delta, psi):
		delta_new = np.add(delta, np.multiply(delta_prev, psi))
		return delta_new
	
	def maclaurin(deltas, psi, stretch):
		delta_new = 0
		if len(deltas) < stretch:
			for i in range(len(deltas)):
				diff = abs(len(deltas) - i - 1)
				factor = (psi) ** (diff)
				new = np.multiply(deltas[i], factor)
				delta_new = np.add(delta_new, new)
		else:
			for i in range(stretch):
				diff = abs(stretch - i - 1)
				factor = (psi) ** (diff)
				place = i - stretch
				new = np.multiply(deltas[place], factor)
				delta_new = np.add(delta_new, new)
		return delta_new
	
	def afm(sMaps):
		aMaps = []
		for i in range(5):
			sMaps1 = sMaps[0][i]
			sMaps2 = sMaps[1][i]
			sMaps3 = sMaps[2][i]
			sMaps4 = sMaps[3][i]
			
			maps = []
			for j in range(len(sMaps4)):
				div4 = j // 4
				div2 = j // 2
				rMap = np.add(sMaps1[div4], sMaps2[div4], sMaps3[div2], sMaps4[j])
				aMap = np.multiply(rMap, 0.25)
				maps.append(aMap)
			aMaps.append(maps)
		return aMaps
	
	def mfm(images, sMaps4):
		mMaps = []
		for i in range(5):
			maps = []
			for j in range(len(sMaps4) // 5):
				mMap = cv2.bitwise_and(images[i], images[i], sMaps4[i][j])
				maps.append(mMap)
			mMaps.append(maps)
		return mMaps
	
	def gradient(error, alpha):
		div = error / 64.0
		sqrt = np.sqrt(div)
		factor = 2 * sqrt
		beta = alpha * factor
		return beta
		
	def correction_gradient(error1, error2, alpha):
		diff = error1 - error2
		percent = diff / error2
		sign = -1 if percent < 0 else 1
		gamma = np.sqrt(percent) * sign
		nu = modifications.gradient(error2, alpha) * gamma
		return nu
	
	
