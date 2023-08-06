import cv2
import numpy as np

class adj:
	def __init__(self):
		pass
	
	def adj(sMap, bMap):
		sSum = 0
		bSum = 0
		
		if isinstance(sMap, list):
		    sMap = np.array(sMap)
                if isinstance(bMap, list):
                    bMap = np.array(bMap)

		(h, w) = sMap.shape[:2]
		
		for i in range(h):
			for j in range(w):
				sSum += sMap[i,j]
				bSum += bMap[i,j]
		
		sAVG = sSum / (h * w)
		bAVG = bSum / (h * w)
		diff = sAVG - bAVG
		
		bFin = np.add(bMap, diff)
		
		return bFin
