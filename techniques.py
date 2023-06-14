import cv2
import numpy as np
import math

class techniques:
	def __init__(self, A, B, C, D, E):
		self.A = A
		self.B = B
		self.C = C
		self.D = D
		self.E = E
	
	def elu(self, x):
		if x <= 0:
			return ((math.e ** x) - 1)
		else:
			return x
	
	def s(self, img1, img2): # similarity coefficients
		(h, w) = img1.shape[:2]
		scTOT = 0
		for i in range(h):
			for j in range(w):
				avg = sum(img1[i][j], img2[i][j]) / 2.0
				avgLN = np.log(avg)
				
				characteristic1 = avgLN / np.log(255)
				
				diff = abs(img1[i][j] - img2[i][j])
				characteristic2 = math.sqrt(diff / 256)
				
				val1 = characteristic1 - characteristic2
				
				val2 = elu(val1)
				
				scTOT += val2
		
		sc = float(scTOT / h / w)
	
	def correction(self, image, section): # applies image correction
		# assumes that correctionDNN has already been applied and does selective dimming
		(h, w) = image.shape[:2]
		min_val = 256
		for i in range(h):
			for j in range(w):
				if image[i][j] > 50 and image[i][j] < min_val:
					min_val = image[i][j]
		
		for i in range(h):
			for j in range(w):
				if image[i][j] < min_val:
					image[i][j] = 0
				else:
					image[i][j] = image[i][j] - min_val
		
		max_val = 0
		(h1, w1) = section.shape[:2]
		for i in range(h1):
			for j in range(w1):
				if section[i][j] > max_val:
					max_val = section[i][j]
		
		denom = max_val - 25 - min_val
		factor = 255 / float(denom)
		
		for i in range(h):
			for j in range(w):
				image[i][j] = image[i][j] * factor
		
		kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
		
		matrix = algorithm.convolution(kernel, image, 1)
		arr = np.array(matrix)
		borders = cv2.dilate(arr, None, 2)
		
		pixelSUM = 0
		pixelCOUNT = 0
		
		for i in range(h):
			for j in range(w):
				if borders[i][j] > image[i][j]:
					image[i][j] = 0
				else:
					image[i][j] = image[i][j] - borders[i][j]
				if image[i][j] > 215:
					pixelSUM += image[i][j]
					pixelCOUNT += 1
		
		pixelAVG = pixelSUM / pixelCOUNT
		denom1 = pixelAVG - 170
		factor1 = 70 / denom1
		
		for i in range(h):
			for j in range(w):
				sub = image[i][j] - 170
				new_sub = sub * factor1
				new = new_sub + 170
				
				if new < 0:
					image[i][j] = 0
				else:
					image[i][j] = new
		
		return image
	
	def plane_recurrence(self): # applies plane recurrence between the filters of a specific node layer
		detA = np.linalg.det(self.A)
		detB = np.linalg.det(self.B)
		detC = np.linalg.det(self.C)
		detD = np.linalg.det(self.D)
		detE = np.linalg.det(self.E)
		matrices = [self.A, self.B, self.C, self.D, self.E]
		dets = [detA, detB, detC, detD, detE]
		
		factors = []
		for l in range(5):
			factor = 0
			for k in range(5):
				addend = ((-0.5) ** abs(l-k)) * dets[k]
				factor += addend
			factors.append(factor)
		
		h = len(self.A)
		w = len(self.A[0])
		
		for i in range(h):
			for j in range(w):
				for k in range(5):
					matrices[k][i][j] = matrices[k][i][j] * factors[k] / dets[k]
		
		return matrices
