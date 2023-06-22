import cv2
import numpy as np
import random
from algorithm import algorithm
from techniques import techniques
from SNN import SNN

class MPImodifiers:
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
		factor = (1 - psi)
		delta_new = np.multiply(delta_new, factor)
		return delta_new
	
	def afm(sMaps1, sMaps2, sMaps3, sMaps4):
		maps = []
		for j in range(len(sMaps4)):
			div4 = j // 4
			div2 = j // 2
			rMap1 = np.add(sMaps1[div4], sMaps2[div4])
			rMap2 = np.add(sMaps3[div2], sMaps4[j])
			rMap = np.add(rMap1, rMap2)
			aMap = np.multiply(rMap, 0.25)
			maps.append(aMap)	
		return maps
	
	def mfm(image, sMaps4):
		maps = []
		for j in range(len(sMaps4)):
			mMap = cv2.bitwise_and(image, image, sMaps4[j])
			maps.append(mMap)
		return mMaps
	
	def gradient(error, alpha):
		div = error / 64.0
		sqrt = np.sqrt(div)
		factor = 2 * sqrt
		beta = alpha * factor
		return beta
		
	def sARR(img1, img2): # similarity coefficients
		if isinstance(img1, list):
			img1 = np.array(img1)
		if isinstance(img2, list):
			img2 = np.array(img2)
		(h, w) = img1.shape[:2]
		scTOT = 0
		avg = (sum(img1, img2) / 2.0) + 0.5
		avgLN = np.log(avg)
				
		characteristic1 = avgLN / np.log(255)
			
		diff = abs(img1 - img2)
		characteristic2 = np.sqrt(diff / 256)
			
		val1 = characteristic1 - characteristic2
		val2 = val1.tolist()
		sc = []
		for i in range(h):
			for j in range(w):
				val3 = techniques.elu(val2[i][j])
				sc.append(val3)
				
		return sc
	
	def image_masks(img1, img2, img3, img4, img5):
		if isinstance(img1, list):
			img1 = np.array(img1)
		sc2 = MPImodifiers.sARR(img1, img2)
		sc3 = MPImodifiers.sARR(img1, img3)
		sc4 = MPImodifiers.sARR(img1, img4)
		sc5 = MPImodifiers.sARR(img1, img5)
		sc1 = []
		for i in range(len(sc2)):
			val = sc2[i] + sc3[i] + sc4[i] + sc5[i]
			sc1.append(val)
		
		new_img = []
		(h, w) = img1.shape[:2]
		for i in range(h):
			new_row = []
			for j in range(w):
				place = (i * w) + j
				vals = img1[i,j] * sc1[place]
				new_row.append(vals)
			new_img.append(new_row)
		new_img = np.array(new_img)
		return new_img
	
	def frame_differences(img1, img2, img3, img4, img5):
            if isinstance(img1, list):
                img1 = np.array(img1)
            if isinstance(img2, list):
                img2 = np.array(img2)
            if isinstance(img3, list):
                img3 = np.array(img3)
            if isinstance(img4, list):
                img4 = np.array(img4)
            if isinstance(img5, list):
                img5 = np.array(img5)
            change1 = np.add(img2, img3)
            change2 = np.add(img4, img5)
            change = np.multiply(np.add(change1, change2), 0.25)
            new = np.square(np.subtract(img1, change))
            return new
	
	def dropout(filter_inputs, layers, threshold):
		new_layers = []
		indices = []
		for k in range(len(layers) - 1): # iterates through all attached layers
			index = []
			for i in range(len(layers[k])): # iterates through all the nodes
				if random.random() > threshold:
					index.append(i) # adds indices
			indices.append(index)
		indices.append([0, 1, 2, 3])
		
		neuralDropout = []
		for k in range(len(layers) - 1):
			neuralValues = []
			for node in indices[k]:
				element = []
				weights = []
				for weight in indices[k + 1]:
					weights.append(layers[k][node][0][weight])
				element.append(weights)
				element.append(layers[k][node][1])
				neuralValues.append(element)
			neuralDropout.append(neuralValues)
		
		neuralDropout.append(layers[-1])
		return neuralDropout
	
	def determinants(convolutional_layers):
		for i in range(len(convolutional_layers)):
			for k in range(len(convolutional_layers[i])):
				det = 1
				try:
					det = np.linalg.det(convolutional_layers[i][k])
				except:
					pass
				normalized = np.sqrt(abs(det))
				convolutional_layers[i][k] = np.divide(convolutional_layers[i][k], normalized)
		return convolutional_layers
	
	def learning_rate(convolutional_layer):
		updated_layer = convolutional_layer.copy()
		for i in range(len(convolutional_layer)):
			det = 1
			try:
				det = np.linalg.det(convolutional_layer[i])
			except:
				pass
			rate = np.log(abs(det))
			sqrt = np.sqrt(rate)
			recip = 1.0 / sqrt
			updated_layers[i][k] = recip
		
		return updated_layer	
