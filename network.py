import numpy as np
import cv2
import random
import math
from algorithm import algorithm
from techniques import techniques
from SNN import SNN
from MPImodifiers import MPImodifiers

class network:
	def __init__(self):
		pass
		
	def generate_layer(prev, nodes):
		layer = []
		for i in range(prev):
			weights = []
			for j in range(nodes):
				weight = random.uniform(-1, 1)
				weights.append(weight)
			bias = random.uniform(-1, 1)
			layer.append([weights, bias])
		return layer

	def generate_filters(num, size): # generates size x size filters
		layer1 = []
		for j in range(5):
			section1 = []
			for i in range(num):
				filter1 = np.random.rand(size, size)
				filter2 = filter1.tolist()
				section1.append(filter2)
			layer1.append(section1)
		return layer1
	
	def parallel_filters(num, size):
		layer1 = []
		for i in range(num):
			filter1 = np.random.rand(size, size)
			filter2 = filter1.tolist()
			layer1.append(filter2)
		return layer1

	def generate_feature_maps(layer, image): # takes in a layer of node filters and images and creates feature maps
		featureMaps1 = []
		for i in range(len(layer)):
			pad_val = len(layer[i]) // 2
			fMap = algorithm.convolution(layer[i], image, pad_val)
			fMapImage = np.array(fMap)
			featureMaps1.append(fMapImage)
		return featureMaps1

	def max_pooling(fMap): # max pooling (by 2)
		if isinstance(fMap, list):
			fMap = np.array(fMap)
		pooledMap = []
		(h, w) = fMap.shape[:2]
		for i in range(0, h - 1, 2):
			pooledRow = []
			for j in range(0, w - 1, 2):
				val1 = fMap[i,j]
				val2 = fMap[(i + 1),(j)]
				val3 = fMap[(i),(j + 1)]
				val4 = fMap[(i + 1),(j + 1)]
				val = max(val1, val2, val3, val4)
				pooledRow.append(val)
			
			pooledMap.append(pooledRow)
		pooledImage = np.array(pooledMap)
		return pooledImage

	def pooled_maps(maps, v = 0): # max pooling but on a list of maps
		pooled = []
		for i in range(len(maps)):
			if v == 0:
				pool = algorithm.max_pooling(maps[i])
				pooled.append(pool)
			else:
				pool = network.max_pooling(maps[i])
				pooled.append(pool)
		return pooled

	def leaky_relu(x):
		if x > 0:
			return x
		else:
			return (0.2 * x)

	def connector_layer(fMaps):
		if isinstance(fMaps[0], list):
			for i in range(len(fMaps)):
				fMaps[i] = np.array(fMaps[i])
		vals = []
		(h, w) = fMaps[0].shape[:2]
		for i in range(h):
			for j in range(w):
				tot = 0
				count = 0
				for k in range(len(fMaps)):
					tot += fMaps[k][i][j]
					count += 1
				val = tot / count
				vals.append(val)
		return vals

	def forward_pass(layer, vals, forward, activation):
		new = []
		for i in range(len(forward)):
			tot = 0
			for j in range(len(layer)):
				added = (layer[j][0][i] * vals[j]) + layer[j][1]
				tot += added
			if activation == 0:
				activ = network.leaky_relu(tot)
				new.append(activ)
			else:
				activ = network.sigmoid(tot)
				new.append(activ)
		return new

	def sigmoid(x):
		if x == 0:
			x = 0.001
		sign = -1 if x < 0 else 1
		val = np.log(abs(x)) * sign
		exp = ((math.e) ** (-1 * val))
		denom = 1 + exp
		frac = 1.0 / denom
		return frac

	def mse(actuals, coords):
		loss = []
		for i in range(5):
			tot = []
			for j in range(4):
				added = (actuals[i][j] - coords[i][j]) ** 2
				tot.append(added)
			loss.append(tot)
		return loss

	def mseINDIV(actuals, coords):
		loss = []
		for j in range(4):
			added = (actuals[j] - coords[j]) ** 2
			loss.append(added)
		return loss

	def backprop(layer, vals, activation, alpha, FP):
		propagated = []
		for i in range(len(layer)):
			propagates = 0
			for j in range(len(FP)):
				added = FP[j] * vals[j]
				propagate = added * layer[i][0][j]
				propagates += propagate
				change = 0
				if activation == 0:
					activ = network.leaky_relu(added)
					change = activ * alpha
				else:
					activ = network.sigmoid(added)
					change = activ * alpha
				layer[i][0][j] -= change
			propagated.append(propagates)
		return (layer, propagated)
		
	def backpropNESTEROV(layer, vals, activation, alpha, FP, layerPREV, psi):
		propagated = []
		for i in range(len(layer)):
			propagates = 0
			for j in range(len(FP)):
				added = FP[j] * vals[j]
				propagate = added * layer[i][0][j]
				propagates += propagate
				changePREV = layerPREV[i][0][j] - layer[i][0][j]
				change = 0
				if activation == 0:
					activ = network.leaky_relu(added)
					change = activ * alpha
				else:
					activ = network.sigmoid(added)
					change = activ * alpha
				changeNEW = MPImodifiers.nesterovMomentum(changePREV, change, psi)
				layer[i][0][j] -= changeNEW
			propagated.append(propagates)
		return (layer, propagated)

	def backpropMACLAURIN(layer, vals, activation, alpha, FP, layerPREVS, psi, stretch):
		propagated = []
		for i in range(len(layer)):
			propagates = 0
			for j in range(len(FP)):
				added = FP[j] * vals[j]
				propagate = added * layer[i][0][j]
				propagates += propagate
				changes = []
				for k in range(layerPREVS):
					changePREV = layerPREVS[k][i][0][j] - layer[i][0][j]
					changes.append(changePREV)
				change = 0
				if activation == 0:
					activ = network.leaky_relu(added)
					change = activ * alpha
				else:
					activ = network.sigmoid(added)
					change = activ * alpha
				changeNEW = MPImodifiers.maclaurin(changePREV, change, psi)
				layer[i][0][j] -= changeNEW
			propagated.append(propagates)
		return (layer, propagated)
	
	def add(matrices): # basic add matrix function for ease of access
		tot = []
		for i in range(len(matrices[0])):
			row = []
			for j in range(len(matrices[0][0])):
				count = 0
				for k in range(len(matrices)):
					count += matrices[k][i][j]
				row.append(count)
			tot.append(row)
		return tot
		
	def subtract(matrix, num):
		new = []
		for i in range(len(matrix)):
			row = []
			for j in range(len(matrix[0])):
				val = matrix[i][j] - num
				row.append(val)
			new.append(row)

	def multiply(matrix, factor): # basic multiply matrix function for ease of access
		for i in range(len(matrix)):
			for j in range(len(matrix[0])):
				matrix[i][j] = matrix[i][j] * factor
		return matrix

	def same_padding(image, pad): # same_padding amount of pad
		arr = []
		(h, w) = image.shape[:2]
		for i in range(h):
			lst = []
			for ext in range(pad): # beginning of list
				lst.append(image[i, 0])
			for j in range(w): # end of list
				lst.append(image[i, j])
			if i == 0 or i == h - 1: # top or bottom of array
				for ext in range(pad):
					arr.append(lst)
			
			arr.append(lst)
		
		return arr

	def anti_pool(image, hG, wG):
		if isinstance(image, list):
			image = np.array(image)
		(hI, wI) = image.shape[:2]
		antipooled = []
		for i in range(hI):
			row = []
			for j in range(wI):
				row.append(image[i][j])
				row.append(image[i][j])
			if wG % 2 == 1:
				row.append(image[i][-1])
			antipooled.append(row)
			antipooled.append(row)
		if hG % 2 == 1:
			new_row = antipooled[-1]
			antipooled.append(new_row)
		return antipooled
	
	def feature_maps_same(layer, section_maps):
		section_generated_maps = []
		for i in range(len(layer)):
			pad_val = len(layer[i]) // 2
			generated_map = algorithm.convolution(layer[i], section_maps[i], pad_val)
			section_generated_maps.append(generated_map)
		return section_generated_maps
	
	def feature_maps_double(layer, section_maps):
		section_generated_maps = []
		for i in range(len(section_maps)):
			pad_val = len(layer[i]) // 2
			filter_place1 = i * 2
			filter_place2 = filter_place1 + 1
			generated_map1 = algorithm.convolution(layer[filter_place1], section_maps[i], pad_val)
			generated_map2 = algorithm.convolution(layer[filter_place2], section_maps[i], pad_val)
			section_generated_maps.append(generated_map1)
			section_generated_maps.append(generated_map2)
		return section_generated_maps
	
	def signed_ln(arr):
		if isinstance(arr, list):
			arr = np.array(arr)
		(h, w) = arr.shape[:2]
		new = []
		for i in range(h):
			row = []
			for j in range(w):
				val = arr[i,j]
				absval = abs(val)
				sign = val / absval
				ln = np.log(absval)
				res = ln * sign
				row.append(res)
			new.append(row)
		return new
	
	def direction(prev, next):
		diff = prev - next
		c = diff / next
		
		absC = abs(c)
		sign = absC / c
		sqrt = np.sqrt(absC)
		res = sqrt * sign
		return res
