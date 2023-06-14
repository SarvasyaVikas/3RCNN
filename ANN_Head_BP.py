import numpy as np
import cv2
import random
import math
from algorithm import algorithm
from techniques import techniques
from SNN import SNN

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

def generate_feature_maps(layer, images): # takes in a layer of node filters and images and creates feature maps
	featureMaps1 = []
	for j in range(len(images)):
		for i in range(len(layer)):
			pad_val = len(layer[i]) // 2
			fMap = algorithm.convolution(layer[i], images[j], pad_val)
			fMapImage = np.array(fMap)
			featureMaps1.append(fMapImage)
	return featureMaps1

def max_pooling(fMap): # max pooling (by 2)
	pooledMap = []
	(h, w) = fMap.shape[:2]
	for i in range(0, h - 1, 2):
		pooledRow = []
		for j in range(0, w - 1, 2):
			val = max(fMap[i][j], fMap[i + 1][j], fMap[i][j + 1], fMap[i + 1][j + 1])
			pooledRow.append(val)			
		pooledMap.append(pooledRow)
	return pooledMap

def pooled_maps(maps): # max pooling but on a list of maps
	pooled = []
	for i in range(len(maps)):
		pool = max_pooling(maps[i])
		pooled.append(pool)
	return pooled

def leaky_relu(x):
	if x > 0:
		return x
	else:
		return (0.2 * x)

def connector_layer(fMaps):
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
			activ = leaky_relu(tot)
			new.append(activ)
		else:
			activ = sigmoid(tot)
			new.append(activ)
	return new

def sigmoid(x):
	exp = ((math.e) ** (-1 * x))
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
				activ = leaky_relu(added)
				change = activ * alpha
			else:
				activ = sigmoid(added)
				change = activ * alpha
			layer[i][0][j] -= change
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
	if hG % 2 == 1:
		new_row = antipooled[-1]
		antipooled.append(new_row)
	return antipooled


CV1 = generate_filters(16, 7)
CV2 = generate_filters(16, 7)
CV3 = generate_filters(32, 5)
CV4 = generate_filters(32, 5)
CV5 = generate_filters(64, 3)
CV6 = generate_filters(64, 3)
FC7 = []
FC8 = []
FC9 = []
network_filters = [CV1, CV2, CV3, CV4, CV5, CV6]
network_nodes = [FC7, FC8, FC9]
result = [0, 0, 0, 0]

for i in range(5):
	FC7.append(generate_layer(49, 25))
	FC8.append(generate_layer(25, 10))
	FC9.append(generate_layer(10, 4))
	
# forward pass starts here
# images contains data for this set
# actuals contains coords for this set
# alpha is the hyperparameter learning rate

# load sample images here
images = []
for i in range(5):
	image = cv2.imread("NGCT1_IMG/ngct1_{}.png".format(i + 1), 1)
	images.append(image)

fMaps1 = []
for i in range(5):
	fMap1 = generate_feature_maps(CV1[i], images)
	pMap1 = pooled_maps(fMap1)
	fMaps1.append(pMap1)

sMaps1 = SNN.states(images, fMaps1)
fMaps2 = []
for i in range(5):
	fMap2 = generate_feature_maps(CV2[i], sMaps1[i])
	pMap2 = pooled_maps(fMap2)
	fMaps2.append(pMap2)

sMaps2 = SNN.states(sMaps1, fMaps2)
fMaps3 = []
for i in range(5):
	fMap3 = generate_feature_maps(CV3[i], sMaps2[i])
	pMap3 = pooled_maps(fMap3)
	fMaps3.append(pMap3)

sMaps3 = SNN.states(sMaps2, fMaps3)
fMaps4 = []
for i in range(5):
	fMap4 = generate_feature_maps(CV4[i], sMaps3[i])
	pMap4 = pooled_maps(fMap4)
	fMaps4.append(pMap4)

sMaps4 = SNN.states(sMaps3, fMaps4)
fMaps5 = []
for i in range(5):
	fMap5 = generate_feature_maps(CV5[i], sMaps4[i])
	pMap5 = pooled_maps(fMap5)
	fMaps5.append(pMap5)

sMaps5 = SNN.states(sMaps4, fMaps5)
fMaps6 = []
for i in range(5):
	fMap6 = generate_feature_maps(CV6[i], sMaps5[i])
	pMap6 = pooled_maps(fMap6)
	fMaps6.append(pMap6)
	
sMaps6 = SNN.states(sMaps5, fMaps6)

cnInputs = []
fcInputs8 = []
fcInputs9 = []
coords = []

for i in range(5):
	cnInput = connector_layer(sMaps6[i])
	cnInputs.append(cnInput)
	fcInput8 = forward_pass(FC7[i], cnInput, FC8[i], 0)
	fcInputs8.append(fcInput8)
	fcInput9 = forward_pass(FC8[i], fcInput8, FC9[i], 0)
	fcInputs9.append(fcInput9)
	fcOutput = forward_pass(FC9[i], fcInput9, result, 1)
	coords.append(fcOutput)

loss = mse(actuals, coords)
filter_loss = []
for i in range(5):
	(FC9[i], P9) = backprop(FC9[i], loss[i], 1, alpha, coords[i])
	(FC8[i], P8) = backprop(FC8[i], P9, 0, alpha, fcInputs9[i])
	(FC7[i], P7) = backprop(FC7[i], P8, 0, alpha, fcInputs8[i])
	filter_loss.append(P7)

filter_matrices = []
for k in range(5):
	filter_matrix = []
	for i in range(7):
		filter_row = []
		for j in range(7):
			place = (7 * i) + j
			filter_row.append(filter_loss[k][place])
		filter_matrix.append(filter_row)
	filter_matrices.append(filter_matrix)

deltaCV6 = []
deltaCV5 = []
deltaCV4 = []
deltaCV3 = []
deltaCV2 = []
deltaCV1 = []
for i in range(5):
	deltas6 = []
	deltas5 = []
	deltas4 = []
	deltas3 = []
	deltas2 = []
	
	deltaSIX = []
	deltaFIVE = []
	deltaFOUR = []
	deltaTHREE = []
	deltaTWO = []
	
	fms5 = []
	fms4 = []
	fms3 = []
	fms2 = []
	
	fmFIVE = []
	fmFOUR = []
	fmTHREE = []
	fmTWO = []
	
	for k in range(64):
		deltas6.append([])
	for k in range(32):
		deltas5.append([])
		deltas4.append([])
		fms5.append([])
		fms4.append([])
	for k in range(16):
		deltas3.append([])
		deltas2.append([])
		fms3.append([])
		fms2.append([])
	for j in range(len(sMaps6[i])):
		delta6 = algorithm.convolution(filter_matrices[i], sMaps6[i][j], 1) # this 1 is to make the resultant matrix kernel size 3
		# filter_matrices[i] and sMaps6[i][j] have dim 7x7
		rem = j % 64
		change6 = multiply(delta6, -1 * alpha) # dimensions are 3x3
		CV6[i][rem] = add([CV6[i][rem], change6])
		deltas6[rem].append(delta6)
	for k in range(64):
		tot = add(deltas6[k])
		count = len(deltas6[k])
		denom = 1.0 / count
		avg = multiply(tot, denom)
		deltaSIX.append(avg)
	for j in range(len(sMaps5[i])):
		rem = j % 64
		loss5 = algorithm.convolution(deltaSIX[rem], filter_matrices[i], 1) # this 1 is to maintain the size, since the filter is size 3
		fm5 = anti_pool(loss5, 15, 15) # size 15x15
		delta5 = algorithm.convolution(fm5, sMaps5[i][j], 1) # this 1 is between two 15x15, so it makes kernel size 3
		change5 = multiply(delta5, -1 * alpha) # dimensions are 3x3
		CV5[i][rem] = add([CV5[i][rem], change5])
		deltas5[rem // 2].append(delta5)
		fms5[rem // 2].append(fm5)
	for k in range(32):
		tot = add(deltas5[k])
		count = len(deltas5[k])
		denom = 1.0 / count
		avg = multiply(tot, denom)
		deltaFIVE.append(avg)
		
		tot = add(fms5[k])
		count = len(fms5[k])
		denom = 1.0 / count
		avg = multiply(tot, denom)
		fmFIVE.append(avg)
	for j in range(len(sMaps4[i])):
		rem = j % 32
		loss4 = algorithm.convolution(deltaFIVE[rem], fmFIVE[rem], 1) # this is to maintain the size and happens between 3x3 and 15x15, so it makes 15x15
		fm4 = anti_pool(loss4, 30, 30) # size 30x30
		delta4 = algorithm.convolution(fm4, sMaps4[i][j], 2) # this is between two 30x30 and is to make kernel size 5
		change4 = multiply(delta4, -1 * alpha) # 5x5
		CV4[i][rem] = add([CV4[i][rem], change4])
		deltas4[rem].append(delta4)
		fms4[rem].append(fm4)
	for k in range(32):
		tot = add(deltas4[k])
		count = len(deltas4[k])
		denom = 1.0 / count
		avg = multiply(tot, denom)
		deltaFOUR.append(avg)
		
		tot = add(fms4[k])
		count = len(fms4[k])
		denom = 1.0 / count
		avg = multiply(tot, denom)
		fmFOUR.append(avg)
	for j in range(len(sMaps3[i])):
		rem = j % 32
		loss3 = algorithm.convolution(deltaFOUR[rem], fmFOUR[rem], 2)
		fm3 = anti_pool(loss3, 60, 60)
		delta3 = algorithm.convolution(fm3, sMaps3[i][j], 2)
		change3 = multiply(delta3, -1 * alpha)
		CV3[i][rem] = add([CV3[i][rem], change3])
		deltas3[rem // 2].append(delta3)
		fms3[rem // 2].append(fm3)
	for k in range(16):
		tot = add(deltas3[k])
		count = len(deltas3[k])
		denom = 1.0 / count
		avg = multiply(tot, denom)
		deltaTHREE.append(avg)
		
		tot = add(fms3[k])
		count = len(fms3[k])
		denom = 1.0 / count
		avg = multiply(tot, denom)
		fmTHREE.append(avg)
	for j in range(len(sMaps2[i])):
		rem = j % 16
		loss2 = algorithm.convolution(deltaTHREE[rem], fmTHREE[rem], 2)
		fm2 = anti_pool(loss2, 121, 121)
		delta2 = algorithm.convolution(fm2, sMaps2[i][j], 3)
		change2 = multiply(delta2, -1 * alpha)
		CV2[i][rem] = add([CV2[i][rem], change2])
		deltas2[rem].append(delta2)
		fms2[rem].append(fm2)
	for k in range(16):
		tot = add(deltas2[k])
		count = len(deltas2[k])
		denom = 1.0 / count
		avg = multiply(tot, denom)
		deltaTWO.append(avg)
		
		tot = add(fms2[k])
		count = len(fms2[k])
		denom = 1.0 / count
		avg = multiply(tot, denom)
		fmTWO.append(avg)
	for j in range(len(sMaps1[i])):
		rem = j % 16
		loss1 = algorithm.convolution(deltaTWO[rem], fmTWO[rem], 3)
		fm1 = anti_pool(loss1, 242, 242)
		delta1 = algorithm.convolution(fm1, sMaps1[i][j], 3)
		change1 = multiply(delta2, -1 * alpha)
		CV1[i][rem] = add([CV1[i][rem], change1])

# concludes regular backpropagation
# starts plane recurrence

for l in range(len(network_filters)):
	for m in range(len(network_filters[l][0])):
		a = network_filters[l][0][m]
		b = network_filters[l][1][m]
		c = network_filters[l][2][m]
		d = network_filters[l][3][m]
		e = network_filters[l][4][m]
		
		matrices = techniques(a, b, c, d, e).plane_recurrence()
		
		for n in range(5):
			network_filters[l][n][m] = matrices[n]
