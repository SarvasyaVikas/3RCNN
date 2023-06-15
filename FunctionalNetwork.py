import numpy as np
import cv2
import random
import math
from algorithm import algorithm
from techniques import techniques
from SNN import SNN
from network import network

class FunctionalNetwork:
	def __init__(self):
		pass

	def DRCNN(images, actuals, networkN):

		filters = networkN[0]
		nodes = networkN[1]
		
		CV1 = filters[0]
		CV2 = filters[1]
		CV3 = filters[2]
		CV4 = filters[3]
		
		FC5 = nodes[0]
		FC6 = nodes[1]

		# hyperparameters
		alpha = -0.01
		result = [0, 0, 0, 0]

		# convolutional forward propagation
		fMaps1 = []
		for i in range(5):
			fMap1 = network.generate_feature_maps(CV1[i], images[i])
			pMap1 = network.pooled_maps(fMap1)
			fMaps1.append(pMap1)

		sMaps1 = SNN.states(images, fMaps1)
		fMaps2 = []
		for i in range(5):
			fMap2 = network.feature_maps_same(CV2[i], sMaps1[i])
			pMap2 = network.pooled_maps(fMap2, 1)
			fMaps2.append(pMap2)

		sMaps2 = SNN.states(images, fMaps2)
		fMaps3 = []
		for i in range(5):
			fMap3 = network.feature_maps_double(CV3[i], sMaps2[i])
			pMap3 = network.pooled_maps(fMap3, 1)
			fMaps3.append(pMap3)

		sMaps3 = SNN.states(images, fMaps3)
		fMaps4 = []
		for i in range(5):
			fMap4 = network.feature_maps_double(CV4[i], sMaps3[i])
			pMap4 = network.pooled_maps(fMap4, 1)
			fMaps4.append(pMap4)

		sMaps4 = SNN.states(images, fMaps4)

		# artificial forward propagation
		cnInputs = []
		fcInputs = []
		coords = []
		for i in range(5):
			cnInput = network.connector_layer(sMaps4[i])
			cnInputs.append(cnInput)
			fcInput = network.forward_pass(FC5[i], cnInput, FC6[i], 0)
			fcInputs.append(fcInput)
			fcOutput = network.forward_pass(FC6[i], fcInput, result, 1)
			coords.append(fcOutput)

		loss = network.mse(actuals, coords)
		filter_loss = []
		for i in range(5):
			(FC6[i], P6) = network.backprop(FC6[i], loss[i], 1, alpha, coords[i])
			(FC5[i], P5) = network.backprop(FC5[i], P6, 0, alpha, fcInputs[i])
			filter_loss.append(P5)
				
		filter_matrices = []
		for k in range(5):
			filter_matrix = []
			for i in range(8):
				filter_row = []
				for j in range(8):
					place = (8 * i) + j
					filter_row.append(filter_loss[k][place])
				filter_matrix.append(filter_row)
			filter_matrices.append(filter_matrix)
		cMaps = []
		for i in range(5):
			for j in range(len(CV4[i])):
				conv = algorithm.anticonvolution(filter_matrices[i], np.array(sMaps4[i][j]), 1)
				delta = network.multiply(conv, alpha)
				CV4[i][j] = network.add([CV4[i][j], delta])
			cRow = []
			for j in range(len(CV3[i])):
				fMap = network.anti_pool(np.array(filter_matrices[i]), 16, 16)
				filter_place1 = 2 * j
				filter_place2 = filter_place1 + 1
				avg = network.multiply(network.add([CV4[i][filter_place1], CV4[i][filter_place2]]), 0.5)
				cMap = algorithm.convolution(avg, fMap, 1)
				conv = algorithm.anticonvolution(cMap, np.array(sMaps3[i][j]), 1)
				delta = network.multiply(conv, alpha)
				CV3[i][j] = network.add([CV3[i][j], delta])
				cRow.append(cMap)
			cMaps.append(cRow)
			for j in range(len(CV2[i])):
				filter_place1 = 2 * j
				filter_place2 = filter_place1 + 1
				fMap1 = network.anti_pool(np.array(cMaps[i][filter_place1]), 32, 32)
				fMap2 = network.anti_pool(np.array(cMaps[i][filter_place2]), 32, 32)
				avg = network.multiply(network.add([CV3[i][filter_place1], CV3[i][filter_place2]]), 0.5)
				avgF = network.multiply(network.add([fMap1, fMap2]), 0.5)
				dMap = algorithm.convolution(avg, avgF, 1)
				conv = algorithm.anticonvolution(dMap, np.array(sMaps2[i][j]), 2)
				delta = network.multiply(conv, alpha)
				CV2[i][j] = network.add([CV2[i][j], delta])
				
				fMap = network.anti_pool(np.array(dMap), 64, 64)
				eMap = algorithm.convolution(CV2[i][j], fMap, 2)
				conv = algorithm.anticonvolution(eMap, np.array(sMaps1[i][j]), 2)
				delta = network.multiply(conv, alpha)
				CV1[i][j] = network.add([CV1[i][j], delta])

		for l in range(len(filters)):
			for m in range(len(filters[l][0])):
				a = filters[l][0][m]
				b = filters[l][1][m]
				c = filters[l][2][m]
				d = filters[l][3][m]
				e = filters[l][4][m]
				
				matrices = techniques(a, b, c, d, e).plane_recurrence()
				
				for n in range(5):
					filters[l][n][m] = matrices[n]
		
		final_filters = [CV1, CV2, CV3, CV4]
		final_nodes = [FC5, FC6]
		
		final_network = [final_filters, final_nodes]
		
		return (final_network, loss)
