import numpy as np
import cv2
import random
import math
from algorithm import algorithm
from techniques import techniques
from SNN import SNN
from network import network
from optimizerMV import optimizerMV
from Modifications import Modifications

class FunctionalNetwork:
	def __init__(self):
		pass

	def DRCNN(images, actuals, networkN, prev, neuralNetworks, rank):
		# adjust gradients to prevent against increasing loss

		filters = networkN[0]
		nodes = networkN[1]
		SF = networkN[2]
		
		CV1 = filters[0]
		CV2 = filters[1]
		CV3 = filters[2]
		CV4 = filters[3]
		
		FC5 = nodes[0]
		FC6 = nodes[1]

		# hyperparameters
		alpha = -0.01
		result = [0, 0, 0, 0]
		print("f1")
		
		# convolutional forward propagation
		fMaps1 = [0, 0, 0, 0, 0]
		fMap1 = network.generate_feature_maps(CV1[rank], images[rank])
		pMap1 = network.pooled_maps(fMap1)
		fMaps1[rank] = pMap1
		
		print("f2")
		sMaps1 = SNN.states(images, fMaps1)
		fMaps2 = [0, 0, 0, 0, 0]
		fMap2 = network.feature_maps_same(CV2[rank], sMaps1[rank])
		pMap2 = network.pooled_maps(fMap2, 1)
		fMaps2[rank] = pMap2
		
		print("f3")
		sMaps2 = SNN.states(images, fMaps2)
		fMaps3 = [0, 0, 0, 0, 0]
		fMap3 = network.feature_maps_double(CV3[rank], sMaps2[rank])
		pMap3 = network.pooled_maps(fMap3, 1)
		fMaps3[rank] = pMap3
		
		print("f4")
		sMaps3 = SNN.states(images, fMaps3)
		fMaps4 = [0, 0, 0, 0, 0]
		fMap4 = network.feature_maps_double(CV4[rank], sMaps3[rank])
		pMap4 = network.pooled_maps(fMap4, 1)
		fMaps4[rank] = pMap4

		sMaps4 = SNN.states(images, fMaps4)
		print("fp")
		
		# artificial forward propagation
		cnInputs = [0, 0, 0, 0, 0]
		fcInputs = [0, 0, 0, 0, 0]
		coords = [0, 0, 0, 0, 0]
		sm = [0, 0, 0, 0, 0]
		
		cnInput = network.connector_layer(sMaps4[rank])
		cnInputs[rank] = cnInput
		fcInput = network.forward_pass(FC5[rank], cnInput, FC6[rank], 0)
		fcInputs[rank] = fcInput
		fcOutput = network.forward_pass(FC6[rank], fcInput, result, 1)
		coords[rank] = fcOutput
		softmax = network.forward_pass(SF[rank], fcOutput, [0, 0], 0)
		sm[rank] = softmax
		
		print("fcb")
		loss = network.mse(actuals, coords)
		error = 0
		zeros = [0, 0, 0, 0, 0]
		
		if sm[0] > sm[1]:
			zeros[rank] = [0, 0, 0, 0]
		else:
			zeros[rank] = coords[rank]
		spec = network.mse(actuals, zeros)
		
		for j in range(len(loss[0])):
			error += abs(loss[rank][j])
			
		(SF[rank], _) = network.backprop(SF[rank], spec[rank], 0, alpha, sm[rank])
		rho = network.direction(prev, error)
		filter_loss = [0, 0, 0, 0, 0]
		
		(FC6[rank], P6) = network.backprop(FC6[rank], loss[rank], 1, alpha * rho, coords[rank])
		(FC5[rank], P5) = network.backprop(FC5[rank], P6, 0, alpha * rho, fcInputs[rank])
		filter_loss[rank] = P5
		
		print("fb")		
		filter_matrices = [0, 0, 0, 0, 0]
		filter_matrix = [0, 0, 0, 0, 0, 0, 0, 0]
		
		for k in range(8):
			filter_row = [0, 0, 0, 0, 0, 0, 0, 0]
			for j in range(8):
				place = (8 * k) + j
				filter_row[j] = filter_loss[rank][place]
			filter_matrix[k] = filter_row
		filter_matrices[rank] = filter_matrix
		
		cMaps = [0, 0, 0, 0, 0]
		
		print("4")
		for j in range(len(CV4[rank])):
			conv = algorithm.anticonvolution(filter_matrices[rank], np.array(sMaps4[rank][j]), 1)
			delta = network.multiply(conv, alpha * rho)
			CV4[rank][j] = network.add([CV4[rank][j], delta])
		cRow = []
		for j in range(len(CV3[rank])):
			cRow.append(0)
		print("3")
		for j in range(len(CV3[rank])):
			fMap = network.anti_pool(np.array(filter_matrices[rank]), 16, 16)
			filter_place1 = 2 * j
			filter_place2 = filter_place1 + 1
			avg = network.multiply(network.add([CV4[rank][filter_place1], CV4[rank][filter_place2]]), 0.5)
			avgS = network.multiply(network.add([sMaps4[rank][filter_place1], sMaps4[rank][filter_place2]]), 0.5)
			apS = network.anti_pool(np.array(avgS), 16, 16)
			
			cMap = algorithm.convolution(avg, fMap, 1) # this is l
			
			CV3[rank][j] = optimizerMV.optimize(cMap, sMaps3[rank][j], apS, CV3[rank][j], alpha * rho) # OPTIMIZER
			
			# conv = algorithm.anticonvolution(cMap, np.array(sMaps3[rank][j]), 1)
			# delta = network.multiply(conv, alpha)
			# CV3[rank][j] = network.add([CV3[rank][j], delta])
			cRow[j] = cMap
		print("12")
		cMaps[rank] = cRow
		for j in range(len(CV2[rank])):
			filter_place1 = 2 * j
			filter_place2 = filter_place1 + 1
			fMap1 = network.anti_pool(np.array(cMaps[rank][filter_place1]), 32, 32)
			fMap2 = network.anti_pool(np.array(cMaps[rank][filter_place2]), 32, 32)
			avg = network.multiply(network.add([CV3[rank][filter_place1], CV3[rank][filter_place2]]), 0.5)
			avgF = network.multiply(network.add([fMap1, fMap2]), 0.5)
			avgS = network.multiply(network.add([sMaps3[rank][filter_place1], sMaps3[rank][filter_place2]]), 0.5)
			apS = network.anti_pool(np.array(avgS), 32, 32)
			dMap = algorithm.convolution(avg, avgF, 1)
			
			CV2[rank][j] = optimizerMV.optimize(dMap, sMaps2[rank][j], apS, CV2[rank][j], alpha * rho)
			
			# conv = algorithm.anticonvolution(dMap, np.array(sMaps2[rank][j]), 2)
			# delta = network.multiply(conv, alpha)
			# CV2[rank][j] = network.add([CV2[rank][j], delta])
			fMap = network.anti_pool(np.array(dMap), 64, 64)
			eMap = algorithm.convolution(CV2[rank][j], fMap, 2)
			
			CV1[rank][j] = optimizerMV.optimize(eMap, sMaps1[rank][j], network.anti_pool(sMaps2[rank][j], 64, 64), CV1[rank][j], alpha * rho)
			
			# conv = algorithm.anticonvolution(eMap, np.array(sMaps1[rank][j]), 2)
			# delta = network.multiply(conv, alpha)
			# CV1[rank][j] = network.add([CV1[rank][j], delta])
		print("p")

		filters = [CV1, CV2, CV3, CV4]

		# assign these jobs to four different processors
		if rank in [0, 1, 2, 3]:
			for m in range(len(filters[rank][0])):
				a = np.array(network.signed_ln(filters[rank][0][m]))
				b = np.array(network.signed_ln(filters[rank][1][m]))
				c = np.array(network.signed_ln(filters[rank][2][m]))
				d = np.array(network.signed_ln(filters[rank][3][m]))
				e = np.array(network.signed_ln(filters[rank][4][m]))
					
				matrices = techniques(a, b, c, d, e).plane_recurrence()
					
				for n in range(5):
					filters[rank][n][m] = matrices[n]

		final_nodes = [FC5, FC6]
		
		final_network = [filters, final_nodes, SF]
		
		return (final_network, error)
