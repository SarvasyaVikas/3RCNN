import numpy as np
import cv2
import random
import math
from algorithm import algorithm
from techniques import techniques
from SNN import SNN
from network import network
from optimizerMV import optimizerMV

class FunctionalNetwork:
	def __init__(self):
		pass

	def DRCNN(images, actuals, networkN, prev):

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
		fMaps1 = []
		for i in range(5):
			fMap1 = network.generate_feature_maps(CV1[i], images[i])
			pMap1 = network.pooled_maps(fMap1)
			fMaps1.append(pMap1)
		print("f2")
		sMaps1 = SNN.states(images, fMaps1)
		fMaps2 = []
		for i in range(5):
			fMap2 = network.feature_maps_same(CV2[i], sMaps1[i])
			pMap2 = network.pooled_maps(fMap2, 1)
			fMaps2.append(pMap2)
		print("f3")
		sMaps2 = SNN.states(images, fMaps2)
		fMaps3 = []
		for i in range(5):
			fMap3 = network.feature_maps_double(CV3[i], sMaps2[i])
			pMap3 = network.pooled_maps(fMap3, 1)
			fMaps3.append(pMap3)
		print("f4")
		sMaps3 = SNN.states(images, fMaps3)
		fMaps4 = []
		for i in range(5):
			fMap4 = network.feature_maps_double(CV4[i], sMaps3[i])
			pMap4 = network.pooled_maps(fMap4, 1)
			fMaps4.append(pMap4)

		sMaps4 = SNN.states(images, fMaps4)
		print("fp")
		# artificial forward propagation
		cnInputs = []
		fcInputs = []
		coords = []
		sm = []
		for i in range(5):
			cnInput = network.connector_layer(sMaps4[i])
			cnInputs.append(cnInput)
			fcInput = network.forward_pass(FC5[i], cnInput, FC6[i], 0)
			fcInputs.append(fcInput)
			fcOutput = network.forward_pass(FC6[i], fcInput, result, 1)
			coords.append(fcOutput)
			softmax = network.forward_pass(SF[i], fcOutput, [0, 0], 0)
			sm.append(softmax)
		print("fcb")
		loss = network.mse(actuals, coords)
		error = 0
		zeros = []
		for i in range(5):
			if sm[0] > sm[1]:
				zeros.append([0, 0, 0, 0])
			else:
				zeros.append(coords[i])
		spec = network.mse(actuals, zeros)
		
		for i in range(len(loss)):
			for j in range(len(loss[0])):
				error += abs(loss[i][j])
		
		for i in range(5):
			(SF[i], _) = network.backprop(SF[i], spec[i], 0, alpha, sm[i])
		rho = network.direction(prev, error)
		filter_loss = []
		for i in range(5):
			(FC6[i], P6) = network.backprop(FC6[i], loss[i], 1, alpha, coords[i])
			(FC5[i], P5) = network.backprop(FC5[i], P6, 0, alpha, fcInputs[i])
			filter_loss.append(P5)
		print("fb")		
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
			print("4")
			for j in range(len(CV4[i])):
				conv = algorithm.anticonvolution(filter_matrices[i], np.array(sMaps4[i][j]), 1)
				delta = network.multiply(conv, alpha * rho)
				CV4[i][j] = network.add([CV4[i][j], delta])
			cRow = []
			print("3")
			for j in range(len(CV3[i])):
				fMap = network.anti_pool(np.array(filter_matrices[i]), 16, 16)
				filter_place1 = 2 * j
				filter_place2 = filter_place1 + 1
				avg = network.multiply(network.add([CV4[i][filter_place1], CV4[i][filter_place2]]), 0.5)
				avgS = network.multiply(network.add([sMaps4[i][filter_place1], sMaps4[i][filter_place2]]), 0.5)
				apS = network.anti_pool(np.array(avgS), 16, 16)
				
				cMap = algorithm.convolution(avg, fMap, 1) # this is l
				
				CV3[i][j] = optimizerMV.optimize(cMap, sMaps3[i][j], apS, CV3[i][j], alpha * rho) # OPTIMIZER
				
				# conv = algorithm.anticonvolution(cMap, np.array(sMaps3[i][j]), 1)
				# delta = network.multiply(conv, alpha)
				# CV3[i][j] = network.add([CV3[i][j], delta])
				cRow.append(cMap)
			print("12")
			cMaps.append(cRow)
			for j in range(len(CV2[i])):
				filter_place1 = 2 * j
				filter_place2 = filter_place1 + 1
				fMap1 = network.anti_pool(np.array(cMaps[i][filter_place1]), 32, 32)
				fMap2 = network.anti_pool(np.array(cMaps[i][filter_place2]), 32, 32)
				avg = network.multiply(network.add([CV3[i][filter_place1], CV3[i][filter_place2]]), 0.5)
				avgF = network.multiply(network.add([fMap1, fMap2]), 0.5)
				avgS = network.multiply(network.add([sMaps3[i][filter_place1], sMaps3[i][filter_place2]]), 0.5)
				apS = network.anti_pool(np.array(avgS), 32, 32)
				dMap = algorithm.convolution(avg, avgF, 1)
				
				CV2[i][j] = optimizerMV.optimize(dMap, sMaps2[i][j], apS, CV2[i][j], alpha * rho)
				
				# conv = algorithm.anticonvolution(dMap, np.array(sMaps2[i][j]), 2)
				# delta = network.multiply(conv, alpha)
				# CV2[i][j] = network.add([CV2[i][j], delta])
				fMap = network.anti_pool(np.array(dMap), 64, 64)
				eMap = algorithm.convolution(CV2[i][j], fMap, 2)
				
				CV1[i][j] = optimizerMV.optimize(eMap, sMaps1[i][j], network.anti_pool(sMaps2[i][j], 64, 64), CV1[i][j], alpha * rho)
				
				# conv = algorithm.anticonvolution(eMap, np.array(sMaps1[i][j]), 2)
				# delta = network.multiply(conv, alpha)
				# CV1[i][j] = network.add([CV1[i][j], delta])
		print("p")

		for m in range(len(CV1[0])):
			a = np.array(network.signed_ln(CV1[0][m]))
			b = np.array(network.signed_ln(CV1[1][m]))
			c = np.array(network.signed_ln(CV1[2][m]))
			d = np.array(network.signed_ln(CV1[3][m]))
			e = np.array(network.signed_ln(CV1[4][m]))
				
			matrices = techniques(a, b, c, d, e).plane_recurrence()
				
			for n in range(5):
				CV1[n][m] = matrices[n]
		
		for m in range(len(CV2[0])):
			a = np.array(network.signed_ln(CV2[0][m]))
			b = np.array(network.signed_ln(CV2[1][m]))
			c = np.array(network.signed_ln(CV2[2][m]))
			d = np.array(network.signed_ln(CV2[3][m]))
			e = np.array(network.signed_ln(CV2[4][m]))
				
			matrices = techniques(a, b, c, d, e).plane_recurrence()
				
			for n in range(5):
				CV2[n][m] = matrices[n]
				
		for m in range(len(CV3[0])):
			a = np.array(network.signed_ln(CV3[0][m]))
			b = np.array(network.signed_ln(CV3[1][m]))
			c = np.array(network.signed_ln(CV3[2][m]))
			d = np.array(network.signed_ln(CV3[3][m]))
			e = np.array(network.signed_ln(CV3[4][m]))
				
			matrices = techniques(a, b, c, d, e).plane_recurrence()
				
			for n in range(5):
				CV3[n][m] = matrices[n]
		
		for m in range(len(CV4[0])):
			a = np.array(network.signed_ln(CV4[0][m]))
			b = np.array(network.signed_ln(CV4[1][m]))
			c = np.array(network.signed_ln(CV4[2][m]))
			d = np.array(network.signed_ln(CV4[3][m]))
			e = np.array(network.signed_ln(CV4[4][m]))
				
			matrices = techniques(a, b, c, d, e).plane_recurrence()
				
			for n in range(5):
				CV4[n][m] = matrices[n]
		
		final_filters = [CV1, CV2, CV3, CV4]
		final_nodes = [FC5, FC6]
		
		final_network = [final_filters, final_nodes, SF]
		
		return (final_network, error)
