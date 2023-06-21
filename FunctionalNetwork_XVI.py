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
import time
from mpi4py import MPI

class FunctionalNetwork:
	def __init__(self):
		pass

	def F1(image, networkNS):
		# adjust gradients to prevent against increasing loss
		print("f1")
		# convolutional forward propagation
		fMap1 = network.generate_feature_maps(networkNS[0][0], image)
		pMap1 = network.pooled_maps(fMap1)
		
		return pMap1

	def F2(sMaps1, networkNS):
		print("f2")
		fMap2 = network.feature_maps_same(networkNS[0][1], sMaps1)
		pMap2 = network.pooled_maps(fMap2, 1)
		
		return pMap2
		
	def F3(sMaps2, networkNS):
		print("f3")
		fMap3 = network.feature_maps_double(networkNS[0][2], sMaps2)
		pMap3 = network.pooled_maps(fMap3, 1)
		
		return pMap3
	
	def F4(sMaps3, networkNS):
		print("f4")
		fMap4 = network.feature_maps_double(networkNS[0][3], sMaps3)
		pMap4 = network.pooled_maps(fMap4, 1)
		
		return pMap4

	def BP(networkNS, actual, alpha, prev, sMaps4, sMaps3, sMaps2, sMaps1, networkPREVS, psi, stretch):
		print("fp")
		
		# artificial forward propagation
		
		cnInput = network.connector_layer(sMaps4)
		fcInput = network.forward_pass(networkNS[1][0], cnInput, networkNS[1][1], 0)
		fcOutput = network.forward_pass(networkNS[1][1], fcInput, [0, 0, 0, 0], 1)
		softmax = network.forward_pass(networkNS[2], fcOutput, [0, 0], 0)
		
		print("fcb")
		loss = network.mseINDIV(actual, fcOutput)
		error = 0
		zeros = 0
		
		if softmax[0] > softmax[1]:
			zeros = [0, 0, 0, 0]
		else:
			zeros = fcOutput
		spec = network.mseINDIV(actual, zeros)
		
		for j in range(len(loss)):
			error += abs(loss[j])
		prevs2 = []
		for a in range(len(networkPREVS)):
			prevs2.append(networkPREVS[a][2])
		(networkNS[2], _) = network.backpropMACLAURIN(networkNS[2], spec, 0, alpha, softmax, prevs2, psi, stretch)
		rho = network.direction(prev, error)
		
		prevs11 = []
		for a in range(len(networkPREVS)):
			prevs2.append(networkPREVS[a][1][1])
		prevs10 = []
		for a in range(len(networkPREVS)):
			prevs2.append(networkPREVS[a][1][0])
		
		(networkNS[1][1], P6) = network.backpropMACLAURIN(networkNS[1][1], loss, 1, alpha * rho, fcOutput, prevs11, psi, stretch)
		(networkNS[1][0], P5) = network.backpropMACLAURIN(networkNS[1][0], P6, 0, alpha * rho, fcInput, prevs10, psi, stretch)
		filter_loss = P5
		
		print("fb")
		filter_matrix = []
		
		for k in range(8):
			filter_row = []
			for j in range(8):
				place = (8 * k) + j
				filter_row.append(filter_loss[place])
			filter_matrix.append(filter_row)
		
		print("4")
		for j in range(len(networkNS[0][3])):
			conv = algorithm.anticonvolution(filter_matrix, np.array(sMaps4[j]), 1)
			deltaPREV = networkPREV[0][3][j] - networkNS[0][3][j]
			delta = network.multiply(conv, alpha * rho)
			deltaNEW = MPImodifiers.nesterovMomentum(deltaPREV, delta, psi)
			networkNS[0][3][j] = np.subtract(networkNS[0][3][j], deltaNEW)
		cRow = []
		print("3")
		for j in range(len(networkNS[0][2])):
			fMap = network.anti_pool(np.array(filter_matrix), 16, 16)
			filter_place1 = 2 * j
			filter_place2 = filter_place1 + 1
			avg = network.multiply(network.add([networkNS[0][3][filter_place1], networkNS[0][3][filter_place2]]), 0.5)
			avgS = network.multiply(network.add([sMaps4[filter_place1], sMaps4[filter_place2]]), 0.5)
			apS = network.anti_pool(np.array(avgS), 16, 16)
			
			cMap = algorithm.convolution(avg, fMap, 1) # this is l
			
			networkNS[0][2][j] = optimizerMV.optimizeNESTEROV(cMap, sMaps3[j], apS, networkNS[0][2][j], alpha * rho, networkPREV[0][2][j], psi) # OPTIMIZER
			cRow.append(cMap)
		print("12")
		for j in range(len(networkNS[0][1])):
			filter_place1 = 2 * j
			filter_place2 = filter_place1 + 1
			fMap1 = network.anti_pool(np.array(cRow[filter_place1]), 32, 32)
			fMap2 = network.anti_pool(np.array(cRow[filter_place2]), 32, 32)
			avg = network.multiply(network.add([networkNS[0][2][filter_place1], networkNS[0][2][filter_place2]]), 0.5)
			avgF = network.multiply(network.add([fMap1, fMap2]), 0.5)
			avgS = network.multiply(network.add([sMaps3[filter_place1], sMaps3[filter_place2]]), 0.5)
			apS = network.anti_pool(np.array(avgS), 32, 32)
			dMap = algorithm.convolution(avg, avgF, 1)
			
			networkNS[0][1][j] = optimizerMV.optimizeNESTEROV(dMap, sMaps2[j], apS, networkNS[0][1][j], alpha * rho, networkPREV[0][2][j], psi)
			
			fMap = network.anti_pool(np.array(dMap), 64, 64)
			eMap = algorithm.convolution(networkNS[0][1][j], fMap, 2)
			
			networkNS[0][0][j] = optimizerMV.optimizeNESTEROV(eMap, sMaps1[j], network.anti_pool(sMaps2[j], 64, 64), networkNS[0][0][j], alpha * rho, networkPREV[0][2][j], psi)
		
		return (networkNS, error)
	
	def PR(filters): # corresponds to all CV1s, or CVxs
		print("p")
		# assign these jobs to four different processors
		for m in range(len(filters[0])):
			a = np.array(network.signed_ln(filters[0][m]))
			b = np.array(network.signed_ln(filters[1][m]))
			c = np.array(network.signed_ln(filters[2][m]))
			d = np.array(network.signed_ln(filters[3][m]))
			e = np.array(network.signed_ln(filters[4][m]))
					
			matrices = techniques(a, b, c, d, e).plane_recurrence()
					
			for n in range(5):
				filters[n][m] = matrices[n]

		return networkNS
