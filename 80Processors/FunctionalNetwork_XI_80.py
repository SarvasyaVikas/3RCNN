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

	def FC(networkNS, actual, alpha, prev, sMaps4, networkPREV, psi):
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
			
		(networkNS[2], _) = network.backpropNESTEROV(networkNS[2], spec, 0, alpha, softmax, networkPREV, psi)
		rho = network.direction(prev, error)
			
		(networkNS[1][1], P6) = network.backpropNESTEROV(networkNS[1][1], loss, 1, alpha * rho, fcOutput, networkPREV, psi)
		(networkNS[1][0], P5) = network.backpropNESTEROV(networkNS[1][0], P6, 0, alpha * rho, fcInput, networkPREV, psi)
		filter_loss = P5
		
		print("fb")
		filter_matrix = []
		
		for k in range(8):
			filter_row = []
			for j in range(8):
				place = (8 * k) + j
				filter_row.append(filter_loss[place])
			filter_matrix.append(filter_row)
			
		return (networkNS, error, filter_matrix, rho)
		
	def BP(networkNS, error, alpha, filter_matrix, sMaps4, sMaps3, sMaps2, sMaps1, rank, networkPREV, psi, rho):
		print("4")
		for j in range(4):
			place = (4 * rank) + j
			conv = algorithm.anticonvolution(filter_matrix, np.array(sMaps4[place]), 1)
			deltaPREV = np.subtract(networkPREV[0][3][j], networkNS[0][3][j])
			delta = network.multiply(conv, alpha * rho)
			deltaNEW = MPImodifiers.nesterovMomentum(deltaPREV, delta, psi)
			networkNS[0][3][place] = network.add([networkNS[0][3][place], delta])
		cRow = []
		print("3")
		for j in range(2):
			fMap = network.anti_pool(np.array(filter_matrix), 16, 16)
			place = (2 * rank) + j
			filter_place1 = 2 * place
			filter_place2 = filter_place1 + 1
			avg = network.multiply(network.add([networkNS[0][3][filter_place1], networkNS[0][3][filter_place2]]), 0.5)
			avgS = network.multiply(network.add([sMaps4[filter_place1], sMaps4[filter_place2]]), 0.5)
			apS = network.anti_pool(np.array(avgS), 16, 16)
			
			cMap = algorithm.convolution(avg, fMap, 1) # this is l
			
			networkNS[0][2][place] = optimizerMV.optimizeNESTEROV(cMap, sMaps3[place], apS, networkNS[0][2][place], alpha * rho, networkPREV[0][2][place], psi) # OPTIMIZER
			cRow.append(cMap)
		print("12")
		
		filter_place1 = 2 * rank
		filter_place2 = filter_place1 + 1
		fMap1 = network.anti_pool(np.array(cRow[filter_place1]), 32, 32)
		fMap2 = network.anti_pool(np.array(cRow[filter_place2]), 32, 32)
		avg = network.multiply(network.add([networkNS[0][2][filter_place1], networkNS[0][2][filter_place2]]), 0.5)
		avgF = network.multiply(network.add([fMap1, fMap2]), 0.5)
		avgS = network.multiply(network.add([sMaps3[filter_place1], sMaps3[filter_place2]]), 0.5)
		apS = network.anti_pool(np.array(avgS), 32, 32)
		dMap = algorithm.convolution(avg, avgF, 1)
		
		networkNS[0][1][rank] = optimizerMV.optimizeNESTEROV(dMap, sMaps2[rank], apS, networkNS[0][1][rank], alpha * rho, networkPREV[0][1][rank], psi)
		
		fMap = network.anti_pool(np.array(dMap), 64, 64)
		eMap = algorithm.convolution(networkNS[0][1][rank], fMap, 2)
			
		networkNS[0][0][rank] = optimizerMV.optimizeNESTEROV(eMap, sMaps1[rank], network.anti_pool(sMaps2[rank], 64, 64), networkNS[0][0][rank], alpha * rho, networkPREV[0][1][rank], psi)
		
		return (networkNS, error)
	
	def PR(filters): # corresponds to all CV1s, or CVxs
		print("p")
		# assign these jobs to four different processors
		a = np.array(network.signed_ln(filters[0]))
		b = np.array(network.signed_ln(filters[1]))
		c = np.array(network.signed_ln(filters[2]))
		d = np.array(network.signed_ln(filters[3]))
		e = np.array(network.signed_ln(filters[4]))
					
		matrices = techniques(a, b, c, d, e).plane_recurrence()
					
		return matrices
