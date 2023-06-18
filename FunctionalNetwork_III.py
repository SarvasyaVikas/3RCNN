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

	def DRCNN(images, actuals, networkN, prev, neuralNetworks, rank):
		# adjust gradients to prevent against increasing loss
		comm = MPI.COMM_WORLD

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
		fMap1 = network.generate_feature_maps(CV1[rank], images[rank])
		pMap1 = network.pooled_maps(fMap1)
		sMaps1 = 0
		print("f2")
		
		if rank == 1:
			comm.send(pMap1, dest = 0)
		if rank == 2:
			comm.send(pMap1, dest = 0)
		if rank == 3:
			comm.send(pMap1, dest = 0)
		if rank == 4:
			comm.send(pMap1, dest = 0)
		
		if rank == 0:
			pMapA = comm.recv(source = 1)
			pMapB = comm.recv(source = 2)
			pMapC = comm.recv(source = 3)
			pMapD = comm.recv(source = 4)
			fMaps1 = [pMap1, pMapA, pMapB, pMapC, pMapD]
			sMaps1 = SNN.states(images, fMaps1)
			
			comm.send(sMaps1, dest = 1)
			comm.send(sMaps1, dest = 2)
			comm.send(sMaps1, dest = 3)
			comm.send(sMaps1, dest = 4)
		
		if rank == 1:
			sMaps1 = comm.recv(source = 0)
		if rank == 2:
			sMaps1 = comm.recv(source = 0)
		if rank == 3:
			sMaps1 = comm.recv(source = 0)
		if rank == 4:
			sMaps1 = comm.recv(source = 0)
		
		print("processors")
		sMaps2 = 0
		fMap2 = network.feature_maps_same(CV2[rank], sMaps1[rank])
		pMap2 = network.pooled_maps(fMap2, 1)
		
		if rank == 1:
			comm.send(pMap2, dest = 0)
		if rank == 2:
			comm.send(pMap2, dest = 0)
		if rank == 3:
			comm.send(pMap2, dest = 0)
		if rank == 4:
			comm.send(pMap2, dest = 0)
		
		if rank == 0:
			pMapA = comm.recv(source = 1)
			pMapB = comm.recv(source = 2)
			pMapC = comm.recv(source = 3)
			pMapD = comm.recv(source = 4)
			fMaps2 = [pMap2, pMapA, pMapB, pMapC, pMapD]
			sMaps2 = SNN.states(images, fMaps2)
			
			comm.send(sMaps2, dest = 1)
			comm.send(sMaps2, dest = 2)
			comm.send(sMaps2, dest = 3)
			comm.send(sMaps2, dest = 4)
		
		if rank == 1:
			sMaps2 = comm.recv(source = 0)
		if rank == 2:
			sMaps2 = comm.recv(source = 0)
		if rank == 3:
			sMaps2 = comm.recv(source = 0)
		if rank == 4:
			sMaps2 = comm.recv(source = 0)
		
		print("f3")
		sMaps3 = 0
		fMap3 = network.feature_maps_double(CV3[rank], sMaps2[rank])
		pMap3 = network.pooled_maps(fMap3, 1)
		
		if rank == 1:
			comm.send(pMap3, dest = 0)
		if rank == 2:
			comm.send(pMap3, dest = 0)
		if rank == 3:
			comm.send(pMap3, dest = 0)
		if rank == 4:
			comm.send(pMap3, dest = 0)
		
		if rank == 0:
			pMapA = comm.recv(source = 1)
			pMapB = comm.recv(source = 2)
			pMapC = comm.recv(source = 3)
			pMapD = comm.recv(source = 4)
			fMaps3 = [pMap3, pMapA, pMapB, pMapC, pMapD]
			sMaps3 = SNN.states(images, fMaps3)
			
			comm.send(sMaps3, dest = 1)
			comm.send(sMaps3, dest = 2)
			comm.send(sMaps3, dest = 3)
			comm.send(sMaps3, dest = 4)
		
		if rank == 1:
			sMaps3 = comm.recv(source = 0)
		if rank == 2:
			sMaps3 = comm.recv(source = 0)
		if rank == 3:
			sMaps3 = comm.recv(source = 0)
		if rank == 4:
			sMaps3 = comm.recv(source = 0)
		
		print("f4")
		sMaps4 = 0
		fMap4 = network.feature_maps_double(CV4[rank], sMaps3[rank])
		pMap4 = network.pooled_maps(fMap4, 1)
		
		if rank == 1:
			comm.send(pMap4, dest = 0)
		if rank == 2:
			comm.send(pMap4, dest = 0)
		if rank == 3:
			comm.send(pMap4, dest = 0)
		if rank == 4:
			comm.send(pMap4, dest = 0)
		
		if rank == 0:
			pMapA = comm.recv(source = 1)
			pMapB = comm.recv(source = 2)
			pMapC = comm.recv(source = 3)
			pMapD = comm.recv(source = 4)
			fMaps4 = [pMap4, pMapA, pMapB, pMapC, pMapD]
			sMaps4 = SNN.states(images, fMaps4)
			
			comm.send(sMaps4, dest = 1)
			comm.send(sMaps4, dest = 2)
			comm.send(sMaps4, dest = 3)
			comm.send(sMaps4, dest = 4)
		
		if rank == 1:
			sMaps4 = comm.recv(source = 0)
		if rank == 2:
			sMaps4 = comm.recv(source = 0)
		if rank == 3:
			sMaps4 = comm.recv(source = 0)
		if rank == 4:
			sMaps4 = comm.recv(source = 0)


		print("fp")
		
		# artificial forward propagation
		cnInputs = [0, 0, 0, 0, 0]
		fcInputs = [0, 0, 0, 0, 0]
		coords = [0, 0, 0, 0, 0]
		sm = [0, 0, 0, 0, 0]
		
		cnInput = network.connector_layer(sMaps4[rank])
		# cnInputs[rank] = cnInput
		fcInput = network.forward_pass(FC5[rank], cnInput, FC6[rank], 0)
		# fcInputs[rank] = fcInput
		fcOutput = network.forward_pass(FC6[rank], fcInput, result, 1)
		# coords[rank] = fcOutput
		softmax = network.forward_pass(SF[rank], fcOutput, [0, 0], 0)
		# sm[rank] = softmax
		
		print("fcb")
		loss = network.mseINDIV(actuals[rank], fcOutput)
		error = 0
		zeros = 0
		
		if softmax[0] > softmax[1]:
			zeros = [0, 0, 0, 0]
		else:
			zeros = fcOutput
		spec = network.mseINDIV(actuals[rank], zeros)
		
		for j in range(len(loss)):
			error += abs(loss[j])
			
		(SF[rank], _) = network.backprop(SF[rank], spec, 0, alpha, softmax)
		rho = network.direction(prev, error)
		
		(FC6[rank], P6) = network.backprop(FC6[rank], loss, 1, alpha * rho, fcOutput)
		(FC5[rank], P5) = network.backprop(FC5[rank], P6, 0, alpha * rho, fcInput)
		filter_loss = P5
		
		if rank in [1, 2, 3, 4]:
			comm.send([FC6[rank], FC5[rank]], dest = 0)
		
		if rank == 0:
			data1 = comm.recv(source = 1)
			data2 = comm.recv(source = 2)
			data3 = comm.recv(source = 3)
			data4 = comm.recv(source = 4)
			FC6[1] = data1[0]
			FC5[1] = data1[1]
			FC6[2] = data2[0]
			FC5[2] = data2[1]
			FC6[3] = data3[0]
			FC5[3] = data3[1]
			FC6[4] = data4[0]
			FC5[4] = data4[1]
		
		print("fb")
		filter_matrix = [0, 0, 0, 0, 0, 0, 0, 0]
		
		for k in range(8):
			filter_row = [0, 0, 0, 0, 0, 0, 0, 0]
			for j in range(8):
				place = (8 * k) + j
				filter_row[j] = filter_loss[place]
			filter_matrix[k] = filter_row
		
		cMaps = [0, 0, 0, 0, 0]
		
		print("4")
		for j in range(len(CV4[rank])):
			conv = algorithm.anticonvolution(filter_matrix, np.array(sMaps4[rank][j]), 1)
			delta = network.multiply(conv, alpha * rho)
			CV4[rank][j] = network.add([CV4[rank][j], delta])
		cRow = []
		for j in range(len(CV3[rank])):
			cRow.append(0)
		print("3")
		for j in range(len(CV3[rank])):
			fMap = network.anti_pool(np.array(filter_matrix), 16, 16)
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
		
		try:
			CV1.remove(0)
			CV2.remove(0)
			CV3.remove(0)
			CV4.remove(0)
			
			CV1.remove(0)
			CV2.remove(0)
			CV3.remove(0)
			CV4.remove(0)
			
			CV1.remove(0)
			CV2.remove(0)
			CV3.remove(0)
			CV4.remove(0)
			
			CV1.remove(0)
			CV2.remove(0)
			CV3.remove(0)
			CV4.remove(0)
			
			CV1 = CV1[0]
			CV2 = CV2[0]
			CV3 = CV3[0]
			CV4 = CV4[0]
		except:
			pass


		if rank in [1, 2, 3, 4]:
			comm.send([CV1, CV2, CV3, CV4], dest = 0)
		if rank == 0:
			news = [CV1, CV2, CV3, CV4]
			CVAs = comm.recv(source = 1)
			CVBs = comm.recv(source = 2)
			CVCs = comm.recv(source = 3)
			CVDs = comm.recv(source = 4)
			filters = []
			for i in range(4):
				filters.append([news[i], CVAs[i], CVBs[i], CVCs[i], CVDs[i]])
				
			comm.send(filters, dest = 1)
			comm.send(filters, dest = 2)
			comm.send(filters, dest = 3)
			comm.send(filters, dest = 4)

		if rank in [1, 2, 3, 4]:
			filters = comm.recv(source = 0)

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
			
			comm.send(filters[rank], dest = 4)
		
		filters1 = comm.recv(source = 0)
		filters2 = comm.recv(source = 1)
		filters3 = comm.recv(source = 2)
		filters4 = comm.recv(source = 3)

		filters = [filters1, filters2, filters3, filters4]
		
		final_nodes = [FC5, FC6]
		
		final_network = [filters, final_nodes, SF]
		
		return (final_network, error)
