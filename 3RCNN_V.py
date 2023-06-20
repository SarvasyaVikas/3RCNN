import cv2
import numpy as np
import csv
import fnmatch
import os
import csv
from FunctionalNetwork_IV import FunctionalNetwork
from network import network
from optimizerMV import optimizerMV
import time
from Modifications import Modifications
from mpi4py import MPI
from SNN import SNN
from parallel import parallel

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if (rank % 16) == 0:
	CV1 = network.parallel_filters(16, 5)
	CV2 = network.parallel_filters(16, 5)
	CV3 = network.parallel_filters(32, 3)
	CV4 = network.parallel_filters(64, 3)
	FC5 = network.generate_layer(64, 16)
	FC6 = network.generate_layer(16, 4)
	SF = network.generate_layer(4, 2)
	filters = [CV1, CV2, CV3, CV4]
	nodes = [FC5, FC6]
	network = [filters, nodes, SF]

	for i in range(1, 16):
		comm.send(network, dest = rank + i)

else:
	place = rank - (rank % 16)
	network = comm.recv(source = place)


# ABOVE THIS: DO NOT TOUCH
#
#

# This is for RCA

# 1 scans
val = 40
alpha = 0.01

def save(networkN, code, losses):
	filters = networkN[0]
	nodes = networkN[1]
	SF = networkN[2]
	
	CV1 = filters[0]
	CV2 = filters[1]
	CV3 = filters[2]
	CV4 = filters[3]
	
	FC5 = nodes[0]
	FC6 = nodes[1]
	
	lst = [code]
	lst.append(min(losses))
	for i in range(len(losses)):
		lst.append(losses[i])
	lst.append("S")
	for a in range(len(filters)):
		for j in range(len(filters[a])):
			for k in range(len(filters[a][j])):
				for l in range(len(filters[a][j][k])):
					lst.append(filters[a][j][k][l])
	for b in range(len(nodes)):
		for j in range(len(nodes[b])):
			for k in range(len(nodes[b][j][0])):
				lst.append(nodes[b][j][0][k])
			lst.append(nodes[b][j][1])
			
	for c in range(len(SF)):
		for j in range(len(SF[c][0])):
			lst.append(SF[c][j][0][k])
		lst.append(SF[c][j][1])
	
	print(lst)
	csvfile = open("saved.csv", "a+")
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(lst)
	csvfile.close()
	

def div(lst, val):
	fin = []
	for i in range(len(lst)):
		new = lst[i] / val
		fin.append(new)
	return fin

def data(ptn, val = 0):
	rows = []
	with open("3RCNN_Data_Annotations.csv") as csvfile:
		csvreader = csv.reader(csvfile, delimiter = ",")
		for row in csvreader:
			if int(row[1]) == ptn:
				rows.append(row)
	if val == 0:
		val = len(rows) - 4
	images = []
	actuals = []
	for i in range(val):
		vals1 = [int(rows[i][3]), int(rows[i][4]), int(rows[i][5]), int(rows[i][6])]
		vals2 = [int(rows[i + 1][3]), int(rows[i + 1][4]), int(rows[i + 1][5]), int(rows[i + 1][6])]
		vals3 = [int(rows[i + 2][3]), int(rows[i + 2][4]), int(rows[i + 2][5]), int(rows[i + 2][6])]
		vals4 = [int(rows[i + 3][3]), int(rows[i + 3][4]), int(rows[i + 3][5]), int(rows[i + 3][6])]
		vals5 = [int(rows[i + 4][3]), int(rows[i + 4][4]), int(rows[i + 4][5]), int(rows[i + 4][6])]
		actuals.append([div(vals1, 1024), div(vals2, 1024), div(vals3, 1024), div(vals4, 1024), div(vals5, 1024)])

		image1 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 1), 0)
		image2 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 2), 0)
		image3 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 3), 0)
		image4 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 4), 0)
		image5 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 5), 0)
		image1 = cv2.resize(image1, (128, 128))
		image2 = cv2.resize(image2, (128, 128))
		image3 = cv2.resize(image3, (128, 128))
		image4 = cv2.resize(image4, (128, 128))
		image5 = cv2.resize(image5, (128, 128))
		images.append([image1, image2, image3, image4, image5])
		
	return (images, actuals)

losses = [64, 64]
nn = network
# scan 1
scan_start = time.time()
(Is1, As1) = data(1, val)
for i in range(val):
	start = time.time()
	
	section = rank // 16
	rem = rank % 16
	fMap1 = parallel.generate_feature_map(network[0][rem], Is1[i][section])
	pMap1 = network.max_pooling(fMap1)

	if (rank % 5) != 0:
		place = (rank // 5) * 5
		comm.send(pMap1, dest = place)
	
	if (rank % 5) == 0:
		pMapB = comm.recv(source = rank + 1)
		pMapC = comm.recv(source = rank + 2)
		pMapD = comm.recv(source = rank + 3)
		pMapE = comm.recv(source = rank + 4)
		
		pMaps = SNN.states(Is1[i], [pMap1, pMapB, pMapC, pMapD, pMapE])
		
		sMap1 = pMaps[0]
		sMapB = pMaps[1]
		sMapC = pMaps[2]
		sMapD = pMaps[3]
		sMapE = pMaps[4]
		
		comm.send(sMapB, dest = rank + 1)
		comm.send(sMapC, dest = rank + 2)
		comm.send(sMapD, dest = rank + 3)
		comm.send(sMapE, dest = rank + 4)
		
	if (rank % 5) != 0:
		place = (rank // 5) * 5
		sMap1 = comm.recv(source = place)
		
	fMap2 = parallel.generate_feature_map(network[1][rem], sMap1)
	pMap2 = network.max_pooling(fMap2)
	
	if (rank % 5) != 0:
		place = (rank // 5) * 5
		comm.send(pMap2, dest = place)
	
	if (rank % 5) == 0:
		pMapB = comm.recv(source = rank + 1)
		pMapC = comm.recv(source = rank + 2)
		pMapD = comm.recv(source = rank + 3)
		pMapE = comm.recv(source = rank + 4)
		
		pMaps = SNN.states(Is1[i], [pMap2, pMapB, pMapC, pMapD, pMapE])
		
		sMap2 = pMaps[0]
		sMapB = pMaps[1]
		sMapC = pMaps[2]
		sMapD = pMaps[3]
		sMapE = pMaps[4]
		
		comm.send(sMapB, dest = rank + 1)
		comm.send(sMapC, dest = rank + 2)
		comm.send(sMapD, dest = rank + 3)
		comm.send(sMapE, dest = rank + 4)
		
	if (rank % 5) != 0:
		place = (rank // 5) * 5
		sMap2 = comm.recv(source = place)
	
	place1 = rem * 2
	place2 = (rem * 2) + 1
	fMap3A = parallel.generate_feature_map(network[2][place1], sMap2)
	fMap3B = parallel.generate_feature_map(network[2][place2], sMap2)
	
	pMap3A = network.max_pooling(fMap3A)
	pMap3B = network.max_pooling(fMap3B)
	
	if (rank % 5) != 0:
		place = (rank // 5) * 5
		comm.send(pMap3A, dest = place)
	
	if (rank % 5) == 0:
		pMapB = comm.recv(source = rank + 1)
		pMapC = comm.recv(source = rank + 2)
		pMapD = comm.recv(source = rank + 3)
		pMapE = comm.recv(source = rank + 4)
		
		pMaps = SNN.states(Is1[i], [pMap3A, pMapB, pMapC, pMapD, pMapE])
		
		sMap3A = pMaps[0]
		sMapB = pMaps[1]
		sMapC = pMaps[2]
		sMapD = pMaps[3]
		sMapE = pMaps[4]
		
		comm.send(sMapB, dest = rank + 1)
		comm.send(sMapC, dest = rank + 2)
		comm.send(sMapD, dest = rank + 3)
		comm.send(sMapE, dest = rank + 4)
		
	if (rank % 5) != 0:
		place = (rank // 5) * 5
		sMap3A = comm.recv(source = place)
	#
	if (rank % 5) != 0:
		place = (rank // 5) * 5
		comm.send(pMap3B, dest = place)
	
	if (rank % 5) == 0:
		pMapB = comm.recv(source = rank + 1)
		pMapC = comm.recv(source = rank + 2)
		pMapD = comm.recv(source = rank + 3)
		pMapE = comm.recv(source = rank + 4)
		
		pMaps = SNN.states(Is1[i], [pMap3B, pMapB, pMapC, pMapD, pMapE])
		
		sMap3B = pMaps[0]
		sMapB = pMaps[1]
		sMapC = pMaps[2]
		sMapD = pMaps[3]
		sMapE = pMaps[4]
		
		comm.send(sMapB, dest = rank + 1)
		comm.send(sMapC, dest = rank + 2)
		comm.send(sMapD, dest = rank + 3)
		comm.send(sMapE, dest = rank + 4)
		
	if (rank % 5) != 0:
		place = (rank // 5) * 5
		sMap3B = comm.recv(source = place)
	#
	
	placeA = rem * 4
	placeB = (rem * 4) + 1
	placeC = (rem * 4) + 2
	placeD = (rem * 4) + 3
	
	fMap4A = parallel.generate_feature_map(network[3][placeA], sMap3A)
	fMap4B = parallel.generate_feature_map(network[3][placeB], sMap3A)
	fMap4C = parallel.generate_feature_map(network[3][placeC], sMap3B)
	fMap4D = parallel.generate_feature_map(network[3][placeD], sMap3B)
	
	pMap4A = network.max_pooling(fMap4A)
	pMap4B = network.max_pooling(fMap4B)
	pMap4C = network.max_pooling(fMap4C)
	pMap4D = network.max_pooling(fMap4D)
	
	if (rank % 5) != 0:
		place = (rank // 5) * 5
		comm.send(pMap4A, dest = place)
	
	if (rank % 5) == 0:
		pMapB = comm.recv(source = rank + 1)
		pMapC = comm.recv(source = rank + 2)
		pMapD = comm.recv(source = rank + 3)
		pMapE = comm.recv(source = rank + 4)
		
		pMaps = SNN.states(Is1[i], [pMap4A, pMapB, pMapC, pMapD, pMapE])
		
		sMap4A = pMaps[0]
		sMapB = pMaps[1]
		sMapC = pMaps[2]
		sMapD = pMaps[3]
		sMapE = pMaps[4]
		
		comm.send(sMapB, dest = rank + 1)
		comm.send(sMapC, dest = rank + 2)
		comm.send(sMapD, dest = rank + 3)
		comm.send(sMapE, dest = rank + 4)
		
	if (rank % 5) != 0:
		place = (rank // 5) * 5
		sMap4A = comm.recv(source = place)
	#
	if (rank % 5) != 0:
		place = (rank // 5) * 5
		comm.send(pMap4B, dest = place)
	
	if (rank % 5) == 0:
		pMapB = comm.recv(source = rank + 1)
		pMapC = comm.recv(source = rank + 2)
		pMapD = comm.recv(source = rank + 3)
		pMapE = comm.recv(source = rank + 4)
		
		pMaps = SNN.states(Is1[i], [pMap4B, pMapB, pMapC, pMapD, pMapE])
		
		sMap4B = pMaps[0]
		sMapB = pMaps[1]
		sMapC = pMaps[2]
		sMapD = pMaps[3]
		sMapE = pMaps[4]
		
		comm.send(sMapB, dest = rank + 1)
		comm.send(sMapC, dest = rank + 2)
		comm.send(sMapD, dest = rank + 3)
		comm.send(sMapE, dest = rank + 4)
		
	if (rank % 5) != 0:
		place = (rank // 5) * 5
		sMap4B = comm.recv(source = place)
		
	if (rank % 5) != 0:
		place = (rank // 5) * 5
		comm.send(pMap4C, dest = place)
	
	if (rank % 5) == 0:
		pMapB = comm.recv(source = rank + 1)
		pMapC = comm.recv(source = rank + 2)
		pMapD = comm.recv(source = rank + 3)
		pMapE = comm.recv(source = rank + 4)
		
		pMaps = SNN.states(Is1[i], [pMap4C, pMapB, pMapC, pMapD, pMapE])
		
		sMap4C = pMaps[0]
		sMapB = pMaps[1]
		sMapC = pMaps[2]
		sMapD = pMaps[3]
		sMapE = pMaps[4]
		
		comm.send(sMapB, dest = rank + 1)
		comm.send(sMapC, dest = rank + 2)
		comm.send(sMapD, dest = rank + 3)
		comm.send(sMapE, dest = rank + 4)
		
	if (rank % 5) != 0:
		place = (rank // 5) * 5
		sMap4C = comm.recv(source = place)
	#
	if (rank % 5) != 0:
		place = (rank // 5) * 5
		comm.send(pMap4D, dest = place)
	
	if (rank % 5) == 0:
		pMapB = comm.recv(source = rank + 1)
		pMapC = comm.recv(source = rank + 2)
		pMapD = comm.recv(source = rank + 3)
		pMapE = comm.recv(source = rank + 4)
		
		pMaps = SNN.states(Is1[i], [pMap4D, pMapB, pMapC, pMapD, pMapE])
		
		sMap4D = pMaps[0]
		sMapB = pMaps[1]
		sMapC = pMaps[2]
		sMapD = pMaps[3]
		sMapE = pMaps[4]
		
		comm.send(sMapB, dest = rank + 1)
		comm.send(sMapC, dest = rank + 2)
		comm.send(sMapD, dest = rank + 3)
		comm.send(sMapE, dest = rank + 4)
		
	if (rank % 5) != 0:
		place = (rank // 5) * 5
		sMap4D = comm.recv(source = place)
	
	#
	
	if (rank % 16) != 0:
		comm.send([sMap4A, sMap4B, sMap4C, sMap4D, sMap3A, sMap3B, sMap2, sMap1], dest = 0)
	
	if (rank % 16) == 0:
		sMaps4 = []
		sMaps3 = []
		sMaps2 = []
		sMaps1 = []
		for i in range(1, size):
			sMap = comm.recv(source = i)
			sMaps4.append(sMap[0])
			sMaps4.append(sMap[1])
			sMaps4.append(sMap[2])
			sMaps4.append(sMap[3])
			sMaps3.append(sMap[4])
			sMaps3.append(sMap[5])
			sMaps2.append(sMap[6])
			sMaps1.append(sMap[7])
			
		(network, error, filter_matrix) = FunctionalNetwork.FC(network, As1[i][rank // 16], alpha, losses[-1], sMaps4)
		
		for i in range(1, 16):
			comm.send((network, error, filter_matrix, sMaps4, sMaps3, sMaps2, sMaps1), dest = rank + i)
		
	if (rank % 16) != 0:
		place = (rank // 16) * 16
		(network, error, filter_matrix, sMaps4, sMaps3, sMaps2, sMaps1) = comm.recv(source = place)
	
	(network, error) = FunctionalNetwork.BP(network, As1[i][section], alpha, losses[-1], sMaps4, sMaps3, sMaps2, sMaps1, rank)
	#
	if rank != 0:
		comm.send(network, dest = 0)
	
	if rank == 0:
		networkFULL = [network, network, network, network, network]
		for i in range(1, 80):
			section = i // 16
			part = i % 16
			place1 = (2 * i)
			place2 = place1 + 1
			placeA = (4 * i)
			placeB = placeA + 1
			placeC = placeB + 1
			placeD = placeC + 1
			network = comm.recv(source = i)
			networkFULL[section][0][0][part] = network[0][0][part]
			networkFULL[section][0][1][part] = network[0][1][part]
			
			networkFULL[section][0][2][place1] = network[0][2][place1]
			networkFULL[section][0][2][place2] = network[0][2][place2]
			
			networkFULL[section][0][3][placeA] = network[0][3][placeA]
			networkFULL[section][0][3][placeB] = network[0][3][placeB]
			networkFULL[section][0][3][placeC] = network[0][3][placeC]
			networkFULL[section][0][3][placeD] = network[0][3][placeD]
			
			if part == 0:
				networkFULL[section][1] = network[1]
				networkFULL[section][2] = network[2]
	
		planeRecurrence = FunctionalNetwork.PR(networkFULL[0][0][0][0], networkFULL[1][0][0][0], networkFULL[2][0][0][0], networkFULL[3][0][0][0], networkFULL[4][0][0][0])
		for i in range(5):
			networkFULL[i][0][0][0] = planeRecurrence[i]
			
		for i in range(1, 64):	
			comm.send(networkFULL, dest = i)
	
	if rank < 32:
		mod = rank % 16
		sect = rank // 16
		networkFULL = comm.recv(source = 0)
		planeRecurrence = FunctionalNetwork.PR(networkFULL[0][0][sect][mod], networkFULL[1][0][sect][mod], networkFULL[2][0][sect][mod], networkFULL[3][0][sect][mod], networkFULL[4][0][sect][mod])
		comm.send(planeRecurrence, dest = 0)
	elif rank < 48:
		networkFULL = comm.recv(source = 0)
		place1 = (rank - 32) * 2
		place2 = place1 + 1
		planeRecurrence1 = FunctionalNetwork.PR(networkFULL[0][0][2][place1], networkFULL[1][0][2][place1], networkFULL[2][0][2][place1], networkFULL[3][0][2][place1], networkFULL[4][0][2][place1])
		planeRecurrence2 = FunctionalNetwork.PR(networkFULL[0][0][2][place2], networkFULL[1][0][2][place2], networkFULL[2][0][2][place2], networkFULL[3][0][2][place2], networkFULL[4][0][2][place2])
		comm.send((planeRecurrence1, planeRecurrence2), dest = 0)
	elif rank < 64:
		networkFULL = comm.recv(source = 0)
		placeA = (rank - 48) * 4
		placeB = placeA + 1
		placeC = placeB + 1
		placeD = placeC + 1
		
		planeRecurrenceA = FunctionalNetwork.PR(networkFULL[0][0][3][placeA], networkFULL[1][0][3][placeA], networkFULL[2][0][3][placeA], networkFULL[3][0][3][placeA], networkFULL[4][0][3][placeA])
		planeRecurrenceB = FunctionalNetwork.PR(networkFULL[0][0][3][placeB], networkFULL[1][0][3][placeB], networkFULL[2][0][3][placeB], networkFULL[3][0][3][placeB], networkFULL[4][0][3][placeB])
		planeRecurrenceC = FunctionalNetwork.PR(networkFULL[0][0][3][placeC], networkFULL[1][0][3][placeC], networkFULL[2][0][3][placeC], networkFULL[3][0][3][placeC], networkFULL[4][0][3][placeC])
		planeRecurrenceD = FunctionalNetwork.PR(networkFULL[0][0][3][placeD], networkFULL[1][0][3][placeD], networkFULL[2][0][3][placeD], networkFULL[3][0][3][placeD], networkFULL[4][0][3][placeD])
		comm.send((planeRecurrenceA, planeRecurrenceB, planeRecurrenceC, planeRecurrenceD), dest = 0)
	
	if rank == 0:
		for i in range(1, 32):
			mod = rank % 16
			sect = rank // 16
			pR = comm.recv(source = i)
			for j in range(5):
				networkFULL[j][0][sect][mod] = pR[j]
		for i in range(32, 48):
			(pR1, pR2) = comm.recv(source = i)
			place1 = 2 * (i - 32)
			place2 = place1 + 1
			for j in range(5):
				networkFULL[j][0][2][place1] = pR1[j]
				networkFULL[j][0][2][place2] = pR2[j]
		for i in range(48, 64):
			(pRA, pRB, pRC, pRD) = comm.recv(source = i)
			placeA = 4 * (i - 48)
			placeB = placeA + 1
			placeC = placeB + 1
			placeD = placeC + 1
			for j in range(5)
				networkFULL[j][0][3][placeA] = pRA[j]
				networkFULL[j][0][3][placeB] = pRB[j]
				networkFULL[j][0][3][placeC] = pRC[j]
				networkFULL[j][0][3][placeD] = pRD[j]
				
		for i in range(1, 80):
			comm.send(networkFULL, dest = i)
	
	if rank != 0:
		networkFULL = comm.recv(source = 0)
		
	#
	if error < losses[-1]:
		nn = networkFULL
	losses.append(error)
	end = time.time()
	print(end - start)
	
end_scan = time.time()
print(end_scan - start_scan)
save(nn, "SCAN1_ATT1_RANK{}".format(rank), losses)
