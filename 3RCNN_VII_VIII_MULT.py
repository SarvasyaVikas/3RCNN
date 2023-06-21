mult = [[1, 0, 40], [3, 40, 60], [4, 20, 40], [5, 40, 60], [6, 40, 60], [10, 0, 40], [12, 50, 70], [13, 50, 70], [14, 50, 70], [15, 30, 50]]

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
from MPImodifiers import MPImodifiers
from mpi4py import MPI
from SNN import SNN

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank < 5:
	CV1 = network.parallel_filters(16, 5)
	CV2 = network.parallel_filters(16, 5)
	CV3 = network.parallel_filters(32, 3)
	CV4 = network.parallel_filters(64, 3)
	FC5 = network.generate_layer(64, 16)
	FC6 = network.generate_layer(16, 4)
	SF = network.generate_layer(4, 2)
	filters = [CV1, CV2, CV3, CV4]
	nodes = [FC5, FC6]
	networkS = [filters, nodes, SF]
else:
	CV1 = network.parallel_filters(16, 5)
	CV2 = network.parallel_filters(16, 5)
	CV3 = network.parallel_filters(32, 3)
	CV4 = network.parallel_filters(64, 3)
	FC5 = network.generate_layer(64, 16)
	FC6 = network.generate_layer(16, 4)
	SF = network.generate_layer(4, 2)
	filters = [CV1, CV2, CV3, CV4]
	nodes = [FC5, FC6]
	networkS = [filters, nodes, SF]
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
			lst.append(SF[c][0][j])
		lst.append(SF[c][1])
	
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

def data(ptn, start = 0, val = 0):
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
	for i in range(start, val):
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

for j in range(len(mult)):
	if rank < 5:
		losses = [64, 64]
		neurals = [networkS]
		nn = networkS
		# scan 1
		scan_start = time.time()
		(Is1, As1) = data(mult[j][0], mult[j][1], mult[j][2])
		for i in range(val):
			start = time.time()
			print(i)
			pMap1 = FunctionalNetwork.F1(Is1[i][rank], networkS)
			if rank in [1, 2, 3, 4]:
				comm.send(pMap1, dest = 0)
			
			if rank == 0:
				pMapB = comm.recv(source = 1)
				pMapC = comm.recv(source = 2)
				pMapD = comm.recv(source = 3)
				pMapE = comm.recv(source = 4)
				
				pMaps = SNN.states(Is1[i], [pMap1, pMapB, pMapC, pMapD, pMapE])
				
				sMap1 = pMaps[0]
				sMapB = pMaps[1]
				sMapC = pMaps[2]
				sMapD = pMaps[3]
				sMapE = pMaps[4]
				
				comm.send(sMapB, dest = 1)
				comm.send(sMapC, dest = 2)
				comm.send(sMapD, dest = 3)
				comm.send(sMapE, dest = 4)
				
			if rank in [1, 2, 3, 4]:
				sMap1 = comm.recv(source = 0)
				
			pMap2 = FunctionalNetwork.F2(sMap1, networkS)
			
			if rank in [1, 2, 3, 4]:
				comm.send(pMap2, dest = 0)
			
			if rank == 0:
				pMapB = comm.recv(source = 1)
				pMapC = comm.recv(source = 2)
				pMapD = comm.recv(source = 3)
				pMapE = comm.recv(source = 4)
				
				pMaps = SNN.states(Is1[i], [pMap2, pMapB, pMapC, pMapD, pMapE])
				
				sMap2 = pMaps[0]
				sMapB = pMaps[1]
				sMapC = pMaps[2]
				sMapD = pMaps[3]
				sMapE = pMaps[4]
				
				comm.send(sMapB, dest = 1)
				comm.send(sMapC, dest = 2)
				comm.send(sMapD, dest = 3)
				comm.send(sMapE, dest = 4)
				
			if rank in [1, 2, 3, 4]:
				sMap2 = comm.recv(source = 0)
			
			pMap3 = FunctionalNetwork.F3(sMap2, networkS)
			
			if rank in [1, 2, 3, 4]:
				comm.send(pMap3, dest = 0)
			
			if rank == 0:
				pMapB = comm.recv(source = 1)
				pMapC = comm.recv(source = 2)
				pMapD = comm.recv(source = 3)
				pMapE = comm.recv(source = 4)
				
				pMaps = SNN.states(Is1[i], [pMap3, pMapB, pMapC, pMapD, pMapE])
				
				sMap3 = pMaps[0]
				sMapB = pMaps[1]
				sMapC = pMaps[2]
				sMapD = pMaps[3]
				sMapE = pMaps[4]
				
				comm.send(sMapB, dest = 1)
				comm.send(sMapC, dest = 2)
				comm.send(sMapD, dest = 3)
				comm.send(sMapE, dest = 4)
				
			if rank in [1, 2, 3, 4]:
				sMap3 = comm.recv(source = 0)
			
			pMap4 = FunctionalNetwork.F4(sMap3, networkS)
			
			if rank in [1, 2, 3, 4]:
				comm.send(pMap4, dest = 0)
			
			if rank == 0:
				pMapB = comm.recv(source = 1)
				pMapC = comm.recv(source = 2)
				pMapD = comm.recv(source = 3)
				pMapE = comm.recv(source = 4)
				
				pMaps = SNN.states(Is1[i], [pMap4, pMapB, pMapC, pMapD, pMapE])
				
				sMap4 = pMaps[0]
				sMapB = pMaps[1]
				sMapC = pMaps[2]
				sMapD = pMaps[3]
				sMapE = pMaps[4]
				
				comm.send(sMapB, dest = 1)
				comm.send(sMapC, dest = 2)
				comm.send(sMapD, dest = 3)
				comm.send(sMapE, dest = 4)
				
			if rank in [1, 2, 3, 4]:
				sMap4 = comm.recv(source = 0)
			
			maps = MPImodifiers.afm(sMap1, sMap2, sMap3, sMap4)
			
			(networkS, error) = FunctionalNetwork.BP(networkS, As1[i][rank], alpha, losses[-1], maps, sMap3, sMap2, sMap1)
			neurals.append(networkS)
			if error < losses[-1]:
				nn = networkS
			losses.append(error)
			end = time.time()
			print(end - start)
			
		end_scan = time.time()
		print(end_scan - start_scan)
		for i in range(len(neurals)):
			save(neurals[i], "SCAN{}_ATT1_RANK{}_PLACE{}".format(mult[j][0], rank, i), losses)
		save(nn, "SCAN{}_ATT1_RANK{}".format(mult[j][0], rank), losses)
	
	else:
		spot = rank - 5
		losses = [64, 64]
		neurals = [networkS]
		nn = networkS
		# scan 1
		scan_start = time.time()
		(Is1, As1) = data(1, val)
		for i in range(val):
			start = time.time()
			print(i)
			pMap1 = FunctionalNetwork.F1(Is1[i][spot], networkS)
			if spot in [1, 2, 3, 4]:
				comm.send(pMap1, dest = 0)
			
			if spot == 0:
				pMapB = comm.recv(source = 1)
				pMapC = comm.recv(source = 2)
				pMapD = comm.recv(source = 3)
				pMapE = comm.recv(source = 4)
				
				pMaps = SNN.states(Is1[i], [pMap1, pMapB, pMapC, pMapD, pMapE])
				
				sMap1 = pMaps[0]
				sMapB = pMaps[1]
				sMapC = pMaps[2]
				sMapD = pMaps[3]
				sMapE = pMaps[4]
				
				comm.send(sMapB, dest = 1)
				comm.send(sMapC, dest = 2)
				comm.send(sMapD, dest = 3)
				comm.send(sMapE, dest = 4)
				
			if spot in [1, 2, 3, 4]:
				sMap1 = comm.recv(source = 0)
				
			pMap2 = FunctionalNetwork.F2(sMap1, networkS)
			
			if spot in [1, 2, 3, 4]:
				comm.send(pMap2, dest = 0)
			
			if spot == 0:
				pMapB = comm.recv(source = 1)
				pMapC = comm.recv(source = 2)
				pMapD = comm.recv(source = 3)
				pMapE = comm.recv(source = 4)
				
				pMaps = SNN.states(Is1[i], [pMap2, pMapB, pMapC, pMapD, pMapE])
				
				sMap2 = pMaps[0]
				sMapB = pMaps[1]
				sMapC = pMaps[2]
				sMapD = pMaps[3]
				sMapE = pMaps[4]
				
				comm.send(sMapB, dest = 1)
				comm.send(sMapC, dest = 2)
				comm.send(sMapD, dest = 3)
				comm.send(sMapE, dest = 4)
				
			if spot in [1, 2, 3, 4]:
				sMap2 = comm.recv(source = 0)
			
			pMap3 = FunctionalNetwork.F3(sMap2, networkS)
			
			if spot in [1, 2, 3, 4]:
				comm.send(pMap3, dest = 0)
			
			if spot == 0:
				pMapB = comm.recv(source = 1)
				pMapC = comm.recv(source = 2)
				pMapD = comm.recv(source = 3)
				pMapE = comm.recv(source = 4)
				
				pMaps = SNN.states(Is1[i], [pMap3, pMapB, pMapC, pMapD, pMapE])
				
				sMap3 = pMaps[0]
				sMapB = pMaps[1]
				sMapC = pMaps[2]
				sMapD = pMaps[3]
				sMapE = pMaps[4]
				
				comm.send(sMapB, dest = 1)
				comm.send(sMapC, dest = 2)
				comm.send(sMapD, dest = 3)
				comm.send(sMapE, dest = 4)
				
			if spot in [1, 2, 3, 4]:
				sMap3 = comm.recv(source = 0)
			
			pMap4 = FunctionalNetwork.F4(sMap3, networkS)
			
			if spot in [1, 2, 3, 4]:
				comm.send(pMap4, dest = 0)
			
			if spot == 0:
				pMapB = comm.recv(source = 1)
				pMapC = comm.recv(source = 2)
				pMapD = comm.recv(source = 3)
				pMapE = comm.recv(source = 4)
				
				pMaps = SNN.states(Is1[i], [pMap4, pMapB, pMapC, pMapD, pMapE])
				
				sMap4 = pMaps[0]
				sMapB = pMaps[1]
				sMapC = pMaps[2]
				sMapD = pMaps[3]
				sMapE = pMaps[4]
				
				comm.send(sMapB, dest = 1)
				comm.send(sMapC, dest = 2)
				comm.send(sMapD, dest = 3)
				comm.send(sMapE, dest = 4)
				
			if spot in [1, 2, 3, 4]:
				sMap4 = comm.recv(source = 0)
			
			maps = MPImodifiers.mfm(Is1[i][spot], sMap4)
			
			(networkS, error) = FunctionalNetwork.BP(networkS, As1[i][spot], alpha, losses[-1], maps, sMap3, sMap2, sMap1)
			neurals.append(networkS)
			if error < losses[-1]:
				nn = networkS
			losses.append(error)
			end = time.time()
			print(end - start)
			
		end_scan = time.time()
		print(end_scan - start_scan)
		for i in range(len(neurals)):
			save(neurals[i], "SCAN{]_ATT1_RANK{}_PLACE{}".format(mult[j][0], spot, i), losses)
		save(nn, "SCAN{}_ATT1_RANK{}".format(mult[j][0], spot), losses)
