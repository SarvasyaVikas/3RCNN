import cv2
import numpy as np
import csv
import fnmatch
import os
import csv
from FunctionalNetwork_VI_O import FunctionalNetwork
from network import network
import time
from Modifications import Modifications
from mpi4py import MPI
from SNN import SNN
from parallel import parallel
from sets import SETS
from algorithm import algorithm
# Uses DP and MVI

mult = SETS.data(1.0)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("generation")
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
    networkS = [filters, nodes, SF]

    for i in range(1, 16):
        comm.send(networkS, dest = rank + i, tag = 0)

else:
    place = rank - (rank % 16)
    networkS = comm.recv(source = place, tag = 0)
print("networkS")

# ABOVE THIS: DO NOT TOUCH
#
#

# This is for RCA

# 1 scans
val = 40
alpha = 0.01
epochs = 2

def unpack(row, networkI):
	count = 0
	cop = row.copy()
	for i in range(len(row)):
		if str(row[i]) == "S":
			count = i + 1
	
		if i == count:
	 		for i in range(len(row)):
	 			if i >= count:
	 				cop[i] = 0
	 			else:
	 				cop[i] = 1
	 	
	 		for a in range(len(networkI[0])):
	 			for j in range(len(networkI[0][a])):
	 				for k in range(len(networkI[0][a][j])):
	 					for l in range(len(networkI[0][a][j][k])):
	 						adj = cop.index(0, count)
	 						networkI[0][a][j][k][l] = row[adj]
	 						cop[adj] = 1
	 		
	 		for b in range(networkI[1]):
	 			for j in range(len(networkI[1][b])):
	 				for k in range(len(networkI[1][b][j])):
	 					adj = cop.index(0, count + 1664)
	 					networkI[1][b][j][0][k] = row[adj]
	 					cop[adj] = 1
	 				adj = cop.index(0, count + 1664)
	 				networkI[1][b][j][1] = row[adj]
	 				cop[adj] = 1
	 		
	 		for c in range(len(networkI[2])):
	 			for j in range(len(networkI[2][c][0])):
	 				adj = cop.index(0, count + 1664)
	 				networkI[2][c][0][j] = row[adj]
	 				cop[adj] = 1
	 			adj = cop.index(0, count + 1664)
	 			networkI[2][c][1] = row[adj]
	 			cop[adj] = 1
	 		
	 		return networkI

def divide(lst, val):
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
        vals1 = [float(rows[i][3]), float(rows[i][4]), float(rows[i][5]), float(rows[i][6])]
        vals2 = [float(rows[i + 1][3]), float(rows[i + 1][4]), float(rows[i + 1][5]), float(rows[i + 1][6])]
        vals3 = [float(rows[i + 2][3]), float(rows[i + 2][4]), float(rows[i + 2][5]), float(rows[i + 2][6])]
        vals4 = [float(rows[i + 3][3]), float(rows[i + 3][4]), float(rows[i + 3][5]), float(rows[i + 3][6])]
        vals5 = [float(rows[i + 4][3]), float(rows[i + 4][4]), float(rows[i + 4][5]), float(rows[i + 4][6])]
        actuals.append([divide(vals1, 1024), divide(vals2, 1024), divide(vals3, 1024), divide(vals4, 1024), divide(vals5, 1024)])

        try:
            imageO = cv2.imread("NGCT{}_IMG/ngct{}_{}.png".format(ptn, ptn, i + 1), 0)
            (h, w) = imageO.shape[:2]
            left = 5 * w // 16
            right = 13 * w // 16
            top = 3 * h // 16
            bottom = 11 * h // 16
        except:
            pass

        try:
            image1 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 1), 0)
            image1 = cv2.resize(image1, (128, 128))
        except:
            imageO = cv2.imread("NGCT{}_IMG/ngct{}_{}.png".format(ptn, ptn, i + 1), 0)
            imageR = imageO[top:bottom, left:right]
            image1 = algorithm.max_pooling(imageR)
            image1 = cv2.resize(image1, (128, 128))
        try:
            image2 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 2), 0)
            image2 = cv2.resize(image2, (128, 128))
        except:
            imageO = cv2.imread("NGCT{}_IMG/ngct{}_{}.png".format(ptn, ptn, i + 2), 0)
            imageR = imageO[top:bottom, left:right]
            image2 = algorithm.max_pooling(imageR)
            image2 = cv2.resize(image2, (128, 128))
        try:
            image3 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 3), 0)
            image3 = cv2.resize(image3, (128, 128))
        except:
            imageO = cv2.imread("NGCT{}_IMG/ngct{}_{}.png".format(ptn, ptn, i + 3), 0)
            imageR = imageO[top:bottom, left:right]
            image3 = algorithm.max_pooling(imageR)
            image3 = cv2.resize(image3, (128, 128))
        try:
            image4 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 4), 0)
            image4 = cv2.resize(image4, (128, 128))
        except:
            imageO = cv2.imread("NGCT{}_IMG/ngct{}_{}.png".format(ptn, ptn, i + 4), 0)
            imageR = imageO[top:bottom, left:right]
            image4 = algorithm.max_pooling(imageR)
            image4 = cv2.resize(image4, (128, 128))
        try:
            image5 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 5), 0)
            image5 = cv2.resize(image5, (128, 128))
        except:
            imageO = cv2.imread("NGCT{}_IMG/ngct{}_{}.png".format(ptn, ptn, i + 5), 0)
            imageR = imageO[top:bottom, left:right]
            image5 = algorithm.max_pooling(imageR)
            image5 = cv2.resize(image5, (128, 128))
        # image1 = cv2.resize(image1, (128, 128))
        # image2 = cv2.resize(image2, (128, 128))
        # image3 = cv2.resize(image3, (128, 128))
        # image4 = cv2.resize(image4, (128, 128))
        # image5 = cv2.resize(image5, (128, 128))
        images.append([image1, image2, image3, image4, image5])
        
    return (images, actuals)

ind = 1936
networkfile = open("saved_I.csv", "r")
c = 0
for row in networkfile:
	if c == ind:
		networkS = unpack(row, networkS)
	c += 1
errors = []

nn = [networkS, networkS, networkS, networkS, networkS]
neurals = [nn.copy()]

for epoch in range(epochs):
    for j in range(len(mult)):
        print(mult[j][0])
        print("losses")
        losses = [64, 64]
        try:
            a = len(networkS)
            b = len(networkS[0])
            c = len(networkS[0][0])
            if (a == 3) and (b == 4) and (c == 16):
                pass
            else:
                networkS = nn[rank // 16]
        except:
            networkS = nn[rank // 16]
        # scan 1
        scan_start = time.time()
        print("scan start")
        (Is1, As1) = data(mult[j][0], mult[j][1], mult[j][2])
        print("data gen")
        for i in range(mult[j][1], mult[j][2]):
            start = time.time()
            print(i)
            u = i - mult[j][1]
            section = rank // 16
            rem = rank % 16
            fMap1 = parallel.generate_feature_map(networkS[0][0][rem], Is1[u][section])
            pMap1 = network.max_pooling(fMap1)
            print("f1")
            if (rank % 5) != 0:
                place = (rank // 5) * 5
                comm.send(pMap1, dest = place, tag = (rank % 5))
        
            if (rank % 5) == 0:
                pMapB = comm.recv(source = rank + 1, tag = 1)
                pMapC = comm.recv(source = rank + 2, tag = 2)
                pMapD = comm.recv(source = rank + 3, tag = 3)
                pMapE = comm.recv(source = rank + 4, tag = 4)
            
                pMaps = SNN.states(Is1[u], [pMap1, pMapB, pMapC, pMapD, pMapE])
            
                sMap1 = pMaps[0]
                sMapB = pMaps[1]
                sMapC = pMaps[2]
                sMapD = pMaps[3]
                sMapE = pMaps[4]
            
                comm.send(sMapB, dest = rank + 1, tag = 1)
                comm.send(sMapC, dest = rank + 2, tag = 2)
                comm.send(sMapD, dest = rank + 3, tag = 3)
                comm.send(sMapE, dest = rank + 4, tag = 4)
        
            if (rank % 5) != 0:
                place = (rank // 5) * 5
                sMap1 = comm.recv(source = place, tag = (rank % 5))
            print("f2")
            fMap2 = parallel.generate_feature_map(networkS[0][1][rem], sMap1)
            pMap2 = network.max_pooling(fMap2)
        
            if (rank % 5) != 0:
                place = (rank // 5) * 5
                comm.send(pMap2, dest = place, tag = (rank % 5))
    
            if (rank % 5) == 0:
    
                pMapB = comm.recv(source = rank + 1, tag = 1)
                pMapC = comm.recv(source = rank + 2, tag = 2)
                pMapD = comm.recv(source = rank + 3, tag = 3)
                pMapE = comm.recv(source = rank + 4, tag = 4)
                pMaps = SNN.states(Is1[u], [pMap2, pMapB, pMapC, pMapD, pMapE])
            
                sMap2 = pMaps[0]
                sMapB = pMaps[1]
                sMapC = pMaps[2]
                sMapD = pMaps[3]
                sMapE = pMaps[4]
            
                comm.send(sMapB, dest = rank + 1, tag = 1)
                comm.send(sMapC, dest = rank + 2, tag = 2)
                comm.send(sMapD, dest = rank + 3, tag = 3)
                comm.send(sMapE, dest = rank + 4, tag = 4)
    
            if (rank % 5) != 0:
                place = (rank // 5) * 5
                sMap2 = comm.recv(source = place, tag = (rank % 5))
    
            print("f3")
            place1 = rem * 2
            place2 = (rem * 2) + 1
            fMap3A = parallel.generate_feature_map(networkS[0][2][place1], sMap2)
            fMap3B = parallel.generate_feature_map(networkS[0][2][place2], sMap2)
        
            pMap3A = network.max_pooling(fMap3A)
            pMap3B = network.max_pooling(fMap3B)
        
            if (rank % 5) != 0:
                place = (rank // 5) * 5
                comm.send(pMap3A, dest = place, tag = 0)
                comm.send(pMap3B, dest = place, tag = 1)
        
            if (rank % 5) == 0:
                lstA = [pMap3A]
                lstB = [pMap3B]
                for srcs in range(1, 5):
                    pMapA = comm.recv(source = rank + srcs, tag = 0)
                    pMapB = comm.recv(source = rank + srcs, tag = 1)
                    lstA.append(pMapA)
                    lstB.append(pMapB)
            
                pMapsA = SNN.states(Is1[u], lstA)
                pMapsB = SNN.states(Is1[u], lstB)
            
                sMap3A = pMapsA[0]
                sMap3B = pMapsB[0]
                
                for dests in range(1, 5):
                    comm.send(pMapsA[dests], dest = rank + dests, tag = 0)
                    comm.send(pMapsB[dests], dest = rank + dests, tag = 1)
            
            if (rank % 5) != 0:
                place = (rank // 5) * 5
                sMap3A = comm.recv(source = place, tag = 0)
                sMap3B = comm.recv(source = place, tag = 1)
        #
            print("f4")
            placeA = rem * 4
            placeB = (rem * 4) + 1
            placeC = (rem * 4) + 2
            placeD = (rem * 4) + 3
        
            fMap4A = parallel.generate_feature_map(networkS[0][3][placeA], sMap3A)
            fMap4B = parallel.generate_feature_map(networkS[0][3][placeB], sMap3A)
            fMap4C = parallel.generate_feature_map(networkS[0][3][placeC], sMap3B)
            fMap4D = parallel.generate_feature_map(networkS[0][3][placeD], sMap3B)
        
            pMap4A = network.max_pooling(fMap4A)
            pMap4B = network.max_pooling(fMap4B)
            pMap4C = network.max_pooling(fMap4C)
            pMap4D = network.max_pooling(fMap4D)
        
            if (rank % 5) != 0:
                place = (rank // 5) * 5
                comm.send(pMap4A, dest = place, tag = 0)
                comm.send(pMap4B, dest = place, tag = 1)
                comm.send(pMap4C, dest = place, tag = 2)
                comm.send(pMap4D, dest = place, tag = 3)
        
            if (rank % 5) == 0:
                lstA = [pMap4A]
                lstB = [pMap4B]
                lstC = [pMap4C]
                lstD = [pMap4D]
                
                for srcs in range(1, 5):
                    pMapA = comm.recv(source = rank + srcs, tag = 0)
                    pMapB = comm.recv(source = rank + srcs, tag = 1)
                    pMapC = comm.recv(source = rank + srcs, tag = 2)
                    pMapD = comm.recv(source = rank + srcs, tag = 3)
                
                    lstA.append(pMapA)
                    lstB.append(pMapB)
                    lstC.append(pMapC)
                    lstD.append(pMapD)
            
                pMapsA = SNN.states(Is1[u], lstA)
                pMapsB = SNN.states(Is1[u], lstB)
                pMapsC = SNN.states(Is1[u], lstC)
                pMapsD = SNN.states(Is1[u], lstD)
            
                sMap4A = pMapsA[0]
                sMap4B = pMapsB[0]
                sMap4C = pMapsC[0]
                sMap4D = pMapsD[0]
                
                for dests in range(1, 5):      
                    comm.send(sMap4A, dest = rank + dests, tag = 0)
                    comm.send(sMap4B, dest = rank + dests, tag = 1)
                    comm.send(sMap4C, dest = rank + dests, tag = 2)
                    comm.send(sMap4D, dest = rank + dests, tag = 3)
            
            if (rank % 5) != 0:
                place = (rank // 5) * 5
                sMap4A = comm.recv(source = place, tag = 0)
                sMap4B = comm.recv(source = place, tag = 1)
                sMap4C = comm.recv(source = place, tag = 2)
                sMap4D = comm.recv(source = place, tag = 3)
        #
            if (rank % 16) != 0:
                div = rank - (rank % 16)
                comm.send([sMap4A, sMap4B, sMap4C, sMap4D, sMap3A, sMap3B, sMap2, sMap1], dest = div)
        
            if (rank % 16) == 0:
                sMaps4 = [sMap4A, sMap4B, sMap4C, sMap4D]
                sMaps3 = [sMap3A, sMap3B]
                sMaps2 = [sMap2]
                sMaps1 = [sMap1]
                for k in range(1, 16):
                    sMap = comm.recv(source = rank + k)
                    sMaps4.append(sMap[0])
                    sMaps4.append(sMap[1])
                    sMaps4.append(sMap[2])
                    sMaps4.append(sMap[3])
                    sMaps3.append(sMap[4])
                    sMaps3.append(sMap[5])
                    sMaps2.append(sMap[6])
                    sMaps1.append(sMap[7])
                print("fc")
                err = FunctionalNetwork.forward_pass(networkS, As1[u][rank // 16], sMaps4)
                if rank != 0:
                	comm.send(err, dest = 0)
            if rank == 0:
            	error = err
            	for i in range(1, 5):
            		new = comm.recv(source = i * 16)
            		error += new
            	errors.append(error)
    
if rank == 0:        	
	csvfile = open("testing.csv", "a+")
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(errors)
	csvfile.close()
