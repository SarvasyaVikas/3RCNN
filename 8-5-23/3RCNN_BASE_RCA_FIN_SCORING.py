import cv2
import numpy as np
import csv
import fnmatch
import os
import csv
from FunctionalNetwork_BASE_FIN_SCORING import FunctionalNetwork
from network import network
import time
from Modifications import Modifications
from mpi4py import MPI
from SNN import SNN
from parallel import parallel
from sets_RCA import SETS
from algorithm import algorithm
from processes_VIII import process
from processes_VIII import antibone as ab
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
    
    csvfile = open("networks.csv", "a+")
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(lst)
    csvfile.close()
    

def divide(lst, val):
    fin = []
    for i in range(len(lst)):
        new = lst[i] / val
        fin.append(new)
    return fin

def data(ptn):
    rows = []
    with open("RCA_VALUES.csv") as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ",")
        for row in csvreader:
            if int(row[1]) == ptn:
                rows.append(row)
    images = []
    actuals = []
    for i in range(len(rows) - 4):
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
        
        for i in range(5):
            images[-1][i] = ab.detectExternal(images[-1][i])

    return (images, actuals)

nn = [networkS, networkS, networkS, networkS, networkS]
neurals = [nn.copy()]

for epoch in range(epochs):
    for j in range(len(mult)):

        CACscore = 0
        cacscores = open("NGCT_CAC_Scores.csv", "r")
        cacreader = csv.reader(cacscores)
        cacrow = []
        for row in cacreader:
            if row[0] == "NGCT{}".format(mult[j][0]):
                cacrow = row
        FOV = float(cacrow[9])
        CACactual = float(cacrow[5])
        cacscores.close()

        minmaxes = open("minmaxes.csv", "r")
        minmaxreader = csv.reader(minmaxes)
        minval = 0
        maxval = 0
        num = 0
        for row in minmaxreader:
            if num == mult[j][0]:
                minval = int(row[0])
                maxval = int(row[1])
            num += 1

        print(mult[j][0])
        print("losses")
        losses = [(10 ** 5), (10 ** 5)]
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
        (Is1, As1) = data(mult[j][0])
        print("data gen")
        for i in range(mult[j][1], mult[j][2] - 4):
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
                (networkS, error, filter_matrix, rho, reverseMatrix, zeros, softmax) = FunctionalNetwork.FC(networkS, As1[u][rank // 16], alpha, losses[-1], sMaps4)
                splice_vals = [min(zeros[0], zeros[2]), min(zeros[1], zeros[3]), max(zeros[0], zeros[2]), max(zeros[1], zeros[3])]
                splice_vals = np.array(splice_vals, dtype = np.uint8)
                print(splice_vals)

                splices = open("splices.csv", "a+")
                splicewriter = csv.writer(splices)
                splicewriter.writerow(["NGCT{}_BASE".format(mult[j][0]), i, splice_vals[0], splice_vals[1], splice_vals[2], splice_vals[3], As1[u][rank // 16][0], As1[u][rank // 16][1], As1[u][rank // 16][2], As1[u][rank // 16][3], softmax[0], softmax[1]])
                splices.close()

                if rank == 0:
                    score = process.cut_scoring(Is1[u][rank // 16][splice_vals[1]:splice_vals[3], splice_vals[0]:splice_vals[2]], minval, maxval, FOV)
                    CACscore += score

                for k in range(1, 16):
                    comm.send((networkS, error, filter_matrix, sMaps4, sMaps3, sMaps2, sMaps1, rho, reverseMatrix), dest = rank + k)
            
            if (rank % 16) != 0:
                place = (rank // 16) * 16
                (networkS, error, filter_matrix, sMaps4, sMaps3, sMaps2, sMaps1, rho, reverseMatrix) = comm.recv(source = place)
            print("bp")
            (networkS, error) = FunctionalNetwork.BP(networkS, error, alpha, filter_matrix, sMaps4, sMaps3, sMaps2, sMaps1, rank, rho, reverseMatrix)
        #
            if rank != 0:
                comm.send(networkS, dest = 0)
        
            if rank == 0:
                networkFULL = [networkS, networkS, networkS, networkS, networkS]
                for k in range(1, 80):
                    section = k // 16
                    part = k % 16
                    place1 = (2 * part)
                    place2 = place1 + 1
                    placeA = (4 * part)
                    placeB = placeA + 1
                    placeC = placeB + 1
                    placeD = placeC + 1
                    networkS = comm.recv(source = k)
                    networkFULL[section][0][0][part] = networkS[0][0][part]
                    networkFULL[section][0][1][part] = networkS[0][1][part]
                
                    networkFULL[section][0][2][place1] = networkS[0][2][place1]
                    networkFULL[section][0][2][place2] = networkS[0][2][place2]
                
                    networkFULL[section][0][3][placeA] = networkS[0][3][placeA]
                    networkFULL[section][0][3][placeB] = networkS[0][3][placeB]
                    networkFULL[section][0][3][placeC] = networkS[0][3][placeC]
                    networkFULL[section][0][3][placeD] = networkS[0][3][placeD]
                
                    if part == 0:
                        networkFULL[section][1] = networkS[1]
                        networkFULL[section][2] = networkS[2]
    
                    comm.send(networkFULL, dest = k)
    
            if rank != 0:
                networkFULL = comm.recv(source = 0)
    
            if rank < 32:
                sect = rank // 16
                rem = rank % 16
                pR1 = FunctionalNetwork.PR([networkFULL[0][0][sect][rem], networkFULL[1][0][sect][rem], networkFULL[2][0][sect][rem], networkFULL[3][0][sect][rem], networkFULL[4][0][sect][rem]])
                pR2 = FunctionalNetwork.PR([networkFULL[0][0][2][rank], networkFULL[1][0][2][rank], networkFULL[2][0][2][rank], networkFULL[3][0][2][rank], networkFULL[4][0][2][rank]])
                for k in range(5):
                    networkFULL[k][0][sect][rem] = pR1[k]
                    networkFULL[k][0][2][rank] = pR2[k]
    
                if rank != 0:
                    comm.send(networkFULL, dest = 0)
            elif rank < 64:
                place1 = (rank - 32) * 2
                place2 = place1 + 1
                pR1 = FunctionalNetwork.PR([networkFULL[0][0][3][place1], networkFULL[1][0][3][place1], networkFULL[2][0][3][place1], networkFULL[3][0][3][place1], networkFULL[4][0][3][place1]])
                pR2 = FunctionalNetwork.PR([networkFULL[0][0][3][place2], networkFULL[1][0][3][place2], networkFULL[2][0][3][place2], networkFULL[3][0][3][place2], networkFULL[4][0][3][place2]])
                for k in range(5):
                        networkFULL[k][0][3][place1] = pR1[k]
                        networkFULL[k][0][3][place2] = pR2[k]
    
                comm.send(networkFULL, dest = 0)
        
            if rank == 0:
                for k in range(1, 32):
                    mod = k % 16
                    sect = k // 16
                    pR = comm.recv(source = k)
                    for l in range(5):
                        networkFULL[l][0][sect][mod] = pR[l][0][sect][mod]
                        networkFULL[l][0][2][k] = pR[l][0][2][k]
                for k in range(32, 64):
                    pR = comm.recv(source = k)
                    place1 = 2 * (k - 32)
                    place2 = place1 + 1
                    for l in range(5):
                        networkFULL[l][0][3][place1] = pR[l][0][3][place1]
                        networkFULL[l][0][3][place2] = pR[l][0][3][place2]
                    
                for k in range(1, 80):
                    comm.send(networkFULL, dest = k)
        
            print("reconstruction")
            if rank != 0:
                networkFULL = comm.recv(source = 0)
            
        #
            if error < losses[-1]:
                nn = networkFULL.copy()
            losses.append(error)
            
            networkS = networkFULL[rank // 16].copy()
    
            end = time.time()
            print(end - start)
        
            if rank == 0:
                for n in range(5):
                    save(networkFULL[n], "RCA_PLACE{}_SECTION_{}".format(u, n), losses)
            print("save")
    
        end_scan = time.time()
        print("end")
        print(end_scan - scan_start)
        if rank == 0:
            for n in range(5):
                save(nn[n], "RCA_FIN_SECTION{}".format(n), losses)

        if rank == 0:
            scores = open("scores.csv", "a+")
            avg = (CACactual + CACscore) / 2
            mserror = 0
            if avg != 0:
                mserror = ((CACactual - CACscore) / avg) ** 2
            lst = ["NGCT{}_BASE".format(mult[j][0]), CACscore, CACactual, mserror, abs(CACscore - CACactual)]
            scoreswriter = csv.writer(scores)
            scoreswriter.writerow(lst)
            scores.close()
