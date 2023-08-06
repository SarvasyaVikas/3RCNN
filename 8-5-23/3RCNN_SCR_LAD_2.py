import cv2
import numpy as np
import csv
import fnmatch
import os
import csv
from FunctionalNetwork_ROI_LOSS_VI import FunctionalNetwork
from FunctionalDense_CFiR import DENSE as cfi
from network import network
import time
from Modifications import Modifications
from mpi4py import MPI
from SNN import SNN
from parallel import parallel
from sets_LAD_SCL_NEW import sets_LAD_SCL as SETS
from algorithm import algorithm
from processes_VIII import antibone as ab
from processes_X_M import process
from backpropfc_IV import BPFC as bp
# from RCNN_FUNC import RCNN
# Uses DP and MVI

mult = SETS.data()
# 64 processors
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("generation")
"""
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
"""
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
    
    csvfile = open("networks_I.csv", "a+")
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
    with open("REVALS_FC_2.csv") as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ",")
        for row in csvreader:
            if int(row[1]) == ptn:
                rows.append(row)
    images = []
    o = []
    actuals = []
    for i in range(len(rows) - 4):
        vals1 = [float(rows[i][7]), float(rows[i][8]), float(rows[i][9]), float(rows[i][10])]
        vals2 = [float(rows[i + 1][7]), float(rows[i + 1][8]), float(rows[i + 1][9]), float(rows[i + 1][10])]
        vals3 = [float(rows[i + 2][7]), float(rows[i + 2][8]), float(rows[i + 2][9]), float(rows[i + 2][10])]
        vals4 = [float(rows[i + 3][7]), float(rows[i + 3][8]), float(rows[i + 3][9]), float(rows[i + 3][10])]
        vals5 = [float(rows[i + 4][7]), float(rows[i + 4][8]), float(rows[i + 4][9]), float(rows[i + 4][10])]
        actuals.append([divide(vals1, 1024), divide(vals2, 1024), divide(vals3, 1024), divide(vals4, 1024), divide(vals5, 1024)])

        left = 160
        right = 416
        top = 96
        bottom = 352

        try:
            imageO = cv2.imread("SCL/NGCT{}_SCL/ngct{}_{}.png".format(ptn, ptn, i + 1), 0)
            (h, w) = imageO.shape[:2]
            left = 5 * w // 16
            right = 13 * w // 16
            top = 3 * h // 16
            bottom = 11 * h // 16
        except:
            pass

        try:
            image1 = cv2.imread("SCL/NGCT{}_SCL/pooled_ngct{}_{}.png".format(ptn, ptn, i + 1), 0)
            image1 = cv2.resize(image1, (128, 128))
        except:
            imageO = cv2.imread("SCL/NGCT{}_SCL/ngct{}_{}.png".format(ptn, ptn, i + 1), 0)
            imageR = imageO[top:bottom, left:right]
            image1 = algorithm.max_pooling(imageR)
            image1 = cv2.resize(image1, (128, 128))
        try:
            image2 = cv2.imread("SCL/NGCT{}_SCL/pooled_ngct{}_{}.png".format(ptn, ptn, i + 2), 0)
            image2 = cv2.resize(image2, (128, 128))
        except:
            imageO = cv2.imread("SCL/NGCT{}_SCL/ngct{}_{}.png".format(ptn, ptn, i + 2), 0)
            imageR = imageO[top:bottom, left:right]
            image2 = algorithm.max_pooling(imageR)
            image2 = cv2.resize(image2, (128, 128))
        try:
            image3 = cv2.imread("SCL/NGCT{}_SCL/pooled_ngct{}_{}.png".format(ptn, ptn, i + 3), 0)
            image3 = cv2.resize(image3, (128, 128))
        except:
            imageO = cv2.imread("SCL/NGCT{}_SCL/ngct{}_{}.png".format(ptn, ptn, i + 3), 0)
            imageR = imageO[top:bottom, left:right]
            image3 = algorithm.max_pooling(imageR)
            image3 = cv2.resize(image3, (128, 128))
        try:
            image4 = cv2.imread("SCL/NGCT{}_SCL/pooled_ngct{}_{}.png".format(ptn, ptn, i + 4), 0)
            image4 = cv2.resize(image4, (128, 128))
        except:
            imageO = cv2.imread("SCL/NGCT{}_SCL/ngct{}_{}.png".format(ptn, ptn, i + 4), 0)
            imageR = imageO[top:bottom, left:right]
            image4 = algorithm.max_pooling(imageR)
            image4 = cv2.resize(image4, (128, 128))
        try:
            image5 = cv2.imread("SCL/NGCT{}_SCL/pooled_ngct{}_{}.png".format(ptn, ptn, i + 5), 0)
            image5 = cv2.resize(image5, (128, 128))
        except:
            imageO = cv2.imread("SCL/NGCT{}_SCL/ngct{}_{}.png".format(ptn, ptn, i + 5), 0)
            imageR = imageO[top:bottom, left:right]
            image5 = algorithm.max_pooling(imageR)
            image5 = cv2.resize(image5, (128, 128))
        # image1 = cv2.resize(image1, (128, 128))
        # image2 = cv2.resize(image2, (128, 128))
        # image3 = cv2.resize(image3, (128, 128))
        # image4 = cv2.resize(image4, (128, 128))
        # image5 = cv2.resize(image5, (128, 128))
        images.append([image1, image2, image3, image4, image5])
        ol = []
        print("ol")
        for k in range(5):
            img = cv2.imread("SCL/NGCT{}_SCL/ngct{}_{}.png".format(ptn, ptn, i + k + 1), 0)
            ol.append(img)
        o.append(ol)
        print("o")
        # for k in range(5):
        #    images[-1][k] = ab.detectExternal(images[-1][k])
        print("ab")
    return (images, actuals, o)


for j in range(len(mult)):
    print("epoch")
    # print(epoch)
    CACscore = 0
    CACscore1 = 0
    print("j")
    cacscores = open("NGCT_CAC_Scores.csv", "r")
    cacreader = csv.reader(cacscores)
    cacrow = []
    for row in cacreader:
        if row[0] == "NGCT{}".format(mult[j][0]):
            cacrow = row
    FOV = float(cacrow[9])
    CACactual = float(cacrow[2])
    cacscores.close()
    if CACactual != 0:
        minmaxes = open("minmaxes.csv", "r")
        minmaxreader = csv.reader(minmaxes)
        minval = 0
        maxval = 0
        num = 0
        for row in minmaxreader:
            if num == mult[j][0]:
                minval = row[0]
                maxval = row[1]
            num += 1
        print("data")
        (Is1, As1, f) = data(mult[j][0])
        print("loop")
        lst = []
        
        for i in range(mult[j][1], mult[j][2] - 4):
            u = i - mult[j][1]
            for k in range(4):
                lst.append(int(As1[u][0][k] * 512))
            cut = f[u][0][int(As1[u][0][1] * 512):int(As1[u][0][3] * 512), int(As1[u][0][0] * 512):int(As1[u][0][2] * 512)]
            scr = process.cut_scoring(cut, FOV, 61 + rank)
            scr1 = process.cut_scoring(cut, FOV, 60 - rank)
            (h, w) = cut.shape[:2]
            lst.append(h * w)
            CACscore += scr
            CACscore1 += scr1
            print("i")
            print(i)

        print("perdiff")
        perdiff = abs(CACscore - CACactual) / (CACactual + CACscore) * 200
        rec = open("scoresCH61.csv", "a+")
        new = ["NGCT{}".format(mult[j][0]), "SCR_LAD", "61+", rank, 61 + rank, CACscore, CACactual, perdiff]
        new1 = new.copy()
        new1[2] = "61-"
        new1[4] = 61 - rank
        (csv.writer(rec)).writerow(new)
        (csv.writer(rec)).writerow(new1)
        print("written")
        rec.close()

class RCNN:
    def __init__(self):
        pass
    
    def NN(networkS, Is1, As1, mult, j, i, alpha, losses, rank):
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
            
            return (sMaps1, sMaps2, sMaps3, sMaps4)

            
    def VR(rank, networkS, Is1, As1, mult, j, i, alpha, losses, sMaps1, sMaps2, sMaps3, sMaps4):
        if (rank % 16) == 0:
            init_inputs = network.connector_layer(sMaps4[(rank):(rank + 16)])
            # init inputs
            (fc5, fc6, fc7, wr5, wr6, wr7) = bp.INIT_PASS(networkS, init_inputs)

            zeros = fc6.copy()
            if fc7[0] > fc7[1]:
                zeros = [0, 0, 0, 0]
            
            err = network.mseINDIV(zeros, As1[u][rank // 16])
            rho = network.direction(losses[-1], sum(err))
            wrt = (wr5 + wr6 + wr7) / (2 ** 14)
            wrerr = err + wrt
            # loss4 is err
            loss16 = networkADV.loss_pass(networkS[1][1], err, networkS[2], 1)
            loss64 = networkADV.loss_pass(networkS[1][0], loss16, networkS[1][1])
            # rM 1 is loss64

            losso16 = networkADV.loss_pass(networkS[1][1], err, networkS[2], 1)
            losso64 = networkADV.loss_pass(networkS[1][0], losso16, networkS[1][1])
            # rM 2 is losso64

            fin_outputs = [0, 0] if sum(As1[u][rank // 16]) == 0 else [0.49, 0.51]
            bp7 = bp.NMBP(networkS[2], fin_outputs)
            bp6 = bp.NMBP(networkS[1][1], bp7, 1)
            bp5 = bp.NMBP(networkS[1][0], bp6)
            filter_loss = bp5

            filter_matrix = []
            for i in range(8):
                filter_row = []
                for j in range(8):
                    place = (8 * i) + j
                    filter_row.append(filter_loss[place])
                filter_matrix.append(filter_row)

            orig = networkS.copy()
            (upd, updE) = FunctionalNetwork.BP(networkS, err, alpha, filter_matrix, sMaps4, sMaps3, sMaps2, sMaps1, rank, rho, loss64)
            (ovf, ovfE) = FunctionalNetwork.BP(networkS, err, alpha, filter_matrix, sMaps4, sMaps3, sMaps2, sMaps1, rank, rho, losso64)

            for i in range(1, 16):
                comm.send((upd, Is1, As1, mult, j, i, alpha, losses), dest = rank + i, tag = i)

        if (rank % 16) != 0:
            (upd, Is1, As1, mult, j, i, alpha, losses) = comm.recv(source = (rank - (rank % 16)), tag = (rank % 16))
        
            RCNN.NN(upd, Is1, As1, mult, j, i, alpha, losses, rank)
            RCNN.NN(ovf, Is1, As1, mult, j, i, alpha, losses, rank)

        if (rank % 16) == 0:
            (uMap1, uMap2, uMap3, uMap4) = RCNN.NN(upd, Is1, As1, mult, j, i, alpha, losses, rank)
            (oMap1, oMap2, oMap3, oMap4) = RCNN.NN(ovf, Is1, As1, mult, j, i, alpha, losses, rank)

            u_inputs = network.connector_layer(uMap4[(rank):(rank + 16)])
            o_inputs = network.connector_layer(oMap4[(rank):(rank + 16)])

            (networkS, ann_err, code) = bp.FULL(orig, init_inputs, fin_outputs, alpha, As1[u][rank // 16], u_inputs, o_inputs)
            # this has the completed fully connected parts

            st = [orig, ovf, upd]
            networkS[0] = st[code - 1][0]

            splice_vals = [float(min(zeros[0], zeros[2])), float(min(zeros[1], zeros[3])), float(max(zeros[0], zeros[2])), float(max(zeros[1], zeros[3]))]
            splice_vals = np.array(np.multiply(splice_vals, 128), dtype = np.uint8)
            splices = open("splices_II.csv", "a+")
            splicewriter = csv.writer(splices)
            splicewriter.writerow(["NGCT{}_ROILOSS".format(mult[j][0]), i, splice_vals[0], splice_vals[1], splice_vals[2], splice_vals[3], As1[u][rank // 16][0], As1[u][rank // 16][1], As1[u][rank // 16][2], As1[u][rank // 16][3]]) 
            splices.close()

            if rank != 0:
                comm.send(networkS, dest = 0)

            # start scoring
            if rank == 0:   
                score = process.cut_scoring(Is1[u][rank // 16][splice_vals[1]:splice_vals[3], splice_vals[0]:splice_vals[2]], minval, maxval, FOV)
                CACscore += score
            # end scoring

        if rank == 0:
            networkS2 = comm.recv(source = 16)
            networkS3 = comm.recv(source = 32)
            networkS4 = comm.recv(source = 48)
            networkS5 = comm.recv(source = 64)
            networkFULL = [networkS, networkS2, networkS3, networkS4, networkS5]
            for k in range(1, 80):
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
    
        ##
        end_scan = time.time()
        print("end")
        print(end_scan - scan_start)
        if rank == 0:
            for n in range(5):
                save(nn[n], "RCA_FIN_SECTION{}".format(n), losses)

        if rank == 0:
            avg = (CACactual + CACscore) / 2.0
            mserror = 0
            if avg != 0:
                mserror = ((CACactual - CACscore) / avg) ** 2
            caclst = ["NGCT{}_ROILOSS".format(mult[j][0]), CACscore, CACactual, mserror, abs(CACactual - CACscore)]
            scoresfile = open("scores.csv", "a+")
            scoreswriter = csv.writer(scoresfile)
            print("write")
            scoreswriter.writerow(caclst)
            scoresfile.close()

        return (losses, networkFULL)
