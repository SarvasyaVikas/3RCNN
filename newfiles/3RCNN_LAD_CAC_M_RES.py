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
from processes_XIV_M import process
from backpropfc_VI import BPFC as bp
from RCNN_FUNC_2 import RCNN_TR
from RCNN_FUNC_3 import RCNN_TE
# Uses DP and MVI

mult = SETS.data()

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
step = 0
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
    
    csvfile = open("LAD_NEW_NETWORKS_RESULTS_8.csv", "a+")
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(lst)
    csvfile.close()
    

def divide(lst, val):
    fin = []
    for i in range(len(lst)):
        new = lst[i] / val
        fin.append(new)
    return fin

def data(ptn, it = 0):
    rows = []
    with open("REVALS_9_UPD_ANNO.csv") as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ",")
        for row in csvreader:
            if int(row[1]) == ptn:
                rows.append(row)
    images = []
    actuals = []
    o = []
    for i in range(len(rows) - 4):
        vals1 = [float(rows[i][3]), float(rows[i][4]), float(rows[i][5]), float(rows[i][6])]
        vals2 = [float(rows[i + 1][3]), float(rows[i + 1][4]), float(rows[i + 1][5]), float(rows[i + 1][6])]
        vals3 = [float(rows[i + 2][3]), float(rows[i + 2][4]), float(rows[i + 2][5]), float(rows[i + 2][6])]
        vals4 = [float(rows[i + 3][3]), float(rows[i + 3][4]), float(rows[i + 3][5]), float(rows[i + 3][6])]
        vals5 = [float(rows[i + 4][3]), float(rows[i + 4][4]), float(rows[i + 4][5]), float(rows[i + 4][6])]
        if it == 0:
        	actuals.append([divide(vals1, 1024), divide(vals2, 1024), divide(vals3, 1024), divide(vals4, 1024), divide(vals5, 1024)])
        else:
        	actuals.append([0, 0, 0, 0, 0])

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
        for k in range(5):
            img = cv2.imread("SCL/NGCT{}_SCL/ngct{}_{}.png".format(ptn, ptn, i + k + 1), 0)
            ol.append(img)
        o.append(ol)
        # for k in range(5):
        #    images[-1][k] = ab.detectExternal(images[-1][k])
        
    return (images, actuals, o)

nn = [networkS, networkS, networkS, networkS, networkS]
neurals = [nn.copy()]

for it in range(2):
    beg = 0 if it == 0 else 60
    newlen = 60 if it == 0 else len(mult)
    for j in range(beg, newlen):
        CACscore = 0
    
        cacscores = open("NGCT_CAC_Scores.csv", "r")
        cacreader = csv.reader(cacscores)
        cacrow = []
        for row in cacreader:
            if row[0] == "NGCT{}".format(mult[j][0]):
                cacrow = row
        FOV = float(cacrow[9])
        CACactual = float(cacrow[2])
        cacscores.close()

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
        (Is1, As1, f) = data(mult[j][0], it)
        print(len(Is1))
        print("data gen")
        sve = open("record.csv", "a+")
        if (rank % 16) == 0 and rank != 0:
            comm.send(networkS, dest = 0)
        if rank == 0:
            n16 = comm.recv(source = 16)
            n32 = comm.recv(source = 32)
            n48 = comm.recv(source = 48)
            n64 = comm.recv(source = 64)
            nn = [networkS, n16, n32, n48, n64]

            for i in range(1, 80):
                comm.send(nn, dest = i)
        else:
            nn = comm.recv(source = 0)

        for i in range(mult[j][1], mult[j][2] - 4):
            start = time.time()
            if it == 0:
            	(sMaps1, sMaps2, sMaps3, sMaps4) = RCNN_TR.NN(networkS, Is1, mult, j, i, alpha, losses, rank)
            	(losses, networkS, CACscore, step) = RCNN_TR.VR(rank, networkS, Is1, As1, mult, j, i, alpha, losses, minval, maxval, FOV, CACscore, start, CACactual, scan_start, nn, step, f, "LAD", sMaps4)
            else:
            	(sMaps1, sMaps2, sMaps3, sMaps4) = RCNN_TE.NN(networkS, Is1, mult, j, i, alpha, losses, rank)
            	(losses, networkS, CACscore, step) = RCNN_TE.VR(rank, networkS, Is1, mult, j, i, alpha, losses, minval, maxval, FOV, CACscore, start, CACactual, scan_start, nn, step, f, "LAD", sMaps4)
            (csv.writer(sve)).writerow(['complete'])
            networkS = bp.REG(networkS)
            end = time.time()
            diff = end - start
            print("Time per Slice: {}".format(diff))
        sve.close()

        losses.pop(0)
        losses.pop(0)
        losslst = open("losslst.csv", "a+")
        (csv.writer(losslst)).writerow(losses)
        losslst.close()

        if (rank % 16) == 0:
            save(nn[rank % 16], "LAD_NEW_NETWORKS_FULL_RESULTS", losses)
