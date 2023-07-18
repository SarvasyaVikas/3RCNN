import numpy as np
import csv
from network import network
import time
from backpropfc_II import BPFC as full
# Uses DP and MVI

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

def ANN(row, N):
    lst = [row]
    for i in range(len(N[1])):
        for j in range(len(N[1][i])):
            for k in range(len(N[1][i][j][0])):
                val = N[1][i][j][0][k]
                lst.append(val)
            val = N[1][i][j][1]
            lst.append(val)

    for i in range(len(N[2])):
        for j in range(len(N[2][i][0])):
            val = N[2][i][0][j]
            lst.append(val)
        val = N[2][i][1]
        lst.append(val)

    ann = open("ann.csv", "a+")
    (csv.writer(ann)).writerow(lst)
    ann.close()


# ABOVE THIS: DO NOT TOUCH
#
#

# This is for RCA
alpha = 0.01
epochs = 100

rM = open("reverseMatrices_ordered.csv", "r")
rMread = csv.reader(rM)
rMarr = list(rMread)
losses = []
for j in range(epochs):
    for i in range(len(rMarr)):
        start = time.time()
        row = rMarr[i]
        for k in range(len(row)):
            row[k] = float(row[k])
        cnInput = row[0:64]
        act = row[64:68]
        sftm = row[68:70]
        # has full capabilities, includes repetitive save and ANN/FC training (complete)
        ANN(i, networkS)
        (networkS, tse, mse) = full.FULL(networkS, cnInput, sftm, alpha, act, updI, ofI)
        end = time.time()
        diff = end - start
        losses.append(tse)
        losses.append(mse)
        losses.append(diff)

L = open("error_ordered.csv", "a+")
(csv.writer(L)).writerow(losses)
L.close()
