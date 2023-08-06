import numpy as np
import csv
from network import network
import time
from mpi4py import MPI
from backpropfc_I import BPFC as full
# Uses DP and MVI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
alpha = 0.01
epochs = 100

rM = open("reverseMatrices.csv", "r")
rMread = csv.reader(rM)
rMarr = list(rMread)
losses = []
for j in range(epochs):
    if rank == 0:
        print(j)
    for i in range(127, len(rMarr), 128):
        start = time.time()
        row = rMarr[i - rank]
        for i in range(len(row)):
            row[i] = float(row[i])
        print(len(row))
        cnInput = row[0:64]
        act = row[64:68]
        sftm = row[68:70]
        (networkS, tse, mse) = full.FULL(networkS, cnInput, sftm, alpha, act, updI, ofI)
        end = time.time()
        diff = end - start
        losses.append(tse)
        losses.append(mse)
        losses.append(diff)

L = open("error_I.csv", "a+")
(csv.writer(L)).writerow(losses)
L.close()
