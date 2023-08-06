import numpy as np
import csv
from FunctionalDense_CF import DENSE as Dense
from network import network
import time
from mpi4py import MPI
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
    for i in range(127, len(rMarr), 128):
        start = time.time()
        row = rMarr[i - rank]
        cnInput = row[0:64]
        act = row[64:68]
        sftm = row[68:70]
        (networkS, loss) = Dense.FC(networkS, act, cnInput, alpha)
        end = time.time()
        diff = end - start
        losses.append(loss)
        losses.append(diff)

L = open("error.csv", "a+")
(csv.writer(L)).writerow(losses)
L.close()
