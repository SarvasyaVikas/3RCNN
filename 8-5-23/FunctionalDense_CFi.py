import numpy as np
import cv2
import random
import math
from algorithm import algorithm
from techniques import techniques
from SNN import SNN
from network_FC_CF_IV import network
from optimizerMVI import optimizerMVI
from Modifications import Modifications
import time
from mpi4py import MPI
from optimizerCF_III import optimizerCF as CF
from optimizerDP_IV import optimizerDP
from processes_VIII import ROI_loss
from processes_VIII import convolutionalModifications as cM
import csv

class DENSE:
    def __init__(self):
        pass

    def FC(networkNS, actual, sMaps4, alpha, prev):
        print("fp")
        
        # artificial forward propagation
        cnInput = network.connector_layer(sMaps4)
        (fcInput, I1) = network.forward_pass(networkNS[1][0], cnInput, networkNS[1][1], 0, True)
        (fcOutput, I2) = network.forward_pass(networkNS[1][1], fcInput, [0, 0, 0, 0], 1, True)
        (softmax, I3) = network.forward_pass(networkNS[2], fcOutput, [0, 0], 0, True)
        
        inactivated = []
        for el in I1:
            inactivated.append(el)
        for el in I2:
            inactivated.append(el)
        for el in I3:
            inactivated.append(el)

        invals = open("inactive_vals.csv", "a+")
        (csv.writer(invals)).writerow(inactivated)
        invals.close()

        print("fcb")
        zeros = fcOutput
        
        tot = (np.e ** softmax[0]) + (np.e ** softmax[1])
        val = (np.e ** softmax[1]) / tot

        if val > 0.736:
            zeros = [0, 0, 0, 0]
        else:
            zeros = fcOutput
        
        err = network.mseINDIV(actual, zeros)   
        loss = sum(err)
        rho = network.direction(prev, loss)
        adj = err * (2 ** 14)
        (networkNS[2], P7, WR7, fdelta7) = network.backprop(networkNS[2], adj, 0, alpha, softmax)
        (networkNS[1][1], P6, WR6, fdelta6) = network.backprop(networkNS[1][1], (adj + P7 + WR7), 1, alpha, fcOutput)
        (networkNS[1][0], P5, WR5, fdelta5) = network.backprop(networkNS[1][0], (adj + P6 + WR7 + WR6), 0, alpha, fcInput)
        
        (L0, L1, L2) = CF.reverseCF([networkNS[1][0], networkNS[1][1], networkNS[2]], softmax)
        networkNS[2] = CF.directional_partials(fdelta7, fcOutput, L2, networkNS[2])
        networkNS[1][1] = CF.directional_partials(fdelta6, fcInput, L1, networkNS[1][1])
        networkNS[1][0] = CF.directional_partials(fdelta5, cnInput, L0, networkNS[1][0])
        
        networkNS[2] = CF.direct_regularization(networkNS[2], WR7)
        networkNS[1][1] = CF.direct_regularization(networkNS[1][1], WR6)
        networkNS[1][0] = CF.direct_regularization(networkNS[1][0], WR5)

        filter_loss = P5
        filter_matrix = []
        for i in range(8):
            filter_row = []
            for j in range(8):
                place = (8 * i) + j
                filter_row.append(filter_loss[place])
            filter_matrix.append(filter_row)

        reverseMatrix = optimizerDP.reverseDense(networkNS[1], fcOutput)

        return (networkNS, loss, filter_matrix, rho, reverseMatrix, zeros)
        
