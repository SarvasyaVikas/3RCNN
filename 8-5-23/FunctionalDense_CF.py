import numpy as np
import cv2
import random
import math
from algorithm import algorithm
from techniques import techniques
from SNN import SNN
from network_FC_CF_III import network
from optimizerMVI import optimizerMVI
from Modifications import Modifications
import time
from mpi4py import MPI
from optimizerCF_II import optimizerCF as CF
from processes_VIII import ROI_loss
from processes_VIII import convolutionalModifications as cM
import csv

class DENSE:
    def __init__(self):
        pass

    def FC(networkNS, actual, cnInput, alpha):
        print("fp")
        
        # artificial forward propagation
        fcInput = network.forward_pass(networkNS[1][0], cnInput, networkNS[1][1], 0)
        fcOutput = network.forward_pass(networkNS[1][1], fcInput, [0, 0, 0, 0], 1)
        softmax = network.forward_pass(networkNS[2], fcOutput, [0, 0], 0)
        
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
        adj = err * (2 ** 14)
        (networkNS[2], P7, WR7, fdelta7) = network.backprop(networkNS[2], adj, 0, alpha, softmax)
        (networkNS[1][1], P6, WR6, fdelta6) = network.backprop(networkNS[1][1], (P7 + WR7), 1, alpha, fcOutput)
        (networkNS[1][0], P5, WR5, fdelta5) = network.backprop(networkNS[1][0], (P6 + WR7 + WR6), 0, alpha, fcInput)
        
        (L0, L1, L2) = CF.reverseCF([networkNS[1][0], networkNS[1][1], networkNS[2]], softmax)
        networkNS[2] = CF.directional_partials(fdelta7, fcOutput, L2, networkNS[2])
        networkNS[1][1] = CF.directional_partials(fdelta6, fcInput, L1, networkNS[1][1])
        networkNS[1][0] = CF.directional_partials(fdelta5, cnInput, L0, networkNS[1][0])
        
        return (networkNS, loss)
        
