import numpy as np
import cv2
import random
import math
from algorithm import algorithm
from techniques import techniques
from SNN import SNN
from network_FC_CF_II import network
from optimizerMVI import optimizerMVI
from Modifications import Modifications
import time
from mpi4py import MPI
from optimizerDP_V import optimizerDP
from processes_VIII import ROI_loss
import csv

class FunctionalNetwork:
    def __init__(self):
        pass
        
    def BP(networkNS, loss, alpha, filter_matrix, sMaps4, sMaps3, sMaps2, sMaps1, rank, rho, reverseMatrix):
        print("4")
        sect = rank // 16
        mod = rank % 16
        
        orig_cv = networkNS.copy()
        
        f4 = [networkNS[0][3][mod * 4], networkNS[0][3][(mod * 4) + 1], networkNS[0][3][(mod * 4) + 2], networkNS[0][3][(mod * 4) + 3]]
        f3 = [networkNS[0][2][mod * 2], networkNS[0][2][(mod * 2) + 1]]
        s4 = [sMaps4[mod * 4], sMaps4[(mod * 4) + 1], sMaps4[(mod * 4) + 2], sMaps4[(mod * 4) + 3]]
        s3 = [sMaps3[mod * 2], sMaps3[(mod * 2) + 1]]
        (r4, r3, r2, r1) = optimizerDP.reverseConvolutional(networkNS[0][0][mod], networkNS[0][1][mod], f3, f4, reverseMatrix, sMaps1[mod], sMaps2[mod], s3, s4)
        
        for j in range(4):
            place = (4 * mod) + j
            conv = algorithm.anticonvolution(reverseMatrix, np.array(sMaps4[place]), 1)
            delta = network.multiply(conv, alpha * rho)
            zeta = optimizerDP.directional_partials(delta, sMaps4[place], r4[j], networkNS[0][3][place])
            networkNS[0][3][place] = np.subtract(networkNS[0][3][place], zeta)
        cRow = []
        print("3")
        for j in range(2):
            fMap = network.anti_pool(np.array(filter_matrix), 16, 16)
            place = (2 * mod) + j
            filter_place1 = 2 * place
            filter_place2 = filter_place1 + 1
            avg = network.multiply(network.add([networkNS[0][3][filter_place1], networkNS[0][3][filter_place2]]), 0.5)
            avgS = network.multiply(network.add([sMaps4[filter_place1], sMaps4[filter_place2]]), 0.5)
            apS = network.anti_pool(np.array(avgS), 16, 16)
            
            cMap = algorithm.convolution(avg, fMap, 1) # this is l
            
            tau = optimizerMVI.optimize(cMap, sMaps3[place], apS, networkNS[0][2][place], alpha * rho) # OPTIMIZER
            zeta = optimizerDP.directional_partials(tau, sMaps3[place], r3[j], networkNS[0][2][place])
            networkNS[0][2][place] = np.subtract(networkNS[0][2][place], zeta)
            cRow.append(cMap)
        print("12")
        
        filter_place1 = 2 * mod
        filter_place2 = filter_place1 + 1
        fMap1 = network.anti_pool(np.array(cRow[0]), 32, 32)
        fMap2 = network.anti_pool(np.array(cRow[1]), 32, 32)
        avg = network.multiply(network.add([networkNS[0][2][filter_place1], networkNS[0][2][filter_place2]]), 0.5)
        avgF = network.multiply(network.add([fMap1, fMap2]), 0.5)
        avgS = network.multiply(network.add([sMaps3[filter_place1], sMaps3[filter_place2]]), 0.5)
        apS = network.anti_pool(np.array(avgS), 32, 32)
        dMap = algorithm.convolution(avg, avgF, 1)
        
        tau = optimizerMVI.optimize(dMap, sMaps2[mod], apS, networkNS[0][1][mod], alpha * rho)
        zeta = optimizerDP.directional_partials(tau, sMaps2[mod], r2, networkNS[0][1][mod])
        networkNS[0][1][mod] = np.subtract(networkNS[0][1][mod], zeta)
        
        fMap = network.anti_pool(np.array(dMap), 64, 64)
        eMap = algorithm.convolution(networkNS[0][1][mod], fMap, 2)
            
        tau = optimizerMVI.optimize(eMap, sMaps1[mod], network.anti_pool(sMaps2[mod], 64, 64), networkNS[0][0][mod], alpha * rho)
        zeta = optimizerDP.directional_partials(tau, sMaps1[mod], r1, networkNS[0][0][mod])
        networkNS[0][0][mod] = np.subtract(networkNS[0][0][mod], zeta)
        
        return (networkNS, loss)
    
    def PR(filters): # corresponds to all CV1s, or CVxs
        print("p")
        # assign these jobs to four different processors
        a = np.array(network.signed_ln(filters[0]))
        b = np.array(network.signed_ln(filters[1]))
        c = np.array(network.signed_ln(filters[2]))
        d = np.array(network.signed_ln(filters[3]))
        e = np.array(network.signed_ln(filters[4]))
                    
        matrices = techniques(a, b, c, d, e).plane_recurrence()
                    
        return matrices
