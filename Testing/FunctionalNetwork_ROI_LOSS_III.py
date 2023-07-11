import numpy as np
import cv2
import random
import math
from algorithm import algorithm
from techniques import techniques
from SNN import SNN
from network import network
from optimizerMVI import optimizerMVI
from Modifications import Modifications
import time
from mpi4py import MPI
from optimizerDP_III import optimizerDP
from processes_VIII import ROI_loss
from processes_VIII import convolutionalModifications as cM
import csv

class FunctionalNetwork:
    def __init__(self):
        pass
    
    def forward_pass(networkNS, actual, sMaps4):
        cnInput = network.connector_layer(sMaps4)
        fcInput = network.forward_pass(networkNS[1][0], cnInput, networkNS[1][1], 0)
        fcOutput = network.forward_pass(networkNS[1][1], fcInput, [0, 0, 0, 0], 1)
        softmax = network.forward_pass(networkNS[2], fcOutput, [0, 0], 0)
        
        res = fcOutput
        if softmax[0] > softmax[1]:
            res = [0, 0, 0, 0]
        
        err = network.mseINDIV(actual, res)
        return err

    def FC(networkNS, actual, alpha, prev, sMaps4, image):
        print("fp")
        
        # artificial forward propagation
        
        cnInput = network.connector_layer(sMaps4)
        fcInput = network.forward_pass(networkNS[1][0], cnInput, networkNS[1][1], 0)
        fcOutput = network.forward_pass(networkNS[1][1], fcInput, [0, 0, 0, 0], 1)
        softmax = network.forward_pass(networkNS[2], fcOutput, [0, 0], 0)
        
        print("fcb")
        zeros = fcOutput
        
        conf = open("confidence.csv", "a+")
        confwriter = csv.writer(conf)
        confwriter.writerow(softmax)
        conf.close()

        if softmax[0] > softmax[1]:
            zeros = [0, 0, 0, 0]
        else:
            zeros = fcOutput
        # (A8, A16, A32, A64, loss) = cM.convolutionalMaps(image, np.array(np.multiply(zeros, 128), dtype = np.uint8), np.array(np.multiply(actual, 128), dtype = np.uint8))
        err = network.mseINDIV(actual, zeros)   
        loss = sum(err)
        (networkNS[2], _) = network.backprop(networkNS[2], err, 0, alpha, softmax)
        rho = network.direction(prev, loss)
            
        (networkNS[1][1], P6) = network.backprop(networkNS[1][1], err, 1, alpha * rho, fcOutput)
        (networkNS[1][0], P5) = network.backprop(networkNS[1][0], P6, 0, alpha * rho, fcInput)
        filter_loss = P5
        
        print("fb")
        filter_matrix = []
        
        for k in range(8):
            filter_row = []
            for j in range(8):
                place = (8 * k) + j
                val = filter_loss[place]
                filter_row.append(val)
            filter_matrix.append(filter_row)
        
        reverseMatrix = optimizerDP.reverseDense(networkNS[1], fcOutput)
            
        return (networkNS, loss, filter_matrix, rho, reverseMatrix, zeros)
        
    def BP(networkNS, loss, alpha, filter_matrix, sMaps4, sMaps3, sMaps2, sMaps1, rank, rho, reverseMatrix):
        print("4")
        sect = rank // 16
        mod = rank % 16
        
        f4 = [networkNS[0][3][mod * 4], networkNS[0][3][(mod * 4) + 1], networkNS[0][3][(mod * 4) + 2], networkNS[0][3][(mod * 4) + 3]]
        f3 = [networkNS[0][2][mod * 2], networkNS[0][2][(mod * 2) + 1]]
        (r4, r3, r2, r1) = optimizerDP.reverseConvolutional(networkNS[0][0][mod], networkNS[0][1][mod], f3, f4, reverseMatrix)
        
        for j in range(4):
            place = (4 * mod) + j
            conv = algorithm.anticonvolution(filter_matrix, np.array(sMaps4[place]), 1)
            delta = network.multiply(conv, alpha * rho)
            zeta = optimizerDP.directional_partials(delta, sMaps4[place], r4[j])
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
            zeta = optimizerDP.directional_partials(tau, sMaps3[place], r3[j])
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
        zeta = optimizerDP.directional_partials(tau, sMaps2[mod], r2)
        networkNS[0][1][mod] = np.subtract(networkNS[0][1][mod], zeta)
        
        fMap = network.anti_pool(np.array(dMap), 64, 64)
        eMap = algorithm.convolution(networkNS[0][1][mod], fMap, 2)
            
        tau = optimizerMVI.optimize(eMap, sMaps1[mod], network.anti_pool(sMaps2[mod], 64, 64), networkNS[0][0][mod], alpha * rho)
        zeta = optimizerDP.directional_partials(tau, sMaps1[mod], r1)
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
