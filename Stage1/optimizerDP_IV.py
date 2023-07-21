from algorithm import algorithm
import numpy as np
from network_FC_CF_II import network
from processes_VIII import convolutionalModifications as cM

class optimizerDP:
    def __init__(self):
        pass
    
    def directional_partials(fdelta, cur, fut, fpres, iterations = 16):
        if not isinstance(fdelta, list):
            fdelta = fdelta.tolist()
        arrs = []
        dets = []
        
        for i in range(iterations):
            f = []
            for j in range(len(fdelta)):
                n = np.linalg.norm(fdelta[j])
                if n == 0:
                    n = 1
                r = np.arccos(np.divide(fdelta[j], n))
                added = (np.pi * i) / iterations
                a = np.add(r, added)
                rot = np.cos(a)
                f.append(rot)
            
            arrs.append(f)
            res = algorithm.convolution(np.add(fpres, f), cur, len(fdelta) // 2)
            diff = np.subtract(fut, res)
            det = np.linalg.det(diff)
            dets.append(abs(det))
        
        posMAX = dets.index(min(dets))
        filt = arrs[posMAX]
        adj = np.add(fpres, filt)
        return adj
            
    def reverseDense(layers, prediction):
        reverseLayer = []
        for i in range(len(layers[-1])):
            val = layers[-1][i][1]
            for j in range(len(layers[-1][i][0])):
                weight = layers[-1][i][0][j] * prediction[j]
                val += weight
            reverseLayer.append(val)
        
        reverseVector = []
        for i in range(len(layers[-2])):
            val = layers[-2][i][1]
            for j in range(len(layers[-2][i][0])):
                weight = layers[-2][i][0][j] * reverseLayer[j]
                val += weight
            reverseVector.append(val)
        
        reverseMatrix = []
        for i in range(8):
            row = []
            for j in range(8):
                place = (8 * i) + j
                row.append(reverseVector[place])
            reverseMatrix.append(row)
        
        return reverseMatrix

    def reverseConvolutional(f1, f2, f3, f4, reverseMatrix):
        r4 = []
        for i in range(4):
            r = algorithm.convolution(f4[i], reverseMatrix, 1)
            r4.append(r)
        
        g3 = [network.anti_pool(np.add(r4[0], r4[1]), 16, 16), network.anti_pool(np.add(r4[2], r4[3]), 16, 16)]
        r3 = []
        for i in range(2):
            r = algorithm.convolution(f3[i], g3[i], 1)
            r3.append(r)
        
        g2 = network.anti_pool(np.add(r3[0], r3[1]), 32, 32)
        r2 = algorithm.convolution(f2, g2, 2)
        
        g1 = network.anti_pool(r2, 64, 64)
        r1 = algorithm.convolution(f1, g1, 2)
        
        return (r4, r3, r2, r1)
