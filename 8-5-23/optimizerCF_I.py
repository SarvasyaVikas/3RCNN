from algorithm import algorithm
import numpy as np
from network_FC_CF_II import network

class optimizerCF:
    def __init__(self):
        pass
    
    def directional_partials(fdelta, cur, fut, fpres, iterations = 16):
        if not isinstance(fdelta, list):
            fdelta = fdelta.tolist()
        arrs = []
        dets = []
        
        # cur is the initial layer input
        # fut is the compressed output from reverseCF
        # fdelta is the list of weight modifications
        # 64 * 16 of them for the first layer, so this is a vector of 64 lists, each with 16 elements
        # 16 * 4 for the second layer
        # 4 * 2 for the third layer

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
                new = np.multiply(rot, n)
                f.append(new)
            
            arrs.append(f)
            fnew = 
            res = network.forward_pass(, cur, fut)
            diff = np.subtract(fut, res)
            det = np.linalg.det(diff)
            dets.append(abs(det))
        
        posMAX = dets.index(min(dets))
        filt = arrs[posMAX]
        return filt

    def sigmoidINVERSE(val):
        x = abs(val)
        if x == 0 or x == 1:
            x = 0.5

        recip = 1.0 / x
        insideLN = abs(recip - 1)
        negY = np.log(insideLN)
        y = -1 * negY
        return y

    def tanhINVERSE(val):
        x = val
        if x =< -1 or x >= 1:
            x = 0

        numer = 1.0 - x
        denom = 1.0 + x
        frac = float(numer / denom)
        adjY = np.log(frac)
        y = -0.5 * adjY
        return y

    def reverseCF(layers, pred_soft):
        # pred_soft should be in terms of the initial softmax output
        # layers = networkNS

        reverseLayer4 = [] # for layers[2] aka coordinates layer
        for i in range(4):
            val = layers[2][i][1]
            for j in range(2):
                weight = layers[2][i][0][j] * pred_soft[j]
                val += weight
            activ = optimizerCF.sigmoidINVERSE(val / 10.0)
            reverseLayer4.append(activ)

        reverseLayer16 = []
        for i in range(16):
            val = layers[1][i][1]
            for j in range(4):
                weight = layers[1][i][0][j] * reverseLayer4[j]
                val += weight
            activ = optimizerCF.tanhINVERSE(val / 10.0)
            reverseLayer16.append(activ)

        reverseLayer64 = []
        for i in range(64):
            val = layers[0][i][1]
            for j in range(16):
                weight = layers[0][i][0][j] * reverseLayer16[j]
                val += weight
            activ = optimizerCF.tanhINVERSE(val / 10.0)
            reverseLayer64.append(activ)

        layer0 = []
        layer1 = []
        layer2 = []
        for i in range(16):
            placeA = i * 4
            placeB = placeA + 1
            placeC = placeB + 1
            placeD = placeC + 1
            layer0val = (reverseLayer64[placeA] + reverseLayer64[placeB] + reverseLayer64[placeC] + reverseLayer64[placeD]) / 4.0
            layer0.append(layer0val)

        for i in range(4):
            placeA = i * 4
            placeB = placeA + 1
            placeC = placeB + 1
            placeD = placeC + 1
            layer1val = (reverseLayer16[placeA] + reverseLayer16[placeB] + reverseLayer16[placeC] + reverseLayer16[placeD]) / 4.0
            layer1.append(layer1val)

        for i in range(2):
            placeA = i * 2
            placeB = placeA + 1
            layer2val = (reverseLayer4[placeA] + reverseLayer4[placeB]) / 2.0
            layer2.append(layer2val)

        return (layer0, layer1, layer2)
