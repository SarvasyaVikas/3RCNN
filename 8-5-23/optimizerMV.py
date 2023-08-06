import cv2
import numpy as np
from algorithm import algorithm
from network import network
from MPImodifiers import MPImodifiers

class optimizerMV:
    def __init__(self):
        pass
    
    def miniconvolve(F, image):
        if isinstance(image, list):
            image = np.array(image)
        pad_val = len(F) // 2
        (h, w) = image.shape[:2]
        image = np.array(algorithm.same_padding(image, pad_val))
        
        fMap = []
        vector = []
        for i in range(len(F)): # turns the filter into a vector
            for j in range(len(F[i])):
                vector.append(F[i][j])
        for i in range(h):
            fRow = []
            for j in range(w):
                area = []
                for k in range(len(F)):
                    for l in range(len(F[0])):
                        area.append(image[i + k, j + l])
                        
                tot = 0
                for m in range(len(vector)): # calculates convolution value
                    val = vector[m] * area[m]
                    tot += val
                
                fRow.append(tot) 
            
            fMap.append(fRow) # creates array feature map
        
        return fMap
    
    def optimize(l, sp, s, f, alpha):
        pad_val = len(f) // 2
        delta = algorithm.anticonvolution(l, s, pad_val)
        ln = algorithm.convolution(delta, sp, pad_val)
        ln = network.signed_ln(ln)
        cs = np.dot(l, ln)
        try:
            ce = np.multiply(l, ln)
        except:
            pass
        (h, w) = ce.shape[:2]
        for i in range(h):
            for j in range(w):
                if ce[i,j] == 0:
                    ce[i, j] = 1
        theta = np.divide(cs, ce)
        normalized = np.subtract(theta, 0.5)
        sigma = optimizerMV.miniconvolve(normalized, delta)
        tau = np.multiply(sigma, alpha)
        zeta = np.subtract(f, tau)
        return zeta
    
    def optimizeNESTEROV(l, sp, s, f, alpha, fPREV, psi):
        if isinstance(f, list):
            f = np.array(f)
        if isinstance(fPREV, list):
            fPREV = np.array(fPREV)
        fCH = np.subtract(fPREV, f)
        pad_val = len(f) // 2
        delta = algorithm.anticonvolution(l, s, pad_val)
        ln = algorithm.convolution(delta, sp, pad_val)
        ln = network.signed_ln(ln)
        cs = np.dot(l, ln)
        try:
            ce = np.multiply(l, ln)
        except:
            pass
        if ce == 0:
            ce = 1
        theta = np.divide(cs, ce)
        normalized = np.subtract(theta, 0.5)
        sigma = optimizerMV.miniconvolve(normalized, delta)
        tau = np.multiply(sigma, alpha)
        zeta = np.subtract(f, MPImodifiers.nesterovMomentum(fCH, f, tau))
        return zeta
