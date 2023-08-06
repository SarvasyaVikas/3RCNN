import cv2
import numpy as np
import math

class techniques:
    def __init__(self, A, B, C, D, E):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
    
    def elu(x):
        if x <= 0:
            return ((math.e ** x) - 1)
        else:
            return x
    
    def s(img1, img2): # similarity coefficients
        (h, w) = img1.shape[:2]
        scTOT = 0
        avg = (sum(img1, img2) / 2.0) + 0.5
        avgLN = np.log(avg)
                
        characteristic1 = avgLN / np.log(255)
            
        diff = abs(img1 - img2)
        characteristic2 = np.sqrt(diff / 256)
            
        val1 = characteristic1 - characteristic2
        val2 = val1.tolist()
        for i in range(h):
            for j in range(w):
                val3 = techniques.elu(val2[i][j])
                scTOT += val3
        
        sc = float(scTOT / h / w)
        return sc
    
    def correction(self, image, section): # applies image correction
        # assumes that correctionDNN has already been applied and does selective dimming
        (h, w) = image.shape[:2]
        min_val = 256
        for i in range(h):
            for j in range(w):
                if image[i][j] > 50 and image[i][j] < min_val:
                    min_val = image[i][j]
        
        for i in range(h):
            for j in range(w):
                if image[i][j] < min_val:
                    image[i][j] = 0
                else:
                    image[i][j] = image[i][j] - min_val
        
        max_val = 0
        (h1, w1) = section.shape[:2]
        for i in range(h1):
            for j in range(w1):
                if section[i][j] > max_val:
                    max_val = section[i][j]
        
        denom = max_val - 25 - min_val
        factor = 255 / float(denom)
        
        for i in range(h):
            for j in range(w):
                image[i][j] = image[i][j] * factor
        
        kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        
        matrix = algorithm.convolution(kernel, image, 1)
        arr = np.array(matrix)
        borders = cv2.dilate(arr, None, 2)
        
        pixelSUM = 0
        pixelCOUNT = 0
        
        for i in range(h):
            for j in range(w):
                if borders[i][j] > image[i][j]:
                    image[i][j] = 0
                else:
                    image[i][j] = image[i][j] - borders[i][j]
                if image[i][j] > 215:
                    pixelSUM += image[i][j]
                    pixelCOUNT += 1
        
        pixelAVG = pixelSUM / pixelCOUNT
        denom1 = pixelAVG - 170
        factor1 = 70 / denom1
        
        for i in range(h):
            for j in range(w):
                sub = image[i][j] - 170
                new_sub = sub * factor1
                new = new_sub + 170
                
                if new < 0:
                    image[i][j] = 0
                else:
                    image[i][j] = new
        
        return image

    def antizero(matrix):
        if isinstance(matrix, list):
            matrix = np.array(matrix)
        (h, w) = matrix.shape[:2]
        for i in range(h):
            for j in range(w):
                if matrix[i,j] == 0:
                    matrix[i,j] = 1
        return matrix

    def plane_recurrence(self): # applies plane recurrence between the filters of a specific node layer
        detA = 1
        detB = 1
        detC = 1
        detD = 1
        detE = 1
        
        try:
            logA = np.log(techniques.antizero(abs(self.A)))
        except:
            logA = abs(self.A)
        try:
            logB = np.log(techniques.antizero(abs(self.B)))
        except:
            logB = abs(self.B)
        try:
            logC = np.log(techniques.antizero(abs(self.C)))
        except:
            logC = abs(self.C)
        try:
            logD = np.log(techniques.antizero(abs(self.D)))
        except:
            logD = abs(self.D)
        try:
            logE = np.log(techniques.antizero(abs(self.E)))
        except:
            logE = abs(self.E)

        matrices = [logA, logB, logC, logD, logE]
        
        dets = [detA, detB, detC, detD, detE]
        for i in range(5):
            try:
                det = np.linalg.det(matrices[i])
                dets[i] = det
            except:
                pass

        for i in range(5):
            if dets[i] == 0:
                dets[i] = 1
        factors = []
        for l in range(5):
            factor = 0
            for k in range(5):
                addend = ((-0.5) ** abs(l-k)) * dets[k]
                factor += addend
            factors.append(factor)
        
        h = len(self.A)
        w = len(self.A[0])
        
        for i in range(h):
            for j in range(w):
                for k in range(5):
                    matrices[k][i][j] = matrices[k][i][j] * factors[k] / dets[k]
        
        return matrices
