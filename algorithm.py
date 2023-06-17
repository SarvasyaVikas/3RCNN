import cv2
import numpy as np
import math

class algorithm:
	def __init__(self):
		pass
	
	def same_padding(image, pad): # same_padding amount of pad
		arr = []
		(h, w) = image.shape[:2]
		for i in range(h):
			lst = []
			for ext in range(pad): # beginning of list
				lst.append(image[i][0])
			for j in range(w): # end of list
				lst.append(image[i][j])
			for ext in range(pad):
				lst.append(image[i][-1])
			if i == 0 or i == h - 1: # top or bottom of array
				for ext in range(pad):
					arr.append(lst)
			
			arr.append(lst)
		
		return arr
		
	def convolution(F, image, pad): # applying convolutional kernel
		if isinstance(image, list):
			image = np.array(image)
		(h, w) = image.shape[:2]
		image = np.array(algorithm.same_padding(image, pad))
		fMap = []
		vector = []
		for i in range(len(F)): # turns the filter into a vector
			for j in range(len(F[i])):
				vector.append(F[i][j]) 
		for i in range(h):
			fRow = []
			for j in range(w):
				area = [] # this is the vector for the kernel area
				for k in range(len(F)):
					for l in range(len(F[0])):
						y = i + k
						x = j + l
						area.append(image[y,x])
						
				tot = 0
				for m in range(len(vector)): # calculates convolution value
					val = algorithm.signed_log(vector[m]) * algorithm.signed_log(area[m])
					tot += val
				
				fRow.append(tot) 
			
			fMap.append(fRow) # creates array feature map

		return fMap
	
	def signed_log(x):
		sign = -1 if x < 0 else 1
		if x == 0:
			x = 1
		log = np.log(abs(x))
		res = sign * log
		return res
	
	def anticonvolution(F, image, pad): # applying convolutional kernel
		if isinstance(image, list):
			image = np.array(image)
		(h, w) = image.shape[:2]
		image = np.array(algorithm.same_padding(image, pad))
		fMap = []
		vector = []
		for i in range(len(F)): # turns the filter into a vector
			for j in range(len(F[i])):
				vector.append(F[i][j]) 
		for i in range((2 * pad) + 1):
			fRow = []
			for j in range((2 * pad) + 1):
				area = []
				for k in range(h):
					for l in range(w):
						y = i + k
						x = j + l
						area.append(image[y,x])
				
				tot = 0
				for m in range(len(vector)): # calculates convolution value
					val = algorithm.signed_log(vector[m]) * algorithm.signed_log(area[m])
					tot += val
				
				fRow.append(tot) 
			
			fMap.append(fRow) # creates array feature map

		return fMap
	
	def max_pooling(fMap): # max pooling (by 2)
		if isinstance(fMap, list):
			fMap = np.array(fMap)
		pooledMap = []
		(h, w) = fMap.shape[:2]
		for i in range(0, h - 1, 2):
			pooledRow = []
			for j in range(0, w - 1, 2):
				val1 = fMap[i,j]
				val2 = fMap[(i + 1),(j)]
				val3 = fMap[(i),(j + 1)]
				val4 = fMap[(i + 1),(j + 1)]
				val = max(val1, val2, val3, val4)
				pooledRow.append(val)
			
			pooledMap.append(pooledRow)
		pooledImage = np.array(pooledMap)
		return pooledImage
	
	def avg_pooling(self, pooledMap, F): # average pooling (to filter, for individual filter adjustment)
		# dimensional variables
		hF = len(F)
		wF = len(F[0])
		hP = len(pooledMap)
		wP = len(pooledMap[0])
		
		# boundary variables
		hMod = hP % hF
		hExt = hF - hMod
		wMod = wP % wF
		wExt = wF - wMod
		
		newMap = [] # adjust pooled map to divide dimensions
		
		for i in range(hP):
			wRow = pooledMap[i]
			for j in range(wExt):
				wRow.append(wRow[-1])
			newMap.append(wRow)

		for i in range(hExt):
			newMap.append(newMap[-1])
		
		hD = hP // hF + 1
		wD = wP // wF + 1
		
		rMap = [] # resultant map
		
		for i in range(hF):
			rRow = []
			for j in range(wF):
				for k in range(hD * i, hD * (i + 1)):
					for l in range(wD * i, wD * (i + 1)):
						tot += newMap[k][l]
				
				avg = float(tot) / float(hD) / float(wD)
				rRow.append(avg)
			
			rMap.append(rRow)
		
		return rMap
	
	def flatten(self, matrix): # flatten matrix into a vector
		flattened = []
		for i in range(len(matrix)):
			for j in range(len(matrix[0])):
				flattened.append(matrix[i][j])
		
		return flattened
	
	def softmax(self, vector): # apply softmax to a vector to get probabilities
		exp_sum = 0
		sm = []
		for i in range(len(vector)):
			exp_val = math.e ** (vector[i])
			exp_sum += exp_val
			sm.append(exp_val)
		
		sm_prob = np.divide(sm, exp_sum)
		
		# generic softmax function
		# does not subtract to normalize to 0	
			
		return sm_prob
	
	def indivFilterAdjustment(self, rMap, sm_prob, F, alpha): # applies individual filter adjustment
		hF = len(F)
		wF = len(F[0])
		normalizeFactor = (hF * wF) ** (-1)
		sm_normalized = [prob - normalizeFactor for prob in sm_prob]
		
		rVector = []
		for i in range(rMap):
			for j in range(rMap[0]):
				rVector.append(rMap[i][j])
		
		delta = []
		for i in range(rVector):
			element = rVector[i] * sm_normalized[i]
			delta.append(element)
		
		f = []
		for i in range(hF):
			for j in range(wF):
				f.append(F[i][j])
		
		fAdjusted = []
		for i in range(hF * wF):
			adjustment = f[i] - (alpha * delta[i])
			fAdjusted.append(adjustment)
		
		newF = []
		for i in range(0, hF):
			rowF = fAdjusted[(wF * i):(wF * (i + 1))]
			newF.append(rowF)
		
		return newF
	
	def MSE(self, image, Xi, Xf, Yi, Yf):
		arr = []
		(h, w) = image.shape[:2]
		for i in range(h):
			row = []
			for j in range(w):
				minX = min((j - Xi) ** 2, (j - Xf) ** 2)
				minY = min((i - Yi) ** 2, (i - Yf) ** 2)
				error = minX + minY
				row.append(error)
			arr.append(row)
		
		return arr
	
	def partials(self, matrix):
		dx = []
		dy = []
		
		h = len(matrix)
		w = len(matrix[0])
		
		yPadded = []
		
		for j in range(w):
			yPadded.append(0)
		
		dy.append(yPadded)
		
		for i in range(h):
			dxRow = []
			dyRow = []
			for j in range(w):
				if j == 0:
					dxRow.append(0)
				else:
					dxVal = matrix[i][j] - matrix[i][j - 1]
					dxRow.append(dxVal)
				if i > 0:
					dyVal = matrix[i + 1][j] - matrix[i][j]
					dyRow.append(dyVal)
			
			dx.append(dxRow)
			dy.append(dyRow)
		
		return (dx, dy)
	
	def dot_product(self, matrix1, matrix2):
		h1 = len(matrix1)
		w1 = len(matrix1[0])
		h2 = len(matrix2)
		w2 = len(matrix2[0])
		if w1 != h2:
			return False
		
		dotted = []
		for i in range(h1):
			dottedRow = []
			for j in range(w2):
				v1 = matrix1[i]
				v2 = []
				for k in range(h2):
					v2.append(matrix2[k][j])
				
				val = 0
				
				for l in range(len(v1)):
					added = v1[l] * v2[l]
					val += added
				
				dottedRow.append(val)
			
			dotted.append(dottedRow)
		
		return dotted
	
	def overallFilterAdjustment(self, dLdx, dLdy, dOdx, dOdy, image, F, alpha):
		dLdO = []
		
		h = len(dLdx)
		w = len(dLdx[0])
		
		hF = len(F)
		wF = len(F[0])
		Fpart = hF // 2
		
		for i in range(h):
			dRow = []
			for j in range(w):
				val1 = float(dLdx[i][j]) / float(dOdx[i][j])
				val2 = float(dLdy[i][j]) / float(dOdy[i][j])
				val3 = val1 + val2
				dRow.append(val3)
			
			dLdO.append(dRow)
		
		padded = same_padding(image, Fpart)
		dOdF = []
		for i in range(h):
			dORow = []
			rows = padded[i:(i + hF)]
			for j in range(w):
				area = []
				for row in rows:
					col = row[j:(j + wF)]
					area.append(col)
				
				dotO = dot_product(area, F)
				tot = 0
				
				for k in range(dotO):
					for l in range(dotO[0]):
						tot += dotO[k][l]
				
				dORow.append(tot)
			
			dOdF.append(dORow)
		
		dLdF = dot_product(dLdO, dOdF)
		
		delF = avg_pooling(dLdF, F)
		Fadjusted = []
		for i in range(len(F)):
			adj_row = []
			for j in range(len(F[0])):
				adj = F[i][j] - (alpha * delF[i][j])
				adj_row.append(adj)
			
			Fadjusted.append(adj_row)
		
		return Fadjusted
