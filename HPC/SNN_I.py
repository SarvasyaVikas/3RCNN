from algorithm import algorithm
from techniques import techniques
import numpy as np
import math

# have to predefine imageset and alpha
class SNN:
	def __init__(self):
		pass
		
	def generate_filters(num): # generates 5x5 filters
		layer1 = []
		for j in range(5):
			section1 = []
			for i in range(num):
				filter1 = np.random.rand(5, 5)
				filter2 = filter1.tolist()
				section1.append(filter2)
			layer1.append(section1)
		return layer1

	def generate_feature_maps(layer, images): # takes in a layer of node filters and images and creates feature maps
		featureMaps1 = []
		if len(images) % len(layer) == 0: # assigns feature maps to a specific set of filters to minimize computational expense
			div = len(images) // len(layer)
			for i in range(0, len(images), div):
				section1 = []
				for j in range(div):
					place = (div * i) + j
					fMap = algorithm.convolution(layer[i], images[place], 2)
					fMapImage = np.array(fMap)
					section1.append(fMapImage)
				featureMaps1.append(section1)
		else:
			for j in range(len(images)):
				section1 = []
				for i in range((layer)):
					fMap = algorithm.convolution(layer[i], images[j], 2)
					fMapImage = np.array(fMap)
					section1.append(fMapImage)
				featureMaps1.append(section1)
		return featureMaps1

	def pooled_maps(maps): # max pooling but on a list of maps
		pooled = []
		for i in range(len(maps)):
			pool = algorithm.max_pooling(maps[i])
			pooled.append(pool)
		return pooled

	def sigmoid(x): # sigmoid activation
		var = math.e ** (-x)
		denom = 1 + var
		frac = 1 / denom
		return frac

	def add(matrices): # basic add matrix function for ease of access
                for i in range(len(matrices)):
                    if isinstance(matrices[i], list):
                            matrices[i] = np.array(matrices[i])
                tot = []
                (h, w) = matrices[0].shape[:2]
		for i in range(h):
			row = []
			for j in range(w):
				count = 0
				for k in range(len(matrices)):
					count += matrices[k][i,j]
				row.append(count)
			tot.append(row)
		return tot

	def multiply(matrix, factor): # basic multiply matrix function for ease of access
		for i in range(len(matrix)):
			for j in range(len(matrix[0])):
				matrix[i][j] = matrix[i][j] * factor
		return matrix

	def states(images, featureMaps): # applies events with states
		# feeds in one (1) set of images
		# meaning FIVE images
		# and then feature maps
		sAB = []
		sAC = []
		sAD = []
		sAE = []
		sBC = []
		sBD = []
		sBE = []
		sCD = []
		sCE = []
		sDE = []
		for i in range(1):
			sAB.append(techniques.s(images[0], images[1]))
			sAC.append(techniques.s(images[0], images[2]))
			sAD.append(techniques.s(images[0], images[3]))
			sAE.append(techniques.s(images[0], images[4]))
			sBC.append(techniques.s(images[1], images[2]))
			sBD.append(techniques.s(images[1], images[3]))
			sBE.append(techniques.s(images[1], images[4]))
			sCD.append(techniques.s(images[2], images[3]))
			sCE.append(techniques.s(images[2], images[4]))
			sDE.append(techniques.s(images[3], images[4]))
		
		forwardPass = [[], [], [], [], []]
		
		for i in range(len(featureMaps[0])): # recreates forward pass w/ coefficients
		# have to divide by sum of coefficients at the end
			j = 0
		
			BsAB = np.multiply(featureMaps[1][i], sAB[j])
			CsAC = np.multiply(featureMaps[2][i], sAC[j])
			DsAD = np.multiply(featureMaps[3][i], sAD[j])
			EsAE = np.multiply(featureMaps[4][i], sAE[j])
			
			totA = SNN.add([featureMaps[0][i], BsAB, CsAC, DsAD, EsAE])
			sumA = 1 + sAB[j] + sAC[j] + sAD[j] + sAE[j]
			denomA = 1.0 / sumA
			finA = np.multiply(totA, denomA)
			forwardPass[0].append(finA)
			
			AsAB = np.multiply(featureMaps[0][i], sAB[j])
			CsBC = np.multiply(featureMaps[2][i], sBC[j])
			DsBD = np.multiply(featureMaps[3][i], sBD[j])
			EsBE = np.multiply(featureMaps[4][i], sBE[j])
			
			totB = SNN.add([AsAB, featureMaps[1][i], CsBC, DsBD, EsBE])
			sumB = sAB[j] + 1 + sBC[j] + sBD[j] + sBE[j]
			denomB = 1.0 / sumA
			finB = np.multiply(totB, denomB)
			forwardPass[1].append(finB)
			
			AsAC = np.multiply(featureMaps[0][i], sAC[j])
			BsBC = np.multiply(featureMaps[1][i], sBC[j])
			DsCD = np.multiply(featureMaps[3][i], sCD[j])
			EsCE = np.multiply(featureMaps[4][i], sCE[j])
			
			totC = SNN.add([AsAC, BsBC, featureMaps[2][i], DsCD, EsCE])
			sumC = sAC[j] + sBC[j] + 1 + sCD[j] + sCE[j]
			denomC = 1.0 / sumC
			finC = np.multiply(totC, denomC)
			forwardPass[2].append(finC)
			
			AsAD = np.multiply(featureMaps[0][i], sAD[j])
			BsBD = np.multiply(featureMaps[1][i], sBD[j])
			CsCD = np.multiply(featureMaps[2][i], sCD[j])
			EsDE = np.multiply(featureMaps[4][i], sDE[j])
			
			totD = SNN.add([AsAD, BsBD, CsCD, featureMaps[3][i], EsDE])
			sumD = sAD[j] + sBD[j] + sCD[j] + 1 + sDE[j]
			denomD = 1.0 / sumD
			finD = np.multiply(totD, denomD)
			forwardPass[3].append(finD)
			
			AsAE = np.multiply(featureMaps[0][i], sAE[j])
			BsBE = np.multiply(featureMaps[1][i], sBE[j])
			CsCE = np.multiply(featureMaps[2][i], sCE[j])
			DsDE = np.multiply(featureMaps[3][i], sDE[j])
			
			totE = SNN.add([AsAE, BsBE, CsCE, DsDE, featureMaps[4][i]])
			sumE = sAE[j] + sBE[j] + sCE[j] + sDE[j] + 1
			denomE = 1.0 / sumE
			finE = np.multiply(totE, denomE)
			forwardPass[4].append(finE)
		
		return forwardPass
"""
sections = [] # sections is the tensor of filters

for i in range(5):
	layer1 = generate_filters(16)
	layer2 = generate_filters(16)
	layer3 = generate_filters(32)
	layer4 = generate_filters(32)
	layers = [layer1, layer2, layer3, layer4]
	sections.append(layers)

fMaps = [[], [], [], []] # tensor of feature maps
rMaps = [] # tensor of resultant maps (not really necessary, just there for storage)

dense = [0, 0, 0, 0]

for i in range(len(imageset)):

	# applying forward pass
	for j in range(len(imageset[0])):

		featureMaps1 = generate_feature_maps(sections[j][0], imageset[i])
		pooledMaps1 = pooled_maps(featureMaps1)
		fMaps[0].append(pooledMaps1)
	
	resultMaps1 = states(imageset[i], fMaps[0])
	rMaps.append(resultMaps1)
	
	for j in range(len(imageset[0])):	
		featureMaps2 = generate_feature_maps(sections[j][1], resultMaps1)
		pooledMaps2 = pooled_maps(featureMaps2)
		fMaps[1].append(pooledMaps2)
	
	resultMaps2 = states(imageset[i], fMaps[1])
	rMaps.append(resultMaps2)
	
	for j in range(len(imageset[0])):
		featureMaps3 = generate_feature_maps(sections[j][2], resultMaps2)
		pooledMaps3 = pooled_maps(featureMaps3)
		fMaps[2].append(pooledMaps3)
	
	resultMaps3 = states(imageset[i], fMaps[2])
	rMaps.append(resultMaps3)	
		
	for j in range(len(imageset[0])):
		featureMaps4 = generate_feature_maps(sections[j][3], resultMaps3)
		pooledMaps4 = pooled_maps(featureMaps4)
		fMaps[3].append(pooledMaps4)
	
	resultMaps4 = states(imageset[i], fMaps[3])
	rMaps.append(resultMaps4)
	
	# finishes feature maps/filter operations
	
	h = len(resultMaps4[0])
	w = len(resultMaps4[0][0])
	
	# calculates bounding box
	
	features = resultMaps4[2].copy()
	horiz = []
	vert = []
	
	for j in range(h):
		for k in range(w):
			val = 0
			for l in range(5):
				added = ((-1) ** (l)) * resultMaps4[l][j][k]
				val += added
			if val > 255:
				features[j][k] = 155
				if k not in horiz:
					horiz.append(k)
				if j not in vert:
					vert.append(j)
			elif val < 100:
				features[j][k] = 0
			else:
				features[j][k] = val - 100
				if k not in horiz:
					horiz.append(k)
				if j not in vert:
					vert.append(j)
	
	xi = min(horiz)
	xf = max(horiz)
	yi = min(vert)
	yf = max(vert)
	
	# calculates loss
	
	loss_matrix = algorithm.MSE(features, Axi, Axf, Ayi, Ayf) # represents actual values
	
	loss = (loss_matrix[yi][xi] + loss_matrix[yi][xf] + loss_matrix[yf][xi] + loss_matrix[yf][xf]) / 4.0
	
	loss_matrix = multiply(loss_matrix, loss) # incorporates actual loss
	
	(dLdx, dLdy) = algorithm.partials(loss_matrix)
	
	# changes filters
	
	for l in range(len(sections)):
		for k in range(len(sections[0])):
			rMaps = []
			layer_vals = []
			adj1 = 0
			for j in range(len(sections[0][k])): # for overall filter adjustment
				padded = algorithm.same_padding(imageset[i][l], 2)
				convolution = algorithm.convolution(sections[l][k][j], padded, 2)
				(dOdx, dOdy) = algorithm.partials(convolution)
				adj1 = algorithm.overallFilterAdjustment(dLdx, dLdy, dOdx, dOdy, imageset[i][l], sections[l][k][j], alpha)
				
				maxPooled = algorithm.max_pooling(convolution)
				avgPooled = algorithm.avg_pooling(maxPooled, sections[l][k][j])
				rMaps.append(avgPooled)
				f = algorithm.flatten(avgPooled)
				for i in range(len(f)):
					layer_vals.append(f[i])
			
			layer_probs = algorithm.softmax(layer_vals)
			adj2 = 0
			
			for j in range(len(sections[0][k])): # for individual filter adjustment
				sm_prob = layer_probs[(5 * j):(5 * (j + 1))]
				adj2 = algorithm.indivFilterAdjustment(rMaps[j], sm_prob, sections[l][k][j], alpha)
				for m in range(5):
					for n in range(5):
						diff = alpha * (adj1[m][n] + adj2[m][n])
						sections[l][k][j][m][n] = sections[l][k][j][m][n] - diff
						
	for j in range(sections[0]): # applies plane recurrence
		for k in range(sections[0][j]):
			a = sections[0][j][k]
			b = sections[1][j][k]
			c = sections[2][j][k]
			d = sections[3][j][k]
			e = sections[4][j][k]
			
			matrices = techniques(a, b, c, d, e).plane_recurrence()
			
			for l in range(5):
				sections[l][j][k] = matrices[l]
"""
