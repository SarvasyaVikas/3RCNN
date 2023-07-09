import cv2
import numpy as np
import csv
from algorithm import algorithm

class process:
	def __init__(self):
		pass
	
	def sharpen(image):
		kernel = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
		sharpened = cv2.filter2D(image, -1, kernel)
		return sharpened
	
	def smooth(image):
		kernel = [[0, 1, 0], [1, 4, 1], [0, 1, 0]]
		smoothed = cv2.filter2D(image, -1, kernel)
		return smoothed
	
	def erosion_dilation(image, itera):
		kernel = np.ones((6, 6), np.uint8)
		image = cv2.erode(image, kernel, cv2.BORDER_REFLECT, iterations = itera)
		image = cv2.dilate(image, kernel, cv2.BORDER_REFLECT, iterations = itera)
		return image
	
	def edges(image):
		kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
		edged = cv2.filter2D(image, -1, kernel)
		return edged
	
	def intersection_over_union(image2, bb1):
		length = abs(bb1[2] - bb1[0])
		width = abs(bb1[3] - bb1[1])
		LX = min(bb1[0], bb1[2]) - length
		RX = max(bb1[0], bb1[2]) + length
		TY = min(bb1[1], bb1[3]) - width
		BY = max(bb1[1], bb1[3]) + width
		proposal = image2[LX:RX, TY:BY]
		
		return proposal
	
	def thresholding(image, val):
		blurred = cv2.GaussianBlur(image, (7, 7), 0)
		(_, thresh) = cv2.threshold(blurred, val, 255, cv2.THRESH_BINARY)
		return thresh
	
	def cut(image, hVAL = 4, wVAL = 4):
		cuts = []
		for i in range(hVAL):
			for j in range(wVAL):
				hS = i * (image // hVAL)
				hE = (i + 1) * (image // hVAL)
				wS = i * (image // wVAL)
				wE = (i + 1) * (image // wVAL)
				spliced = image[hS:hE, wS:wE]
				cuts.append(spliced)
		
		return cuts
	
	def calcium_scores(val, cutoff):
		a = 166.3006356
		b = 0.4202905915
		
		div = val / a
		log = np.log(div)
		lnx = log / b
		dens = np.exp(lnx)
	
		div1 = cutoff / a
		log1 = np.log(div1)
		lnx1 = log1 / b
		dens1 = np.exp(lnx1)
		
		multiplier = dens / dens1
		return multiplier
	
	def scoring(val):
		factor = 0
		if val < 130:
			factor = 0
		elif val < 200:
			factor = 1
		elif val < 300:
			factor = 2
		elif val < 400:
			factor = 3
		else:
			factor = 4
		return factor
	
	def area(fov):
		ratio = fov / 512.0
		pixAREA = ratio ** 2
		return pixAREA
	
	def cut_scoring(cut, minv, fov):
		(h, w) = cut.shape[:2]
		raw = 0
		for i in range(h):
			for j in range(w):
				val = cut[i,j] - (minv + 1000)
				factor = process.scoring(val)
				raw += factor
		pixAREA = process.area(fov)
		scaled = raw * pixAREA
		return scaled
	
	def minmax(init_file):
		ds = dicom.read_file(init_file)
		minval = 10000
		maxval = -10000
		for row in ds.pixel_array:
			for col in row:
				if col < minval:
					minval = col
				if col > maxval:
					maxval = col
		
		return (minval, maxval)

	def minmaxscan(scan):
		minval = 10000
		maxval = -10000
		files = fnmatch.filter(os.listdir("NGCT{}".format(scan)), '*')
		for i in range(len(files)):
			path = "NGCT{}/{}".format(scan, files[i])
			try:
				(minv, maxv) = process.minmax(path)
			except:
				minv = minval
				maxv = maxval
			if minv < minval:
				minval = minv
			if maxv > maxval:
				maxval = maxv
		
		return (minval, maxval)
	
	def intensity_rating(cuts, minv, fov):
		scores = []
		for i in range(len(cuts)):
			score = process.cut_scoring(cuts[i], minv, fov)
			scores.append(score)
		return scores
	
class ROI_loss:
	def __init__(self):
		pass
	
	def within(h, w, vals):
		artery = []
		for i in range(h):
			for j in range(w):
				horiz = []
				vert = []
				for k in range(len(vals)):
					if vals[k][0] == i:
						horiz.append(vals[k][1])
					if vals[k][1] == j:
						vert.append(vals[k][0])
				
				try:
					minH = min(horiz)
					maxH = max(horiz)
				except:
					minH = 0
					maxH = 0
				try:
					minV = min(vert)
					maxV = max(vert)
				except:
					minV = 0
					maxV = 0
				
				if (minH < j) and (maxH > j) and (minV < i) and (maxV > i):
					artery.append([i, j])
		
		return artery
	
	def ROI2ART(bounding_box):
		bounding_box = cv2.GaussianBlur(bounding_box, (7, 7), 0)
		edges = process.edges(bounding_box)
		vals = []
		(h, w) = edges.shape[:2]
		for i in range(h):
			for j in range(w):
				if edges[i,j] > 100:
					vals.append([i, j])
		
		artery = ROI_loss.within(h, w, vals)
		
		return artery
	
	def bounds(bounding_box):
		(h, w) = bounding_box.shape[:2]
		tot = 0
		count = 0
		for i in range(h):
			for j in range(w):
				if (bounding_box[i, j] > 50) and (bounding_box[i, j] < 125):
					tot += bounding_box[i, j]
					count += 1
		avg = tot / count
		return avg

	def boundsCALCIUM(bounding_box):
		avg = ROI_loss.bounds(bounding_box)
		minVAL = 256
		(h, w) = bounding_box.shape[:2]
		for i in range(h):
			for j in range(w):
				if (bounding_box[i, j] > max(100, avg + 20)):
					if (bounding_box[i, j] < minVAL):
						minVAL = bounding_box[i, j]
		return minVAL
				
	def ROI2ART_Advanced(bounding_box):
		# bounding_box = cv2.GaussianBlur(bounding_box, (7, 7), 0)
		# blurring is antithetical to the close ranges shown below
		orig = bounding_box.copy()
		avg = ROI_loss.bounds(bounding_box)
		(h, w) = bounding_box.shape[:2]
		for i in range(h):
			for j in range(w):
				if (bounding_box[i, j] <= avg - 5):
					bounding_box[i, j] = 0
				else:
					bounding_box[i, j] = avg
		
		# at this point, we have all of the outlying heart regions set to black, while the arteries and greater heart regions are gray
		# calcium has also been normalized to the max heart value, 70
		
		contours = cv2.findContours(bounding_box, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
		center = [h / 2.0, w / 2.0]
		contourART = contours[0]
		distance = (10 ** 6)
		for i in range(len(contours)):
			xVAL = 0
			yVAL = 0
			count = 0
			contourFORM = []
			for j in range(len(contours[i])):
				xVAL += contours[i][j][0][0]
				yVAL += contours[i][j][0][1]
				count += 1
				contourFORM.append([contours[i][j][0][0], contours[i][j][0][1]])
			
			xCOM = xVAL / count
			yCOM = yVAL / count
			COM = [xCOM, yCOM]
			dev = ROI_loss.distanceFormula(center, COM)
			if dev < distance:
				distance = dev
				contourART = contourFORM
		
		artery = ROI_loss.within(h, w, contourART)
		
		return artery
	
	def distanceFormula(coor1, coor2):
		diffX = coor1[0] - coor2[0]
		diffY = coor1[1] - coor2[1]
		dist = np.sqrt((diffX ** 2) + (diffY ** 2))
		return dist
	
	def pixelLoss(predicted_artery, actual_artery):
		prob = min(predicted_artery, actual_artery) / max(predicted_artery, actual_artery)
		tot = 0
		for i in range(len(predicted_artery)):
			val = 10000
			for j in range(len(actual_artery)):
				dist = ROI_loss.distanceFormula(predicted_artery[i], actual_artery[j])
				if dist < val:
					val = dist
			tot += val
		
		loss = tot / prob
		return loss
	
	def pixelLossMatrix(predicted_artery, actual_artery, bounding_box):
		arr = []
		(h, w) = bounding_box.shape[:2]
		max_dist = int(np.sqrt(((h ** 2) + (w ** 2))))
		for i in range(h):
			arr_row = []
			for j in range(w):
				arr_row.append(0)
			arr.append(arr_row)
	
		prob = min(predicted_artery, actual_artery) / max(predicted_artery, actual_artery)
		tot = 0
		for i in range(len(predicted_artery)):
			val = 10000
			for j in range(len(actual_artery)):
				dist = ROI_loss.distanceFormula(predicted_artery[i], actual_artery[j])
				if dist < val:
					val = dist
			if val != 10000:
				arr[predicted_artery[i][0]][predicted_artery[i][1]] = (max_dist - val)
				tot += val
		
		for i in range(len(arr)):
			if sum(arr[i]) == 0:
				arr.remove(arr[i])
		
		for j in range(len(arr[0])):
			tot = 0
			for i in range(len(arr)):
				tot += arr[i][j]
			if tot == 0:
				for i in range(len(arr)):
					arr[i].remove(arr[i][j])
		
		loss = tot / prob
		return (arr, prob, loss)
		
		# padded with zeros
		# expanding array to image with same size
		# for 128x128 and put back
		# putting array as input image
		# outside values set to -1?
		
		# could augment this
		# augment original image with loss array
		# convolving array with image
		# using preprocessing
		
		# AUGMENTATION
		# use image several times, with small modifications
		# lighter/darker/rotating
		# no changes to architecture
		# generalizable
		# one network for all types
		# increases training set
		
		# run Selective Search or RPN?
		# add in distance calculations, so the network learns how far away things are
		# tell model to look in vicinity
		# BETTER RUN TIMES
		# do both for comparison?
		# separate ANN for certain things to learn hyperparameters
		
		# similar to a grid search
		# predefined values (layers/sections?)
		
		# calcium scores should be standardized
		# values range
		# same in ecg
		# try forcible scaling
		
		# compare both
		# draw generated bounding boxes
		# find scan radiation values
		
		# expand values
		
		# AI heavy
		# Circulation
		# American Heart Journals
		# Higher Impact
		# IEEE
		# research more
		# email
		# say Clinician actually looked at the scores and adjucated it
		# generic formatting
		# New England Journal of Medicine AI (3k words)
		# International Journal of Cardiology
		
		# start with 4k words

	def loss_conversion(arr, prob):
		pixelLoss_arr = np.array(arr)
		(h, w) = pixelLoss_arr.shape[:2]
		lossMatrix = 0
		if (h > 8) and (w > 8):
			lossMatrix = cv2.resize(pixelLoss_arr, (8, 8), interpolation = cv2.INTER_AREA)
		else:
			lossMatrix = cv2.resize(pixelLoss_arr, (8, 8), interpolation = cv2.INTER_CUBIC)
		
		sq = (prob ** 2)
		lossAdj = np.divide(lossMatrix, sq)
		
		node_vals = []
		for i in range(64):
			node_vals.append(0)
		
		for i in range(8):
			for j in range(8):
				val = lossAdj[i,j]
				place = j + (8 * i)
				node_vals[place] = val
		
		return node_vals
	
	def loss_conversion_var(arr, prob, size):
		pixelLoss_arr = np.array(arr)
		(h, w) = pixelLoss_arr.shape[:2]
		lossMatrix = 0
		if (h > size) and (w > size):
			lossMatrix = cv2.resize(pixelLoss_arr, (size, size), interpolation = cv2.INTER_AREA)
		else:
			lossMatrix = cv2.resize(pixelLoss_arr, (size, size), interpolation = cv2.INTER_CUBIC)
		
		sq = (prob ** 2)
		lossAdj = np.divide(lossMatrix, sq)
		
		node_vals = []
		for i in range(size):
			node_row = []
			for j in range(size):
				node_row.append(0)
			node_vals.append(node_row)
		
		for i in range(size):
			for j in range(size):
				val = lossAdj[i,j]
				node_vals[i][j] = val
		
		return node_vals
	
	def PRED2LOSS_HEAD(image, pred_coords, act_coords):
		bounding_box = image[pred_coords[1]:pred_coords[3], pred_coords[0]:pred_coords[2]]
		actual_box = image[act_coords[1]:act_coords[3], act_coords[0]:act_coords[2]]
		pred_artery = ROI_loss.ROI2ART_Advanced(bounding_box)
		act_artery = ROI_loss.ROI2ART_Advanced(actual_box)
		
		(arr, prob, loss) = ROI_loss.pixelLossMatrix(pred_artery, act_artery, bounding_box)
		
		return (arr, prob, loss)
	
	def PRED2LOSS(image, pred_coords, act_coords):
		bounding_box = image[pred_coords[1]:pred_coords[3], pred_coords[0]:pred_coords[2]]
		actual_box = image[act_coords[1]:act_coords[3], act_coords[0]:act_coords[2]]
		pred_artery = ROI_loss.ROI2ART_Advanced(bounding_box)
		act_artery = ROI_loss.ROI2ART_Advanced(actual_box)
		
		(arr, prob, loss) = ROI_loss.pixelLossMatrix(pred_artery, act_artery, bounding_box)
		lossARR = ROI_loss.loss_conversion(arr, prob)
		
		return (lossARR, loss)
	
	def VARIED(arr, prob, loss, var):
		lossARR = ROI_loss.loss_conversion_var(arr, prob, var)
		
		return (lossARR, loss)

	def convolveLOSS(image, pred_coords, act_coords):
		(arr, prob, loss) = ROI_loss.PRED2LOSS_HEAD(image, pred_coords, act_coords)
		(lossARR, _) = ROI_loss.VARIED(arr, prob, loss, 9)
		resMap = algorithm.convolution(lossARR, image, 4)
		return resMap

class antibone:
	def __init__(self):
		pass
		
	def detectExternal(image):
		avg = ROI_loss.bounds(image)
		lower = ROI_loss.boundsCALCIUM(image)
		thresh = process.thresholding(image, lower)
		calcium_spots = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		calcium_lists = []
		calcium_areas = []
		calcium_com = []
		for i in range(len(calcium_spots)):
			minH = 512
			maxH = 0
			minV = 512
			maxV = 0
			section = []
			for j in range(len(calcium_spots[i])):
				H = calcium_spots[i][j][0][0]
				W = calcium_spots[i][j][0][1]
				if H < minH:
					minH = H
				if H > maxH:
					maxH = H
				if W < minW:
					minW = W
				if W > maxW:
					maxW = W
				section.append([H, W])
			totH = 0
			totW = 0
			count = 0
			for i in range(section):
				totH += section[i][0]
				totW += section[i][1]
				count += 1
			comH = totH / count
			comW = totW / count
			calcium_com.append([comH, comW])
			height = maxH - minH
			width = maxW - minW
			lst = ROI_loss.within(height, width, section)
			calcium_lists.append(lst)
			calcium_areas.append(len(lst))
		(h, w) = image.shape[:2]
		left = 5 * w // 16
		right = 13 * w // 16
		top = 3 * h // 16
		bottom = 11 * h // 16
		
		for i in range(len(calcium_com)):
			if (top < calcium_com[0] < bottom) and (left < calcium_com[1] < right):
				if (calcium_areas[i] > 250) and (calcium_com[i][1] < (h // 2)):
					for j in range(len(calcium_lists[i])):
						image[calcium_lists[i][j][0], calcium_lists[i][j][1]] = avg
			else:
				calcium_areas[i] = 0
				for j in range(len(calcium_lists[i])):
					image[calcium_lists[i][j][0], calcium_lists[i][j][1]] = avg
		
		return image
	
	def image_scoring(image):
		extREMOVED = antibone.detectExternal(image)
		(scores, ratios) = process.intensity_rating([extREMOVED], extREMOVED)
		score = scores[0]
		return score
	
	def section_scoring(image, coord_set):
		sects = []
		for i in range(len(coord_set)):
			sect = image[coord_set[i][1]:coord_set[i][3], coord_set[i][0]:coord_set[i][2]]
			sects.append(sect)
		(scores, ratios) = process.intensity_rating(sects, image)
		return scores
	
	def optimized_scoring(image, coord_set):
		extREMOVED = antibone.detectExternal(image)
		sects = []
		for i in range(len(coord_set)):
			sect = extREMOVED[coord_set[i][1]:coord_set[i][3], coord_set[i][0]:coord_set[i][2]]
			sects.append(sect)
		(scores, ratios) = process.intensity_rating(sects, extREMOVED)
		return scores

class convolutionalModifications:
	def __init__(self):
		pass
	
	def convolutionalMaps(image, pred_coords, act_coords):
		(arr, prob, loss) = ROI_loss.PRED2LOSS_HEAD(image, pred_coords, act_coordS)
		(A8, LOSS) = ROI_loss.VARIED(arr, prob, loss, 8)
		(A16, _) = ROI_loss.VARIED(arr, prob, loss, 16)
		(A32, _) = ROI_loss.VARIED(arr, prob, loss, 32)
		(A64, _) = ROI_loss.VARIED(arr, prob, loss, 64)
		
		return (A8, A16, A32, A64, LOSS)
	
	def lossSCALE(arr):
		maxval = 0
		for i in range(len(arr)):
			for j in range(len(arr[i])):
				if arr[i][j] > maxval:
					maxval = arr[i][j]
		
		for i in range(len(arr)):
			for j in range(len(arr[i])):
				val = arr[i][j] / maxval
				scl = val * 255
				arr[i][j] = int(scl)
		
		return arr
	
	def networkMerge(A, S, LOSS)
		factor = np.sqrt(LOSS)
		AS = convolutionalModifications.lossSCALE(A)
		N = np.subtract(S, np.multiply(AS, factor))
		
		return N

	def averagePredictions(genMap, optFind):
		res = np.divide(np.add(genMap, optFind), 2)
		return res

ROI_loss.ROI2ART_Advanced(cv2.imread("NGCT20_IMG/ngct20_40.png", 0))
