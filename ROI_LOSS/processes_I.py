import cv2
import numpy as np

class process:
	def __init__(self):
		pass
	
	def sharpen(image):
		kernel = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
		sharpened = cv2.filter2D(image, -1, kernel)
		return sharpened
	
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
		(_, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
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
	
	def calcium_scores(val):
		a = 166.3006356
		b = 0.4202905915
		
		div = val / a
		log = np.log(div)
		lnx = log / b
		dens = np.exp(lnx)
	
		div1 = 200 / a
		log1 = np.log(div1)
		lnx1 = log1 / b
		dens1 = np.exp(lnx1)
		
		multiplier = dens / dens1
		return multiplier
	
	def intensity_rating(cuts):
		scores = []
		ratios = []
		for i in range(len(cuts)):
			(h, w) = cuts[i].shape[:2]
			tot = 0
			pixels = 0
			for j in range(h):
				for k in range(w):
					if cuts[i][j,k] > 200:
						addend = process.calcium_scores(cuts[i][j,k])
						tot += addend
						pixels += 1
			ratio = pixels / (h * w)
			scores.append(tot)
			ratios.append(ratio)
		return (scores, ratios)
	
class ROI_loss:
	def __init__(self):
		pass
	
	def within(h, w, vals):
		artery = []
		for i in range(h):
			for j in range(w):
				horiz = []
				vert = []
				for k in range(vals):
					if vals[k][0] == i:
						horiz.append(vals[k][1])
					if vals[k][1] == j:
						vert.append(vals[k][0])
				
				minH = min(horiz)
				maxH = max(horiz)
				minV = min(vert)
				maxV = max(vert)
				
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
	
	def ROI2ART_Advanced(bounding_box):
		# bounding_box = cv2.GaussianBlur(bounding_box, (7, 7), 0)
		# blurring is antithetical to the close ranges shown below
		orig = bounding_box.copy()
		(h, w) = bounding_box.shape[:2]
		for i in range(h):
			for j in range(w):
				if (bounding_box[i, j] <= 61):
					bounding_box[i, j] = 0
				else:
					bounding_box[i, j] = 65
		
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
				arr[predicted_artery[i][0]][predicted_artery[i][1]] = val
				tot += val
		
		loss = tot / prob
		return (arr, prob, loss)

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
	
	def PRED2LOSS(image, pred_coords, act_coords):
		bounding_box = image[pred_coords[1]:pred_coords[3], pred_coords[0]:pred_coords[2]]
		actual_box = image[act_coords[1]:act_coords[3], act_coords[0]:act_coords[2]]
		pred_artery = ROI_loss.ROI2ART_Advanced(bounding_box)
		act_artery = ROI_loss.ROI2ART_Advanced(actual_box)
		
		(arr, prob, loss) = ROI_loss.pixelLossMatrix(pred_artery, act_artery, bounding_box)
		lossARR = ROI_loss.loss_conversion(arr, prob)
		
		return (lossARR, loss)
	
ROI_loss.ROI2ART_Advanced(cv2.imread("NGCT20_IMG/ngct20_40.png", 0))
