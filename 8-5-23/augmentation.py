from processes_VII import process
from processes_VII import ROI_loss
from processes_VII import antibone as ab
import cv2
import numpy as np
from algorithm import algorithm

class procAugment:
	def __init__(self):
		pass
	
	def sharpening(image, itera):
		for i in range(itera):
			image = process.sharpen(image)
		return image
	
	def smoothing(image, itera):
		for i in range(itera):
			image = process.smooth(image)
		return image
	
	def cleaning_thresh(image):
		(h, w) = image.shape[:2]
		new = image.copy()
		for i in range(h):
			for j in range(w):
				if image[i,j] < 50:
					new[i,j] = 0
		
		return new

	def cleaning_erode(image):
		thresh = process.thresholding(image, 50)
		new = process.erosion_dilation(thresh, 10)
		return new
	
	def mask(orig, proc):
		(h, w) = orig.shape[:2]
		new = orig.copy()
		for i in range(h):
			for j in range(w):
				if proc[i,j] == 0:
					new[i,j] = 0
		
		return new
	
	def minicontours(image):
		contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
		new_contours = []
		area_contours = []
		areas = []
		for contour in contours:
			new_contour = []
			minH = 1000
			maxH = 0
			minW = 1000
			maxW = 0
			for coord in contour:
				new_coord = [coord[0][0], coord[0][1]]
				if coord[0][0] < minH:
					minH = coord[0][0]
				if coord[0][0] > maxH:
					maxH = coord[0][0]
				if coord[0][1] < minW:
					minW = coord[0][1]
				if coord[0][1] > maxW:
					maxW = coord[0][1]
				new_contour.append(new_coord)
			new_contours.append(new_contour)
			diffH = maxH - minH
			diffW = maxW - minW
			area_contour = ROI_loss.within(diffH, diffW, new_contour)
			area_contours.append(area_contour)
			areas.append(len(area_contour))
		
		new = image.copy()
		for i in range(len(areas)):
			if areas[i] < 4000:
				for j in range(area_contours[i]):
					new[area_contours[i][j][0], area_contours[i][j][1]] = 0
		
		return new

class manipAugment:
	def __init__(self):
		pass
	
	def darker(image, fac):
		(h, w) = image.shape[:2]
		new = image.copy()
		for i in range(h):
			for j in range(w):
				newval = image[i,j] * fac
				new[i,j] = int(newval)
		
		return new
	
	def lighter(image, fac):
		(h, w) = image.shape[:2]
		new = image.copy()
		for i in range(h):
			for j in range(w):
				newval = image[i,j] * fac + (255 * (1 - fac))
				new[i,j] = int(newval)
		
		return new

	def rotate(image, degree):
		(h, w) = image.shape[:2]
		M = cv2.getRotationMatrix2D((w // 2, h // 2), degree, 1)
		rotated = cv2.warpAffine(image, M, (w, h))
		return rotated

class imagepipeline:
	def __init__(self):
		pass
	
	def default(image512):
		extREMOVED = ab.detectExternel(image512)
		eroded = procAugment.cleaning_erode(extREMOVED)
		cleaned = procAugment.minicontours(eroded)
		masked = procAugment.mask(image512, cleaned)
		cropped = masked[96:352, 160:416]
		pooled = algorithm.max_pooling(cropped)
		# returns 128x128
		return pooled
	
	def edge_detector(default_pipe):
		edged = cv2.Canny(default_pipe, 50, 150)
		return edged
		
class variations:
	def __init__(self):
		pass
	
	def full_set(image, factors, degrees, iteras):
		default_pipe = imagepipeline.default(image)
		variance = [image]
		for factor in factors:
			variance.append(manipAugment.darker(image, factor)
			variance.append(manipAugment.lighter(image, factor)
		for degree in degrees:
			variance.append(manipAugment.rotate(image, degree)
		for itera in iteras:
			variance.append(procAugment.sharpening(image, itera)
			variance.append(procAugment.smoothing(image, itera)
		return variance
	
	def full_set_call_default():
		factors = [0.2, 0.4, 0.6, 0.8]
		degrees = [45, 90, 135, 180, 225, 270, 315]
		iteras = [1, 2, 3]
		return (factors, degrees, iteras)
	
	def generate_full_set(image):
		(factors, degrees, iteras) = variations.full_set_call_default()
		variance = variations.full_set(image, factors, degrees, iteras)
		return variance
