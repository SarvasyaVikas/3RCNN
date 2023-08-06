import cv2
import numpy as np
from processes_VIII import process

class analysisIOU:
	def basicIOU(image2, bb1):
		return process.intersection_over_union(image2, bb1)
	
	def adaptiveIOU(image3, bb1, bb2):
		center1 = [(bb1[0] + bb1[2]) / 2.0, (bb1[1] + bb1[3]) / 2.0]
		center2 = [(bb2[0] + bb2[2]) / 2.0, (bb2[1] + bb2[3]) / 2.0]
		centerDIFF = np.subtract(center2, center1)
		centerPRED = np.add(center2, centerDIFF)
		cornerPRED = np.add(centerPRED, centerDIFF)
		
		LX = min(centerDIFF[0], cornerPRED[2])
		RX = max(centerDIFF[0], cornerPRED[2])
		TY = min(centerDIFF[1], cornerPRED[3])
		BY = max(centerDIFF[1], cornerPRED[3])
		
		proposal = image3[LX:RX, TY:BY]
		return proposal
	
	def smartIOU(imageNEXT, BBprevs, IOUprevs):
		# len(BBprevs) should be equal len(IOUprevs)
		# however, BBprevs starts one later than IOUprevs
		
		BBcenterX = []
		IOUcenterX = []
		diffX = []
		BBcenterY = []
		IOUcenterY = []
		diffY = []
		count = []
		for i in range(len(BBprevs)):
			BBcenter = [(BBprevs[i][0] + BBprevs[i][2]) / 2.0, (BBprevs[i][1] + BBprevs[i][3]) / 2.0]
			IOUcenter = [(IOUprevs[i][0] + IOUprevs[i][2]) / 2.0, (IOUprevs[i][1] + IOUprevs[i][3]) / 2.0]
			
			BBcenterX.append(BBcenter[0])
			IOUcenterX.append(IOUcenter[0])
			BBcenterY.append(BBcenter[0])
			IOUcenterY.append(IOUcenter[0])
			diff = np.subtract(BBcenter, IOUcenter)
			diffX.append(diff[0])
			diffY.append(diff[1])
			count.append(i)
		
		z = np.polyfit(np.array(count), np.array(diffs), 2)
		p = np.poly1d(z)
		
		y = np.polyfit(np.array(count), np.array(IOUcenters), 2)
		o = np.poly1d(y)
		
		x = np.polyfit(np.array(count), np.array(BBcenters), 2)
		n = np.poly1d(x)
		
		w = np.polyfit(np.array(count), np.array(diffs), 2)
		m = np.poly1d(w)
		
		v = np.polyfit(np.array(count), np.array(IOUcenters), 2)
		l = np.poly1d(v)
		
		u = np.polyfit(np.array(count), np.array(BBcenters), 2)
		k = np.poly1d(u)
		
		pred_diffX = p(len(count))
		pred_IOUX = o(len(count))
		pred_BBX = n(len(count))
		pred_diffY = m(len(count))
		pred_IOUY = l(len(count))
		pred_BBY = k(len(count))
		
		pred_centX = (pred_IOUX + pred_BBX) / 2.0
		pred_centY = (pred_IOUY + pred_BBY) / 2.0
		
		LX = pred_centX - abs(diffX)
		RX = pred_centX + abs(diffX)
		TY = pred_centY - abs(diffY)
		BX = pred_centY + abs(diffY)
		
		proposal = imageNEXT[LX:RX, TY:BY]
		
		return proposal
