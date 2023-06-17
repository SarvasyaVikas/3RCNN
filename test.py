import cv2
import numpy as np
import csv
import algorithm


def anticonvolution(F, image, pad): # applying convolutional kernel
		print("a")
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
					val = vector[m] * area[m]
					tot += val
				
				fRow.append(tot) 
			
			fMap.append(fRow) # creates array feature map

		return fMap

a = anticonvolution(1, 1, 1)

rows = []
with open("3RCNN_Data_Annotations.csv") as csvfile:
	csvreader = csv.reader(csvfile, delimiter = ",")
	for row in csvreader:
		if int(row[1]) == ptn:
			rows.append(row)

images = []
actuals = []
IDs = []
for i in range(len(rows) - 4):
	IDs.append(i + 1)
	vals1 = [int(rows[i][2]), int(rows[i][3]), int(rows[i][4]), int(rows[i][5])]
	vals2 = [int(rows[i + 1][2]), int(rows[i + 1][3]), int(rows[i + 1][4]), int(rows[i + 1][5])]
	vals3 = [int(rows[i + 2][2]), int(rows[i + 2][3]), int(rows[i + 2][4]), int(rows[i + 2][5])]
	vals4 = [int(rows[i + 3][2]), int(rows[i + 3][3]), int(rows[i + 3][4]), int(rows[i + 3][5])]
	vals5 = [int(rows[i + 4][2]), int(rows[i + 4][3]), int(rows[i + 4][4]), int(rows[i + 4][5])]
	actuals.append(vals1)
	actuals.append(vals2)
	actuals.append(vals3)
	actuals.append(vals4)
	actuals.append(vals5)

for i in range(5):
	image = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, IDs[i]), 1)
	images.append(image)
