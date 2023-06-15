import cv2
import numpy as np
import csv
import fnmatch
import os
from FunctionalNetwork import FunctionalNetwork
from network import network

# structure generation
CV1 = network.generate_filters(16, 5)
CV2 = network.generate_filters(16, 5)
CV3 = network.generate_filters(32, 3)
CV4 = network.generate_filters(64, 3)
FC5 = []
FC6 = []
network_filters = [CV1, CV2, CV3, CV4]
network_nodes = [FC5, FC6]
networkN = [network_filters, network_nodes]

for i in range(5):
	FC5.append(network.generate_layer(64, 16))
	FC6.append(network.generate_layer(16, 4))

# ABOVE THIS: DO NOT TOUCH
#
#

# This is for RCA

def data(ptn):
	rows = []
	with open("3RCNN_Data_Annotations.csv") as csvfile:
		csvreader = csv.reader(csvfile, delimiter = ",")
		for row in csvreader:
			if int(row[1]) == ptn:
				rows.append(row)

	images = []
	actuals = []
	for i in range(len(rows) - 4):
		vals1 = [int(rows[i][2]), int(rows[i][3]), int(rows[i][4]), int(rows[i][5])]
		vals2 = [int(rows[i + 1][2]), int(rows[i + 1][3]), int(rows[i + 1][4]), int(rows[i + 1][5])]
		vals3 = [int(rows[i + 2][2]), int(rows[i + 2][3]), int(rows[i + 2][4]), int(rows[i + 2][5])]
		vals4 = [int(rows[i + 3][2]), int(rows[i + 3][3]), int(rows[i + 3][4]), int(rows[i + 3][5])]
		vals5 = [int(rows[i + 4][2]), int(rows[i + 4][3]), int(rows[i + 4][4]), int(rows[i + 4][5])]
		actuals.append([vals1, vals2, vals3, vals4, vals5])

		image1 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 1), 1)
		image2 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 2), 1)
		image3 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 3), 1)
		image4 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 4), 1)
		image5 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 5), 1)
		images.append([image1, image2, image3, image4, image5])
		
	return (images, actuals)

losses = []
nn = 0

# scan 1
num_files = len(fnmatch.filter(os.listdir("NGCT1_IMG"), '*.png'))
num_files = num_files // 3
(Is1, As1) = data(1)
for i in range(num_files - 4):
	(networkN, loss) = FunctionalNetwork.DRCNN(Is1[i], As1[i], networkN)
	error = 0
	for j in range(len(loss)):
		for k in range(len(loss[0])):
			error += loss[j][k]
	losses.append(error)
	
	if i == num_files - 5:
		nn = networkN

print(nn)
print(losses)

# scan 2
num_files2 = len(fnmatch.filter(os.listdir("NGCT2_IMG"), '*.png'))
num_files2 = num_files2 // 3
(Is2, As2) = data(2)
for i in range(num_files2 - 4):
	(networkN, loss) = FunctionalNetwork.DRCNN(Is2[i], As2[i], networkN)
	error = 0
	for j in range(len(loss)):
		for k in range(len(loss[0])):
			error += loss[j][k]
	losses.append(error)
	
	if i == num_files2 - 5:
		nn = networkN

# losses data graph
print(nn)
print(losses)
