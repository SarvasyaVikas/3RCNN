import cv2
import numpy as np
import csv
import fnmatch
import os
import csv
from FunctionalNetwork_II import FunctionalNetwork
from network import network
from optimizerMV import optimizerMV
import time
from Modifications import Modifications

# structure generation
CV1 = network.generate_filters(16, 5)
CV2 = network.generate_filters(16, 5)
CV3 = network.generate_filters(32, 3)
CV4 = network.generate_filters(64, 3)
FC5 = []
FC6 = []
SF = []
network_filters = [CV1, CV2, CV3, CV4]
network_nodes = [FC5, FC6]
networkN = [network_filters, network_nodes, SF]

for i in range(5):
	FC5.append(network.generate_layer(64, 16))
	FC6.append(network.generate_layer(16, 4))
	SF.append(network.generate_layer(4, 2))

# ABOVE THIS: DO NOT TOUCH
#
#

# This is for RCA

# 1 scans
val = 40

def save(networkN, code, losses):
	filters = networkN[0]
	nodes = networkN[1]
	SF = networkN[2]
	
	CV1 = filters[0]
	CV2 = filters[1]
	CV3 = filters[2]
	CV4 = filters[3]
	
	FC5 = nodes[0]
	FC6 = nodes[1]
	
	lst = [code]
	lst.append(min(losses))
	for i in range(len(losses)):
		lst.append(losses[i])
	lst.append("S")
	for i in range(5):
		for a in range(len(filters)):
			for j in range(len(filters[a][i])):
				for k in range(len(filters[a][i][j])):
					for l in range(len(filters[a][i][j][k])):
						lst.append(filters[a][i][j][k][l])
		for b in range(len(nodes)):
			for j in range(len(nodes[b][i])):
				for k in range(len(nodes[b][i][j][0])):
					lst.append(nodes[b][i][j][0][k])
				lst.append(nodes[b][i][j][1])
			
		for c in range(len(SF[i])):
			for j in range(len(SF[i][c][0])):
				lst.append(SF[i][c][j][0][k])
			lst.append(SF[i][c][j][1])
	
	print(lst)
	csvfile = open("saved.csv", "a+")
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(lst)
	csvfile.close()
	

def div(lst, val):
	fin = []
	for i in range(len(lst)):
		new = lst[i] / val
		fin.append(new)
	return fin

def data(ptn, val = 0):
	rows = []
	with open("3RCNN_Data_Annotations.csv") as csvfile:
		csvreader = csv.reader(csvfile, delimiter = ",")
		for row in csvreader:
			if int(row[1]) == ptn:
				rows.append(row)
	if val == 0:
		val = len(rows) - 4
	images = []
	actuals = []
	for i in range(val):
		vals1 = [int(rows[i][3]), int(rows[i][4]), int(rows[i][5]), int(rows[i][6])]
		vals2 = [int(rows[i + 1][3]), int(rows[i + 1][4]), int(rows[i + 1][5]), int(rows[i + 1][6])]
		vals3 = [int(rows[i + 2][3]), int(rows[i + 2][4]), int(rows[i + 2][5]), int(rows[i + 2][6])]
		vals4 = [int(rows[i + 3][3]), int(rows[i + 3][4]), int(rows[i + 3][5]), int(rows[i + 3][6])]
		vals5 = [int(rows[i + 4][3]), int(rows[i + 4][4]), int(rows[i + 4][5]), int(rows[i + 4][6])]
		actuals.append([div(vals1, 1024), div(vals2, 1024), div(vals3, 1024), div(vals4, 1024), div(vals5, 1024)])

		image1 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 1), 0)
		image2 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 2), 0)
		image3 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 3), 0)
		image4 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 4), 0)
		image5 = cv2.imread("NGCT{}_IMG/pooled_ngct{}_{}.png".format(ptn, ptn, i + 5), 0)
		image1 = cv2.resize(image1, (128, 128))
		image2 = cv2.resize(image2, (128, 128))
		image3 = cv2.resize(image3, (128, 128))
		image4 = cv2.resize(image4, (128, 128))
		image5 = cv2.resize(image5, (128, 128))
		images.append([image1, image2, image3, image4, image5])
		
	return (images, actuals)

losses = [64, 64]
nn = 0
neuralNetworks = [networkN]
# scan 1
scan_start = time.time()
num_files = len(fnmatch.filter(os.listdir("NGCT1_IMG"), '*.png'))
num_files = num_files // 3
(Is1, As1) = data(1, val)
for i in range(val):
	start = time.time()
	print(i)
	print("f")
	(networkN, error) = FunctionalNetwork.DRCNN(Is1[i], As1[i], networkN, losses[-1], neuralNetworks)
	losses.append(error)
	print(error)
	
	if error < losses[-2]:
		nn = neuralNetworks[-1]
	neuralNetworks.append(networkN)
	end = time.time()
	print(end - start)
	
scan_end = time.time()
print(scan_end - scan_start)
losses.pop(0)
losses.pop(0)
print(losses)
save(nn, "SCAN1_ATT1", losses)

# scan 10
scan_start = time.time()
num_files2 = len(fnmatch.filter(os.listdir("NGCT10_IMG"), '*.png'))
num_files2 = num_files2 // 3
(Is2, As2) = data(10, val)
for i in range(val):
	start = time.time()
	print(i)
	print("f")
	(networkN, error) = FunctionalNetwork.DRCNN(Is2[i], As2[i], networkN, losses[-1], neuralNetworks)
	losses.append(error)
	print(error)
	
	if error < losses[-2]:
		nn = neuralNetworks[-1]
	neuralNetworks.append(networkN)
	end = time.time()
	print(end - start)
	
scan_end = time.time()
print(scan_end - scan_start)
# losses data graph
losses.pop(0)
losses.pop(0)
print(losses)
save(nn, "SCAN10_ATT1", losses)
