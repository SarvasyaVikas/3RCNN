import cv2
import numpy as np
import fnmatch
import os
import argparse
import csv
import time

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--ptn", type = str, default = "6")
ap.add_argument("-t", "--type", type = str, default = "NG")
args = vars(ap.parse_args())

ptn = int(args["ptn"])
CT = args["type"].upper()
num_files = len(fnmatch.filter(os.listdir("{}CT{}_IMG".format(CT, str(ptn))), '*.png'))

LTX_vals = [["RCA"], ["LAD"], ["CX"], ["LM"]]
LTY_vals = [["RCA"], ["LAD"], ["CX"], ["LM"]]
RBX_vals = [["RCA"], ["LAD"], ["CX"], ["LM"]]
RBY_vals = [["RCA"], ["LAD"], ["CX"], ["LM"]]

LX = 0
TY = 0
RX = 0
BY = 0
	
counter = 0

def Interact(action, x, y, flags, *userdata):
	global counter
	global LX
	global TY
	global RX
	global BY

	if action == cv2.EVENT_LBUTTONDBLCLK:
		print('CLICK')
		if counter >= 0:
			counter += 1
			if counter == 1:
				LX = x
				TY = y
			elif counter == 2:
				RX = x
				BY = y
				temp = image.copy()
				cv2.rectangle(temp, (LX, TY), (RX, BY), (0, 0, 255), 2)
				cv2.imshow("Image", temp)
				cv2.waitKey(10)
				
	if action == cv2.EVENT_MBUTTONDOWN:
		counter = 0
		LX = 0
		TY = 0
		RX = 0
		BY = 0
		temp = image.copy()
		print("RESET")
	
	if counter == 1:
		temp = image.copy()
		cv2.rectangle(temp, (LX, TY), (x, y), (0, 255, 0), 2)
		init = time.time()
		while (time.time() - init) < 0.01:
			cv2.imshow("Image", temp)


cv2.namedWindow("Image")
cv2.setMouseCallback("Image", Interact)
		
for files in range(num_files):
	file_num = files + 1
	orig = cv2.imread("{}CT{}_IMG/{}ct{}_{}.png".format(CT, ptn, CT.lower(), ptn, file_num), 1)
	crop = cv2.resize(orig,(0, 0),fx=2, fy=2, interpolation = cv2.INTER_LINEAR)
	(h, w) = crop.shape[:2]
	resized = crop[(h // 4) : (3 * h // 4), (w // 4) : (3 * w // 4)]
	arteries = ["RCA", "LAD", "CX", "LM"]
	
	for i in range(4):
		image = resized.copy()
		(h1, w1) = image.shape[:2]
		cv2.putText(image, arteries[i], (w1 - 75, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4, cv2.LINE_AA)
		
		LX = 0
		TY = 0
		RX = 0
		BY = 0
	
		temp = image.copy()

		while True:
			cv2.imshow("Image", temp)
			
			if cv2.waitKey(10) & 0xFF == ord('q'):
				counter = 0
				break
		
		lx = min(LX, RX) + (w // 4)
		rx = max(LX, RX) + (w // 4)
		ty = min(TY, BY) + (h // 4)
		by = max(TY, BY) + (h // 4)
		
		if lx == rx and rx == ty and ty == by:
			lx = 0
			rx = 0
			ty = 0
			by = 0
		
		print("{}, {}, {}, {}".format(lx, ty, rx, by))
		print("Slice {} Section {}".format(file_num, i + 1))
		
		LTX_vals[i].append(lx)
		LTY_vals[i].append(ty)
		RBX_vals[i].append(rx)
		RBY_vals[i].append(by)

csvfile = open("3RCNN_Data_Annotations.csv", "a+")
csvwriter = csv.writer(csvfile)

for row in range(num_files):
	dtype = CT + "CT"
	lst = [dtype, ptn, row + 1]
	for i in range(4):
		lst.append(LTX_vals[i][row + 1])
		lst.append(LTY_vals[i][row + 1])
		lst.append(RBX_vals[i][row + 1])
		lst.append(RBY_vals[i][row + 1])
	
	csvwriter.writerow(lst)

csvfile.close()
