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

apple = int(args["ptn"])
CT = args["type"].upper()
num_files = len(fnmatch.filter(os.listdir("{}CT{}_IMG".format(CT, str(apple))), '*.png'))

for ptn in range(apple, 101):
	print(ptn)
	for files in range(num_files):
		file_num = files + 1
		print(file_num)
		try:
			orig = cv2.imread("NGCT{}_IMG/ngct{}_{}.png".format(ptn, ptn, file_num), 1)
			
			(h, w) = orig.shape[:2]
			left = 5 * w // 16
			right = 13 * w // 16
			top = 3 * h // 16
			bottom = 11 * h // 16
			crop = orig[top:bottom, left:right]	

			path = 'NGCT{}_IMG/resized_ngct{}_{}.png'.format(ptn, ptn, file_num)
			
			cv2.imwrite(path, crop)
			print(file_num)
		except:
			pass
	print(ptn)
