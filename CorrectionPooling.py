import cv2
import numpy as np
import fnmatch
import os
import argparse
import csv
import time
from algorithm import algorithm

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--ptn", type = str, default = "1")
args = vars(ap.parse_args())

apple = int(args["ptn"])
num_files = len(fnmatch.filter(os.listdir("NGCT{}_IMG".format(str(apple))), '*.png'))

for ptn in range(apple, 101):
	print(ptn)
	for files in range(num_files):
		try:
			file_num = files + 1
			orig = cv2.imread("NGCT{}_IMG/resized_ngct{}_{}.png".format(str(ptn), str(ptn), file_num), 1)

			pooled = algorithm.max_pooling(orig)

			path = 'NGCT{}_IMG/pooled_ngct{}_{}.png'.format(ptn, ptn, file_num)
				
			cv2.imwrite(path, pooled)
			print(file_num)
		except:
			pass
	
	print(ptn)
