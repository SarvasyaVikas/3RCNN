import numpy as np
import cv2
import random
import math
from algorithm import algorithm
from techniques import techniques
from SNN import SNN
from network import network
from optimizerMV import optimizerMV
from Modifications import Modifications
import time
from mpi4py import MPI

class parallel:
	def __init__(self):
		pass
	
	def generate_feature_map(filt, img):
		pad_val = len(filt) // 2
		fMap = algorithm.convolution(filt, img, pad_val)
		fMapImg = np.array(fMap)
		return fMapImg
	
	
