import random

class SETS:
	def __init__(self):
		pass
	
	def data(prob):
		mult = [[1, 0, 40], [3, 40, 60], [4, 20, 40], [5, 40, 60], [6, 40, 60], [10, 0, 40], [12, 50, 70], [13, 50, 70], [14, 50, 70], [15, 30, 50], [17, 25, 45], [18, 40, 60], [19, 70, 90], [22, 60, 80], [23, 0, 60], [24, 40, 60], [26, 45, 65], [27, 20, 60], [28, 50, 70], [29, 50, 70], [30, 58, 78], [31, 10, 30], [32, 40, 80], [33, 40, 80], [34, 40, 100]]
		
		fin = []
		for i in range(len(mult)):
			if random.random() <= prob:
				fin.append(mult[i])
		
		return fin
	
	def imgs(mult):
		tot = 0
		for i in range(len(mult)):
			diff = mult[i][2] - mult[i][1]
			tot += diff
		print(tot)

SETS.imgs(SETS.data(1.0))
