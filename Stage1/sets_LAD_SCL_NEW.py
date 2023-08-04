import csv

class sets_LAD_SCL:
	def __init__(self):
		pass
	
	def data():
		mult = []

		revals = open("REVALS.csv", "r")
		revalsreader = csv.reader(revals)

		scans = []
		count = []
		for row in revalsreader:
			if float(row[1]) not in scans:
				scans.append(float(row[1]))
				count.append([float(row[2]), float(row[2])])
			else:
				count[-1][1] = float(row[2])

		for i in range(len(scans)):
			lst = [scans[i], count[i][0], count[i][1] + 1]
			mult.append(lst)

		return mult
