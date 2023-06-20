import csv

csvfile = open("modifiedBranchIDs.csv", "a+")
csvwriter = csv.writer(csvfile)

deltas = ["BASE", "nesterovMomentum", "maclaurin", "learning_rate", "gradient", "determinants", "dropout"]
imgs = ["BASE", "averageFeatureMaps", "maxFeatureMaps", "imageMasks", "states", "frameDifferences", "planeRecurrence"]

for i in range(7):
	for j in range(7):
		version = (7 * i) + j + 6
		lst = ["Network {}".format(version), "Delta {}".format(i), deltas[i], "Images {}".format(j), imgs[j]]
		csvwriter.writerow(lst)

csvfile.close()
