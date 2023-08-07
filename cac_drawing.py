import csv
import cv2

bb = open("resultsFiles/8-7-23/LAD_BB_FULL_TRAINING_RESULTS_8.csv", "r")
bbr = csv.reader(bb)
bbrl = list(bbr)

for r in range(len(bbrl)):
	row = bbrl[r]
	lst = row[0].split("_")
	scan = int(lst[0][4:]) if len(lst) > 1 else 0
	slce = int(lst[1][5:]) if len(lst) > 1 else 0
	row = [int(float(o)) for o in row[1:]] if r != 0 else 0
	if scan == 53 and slce == 53 and r > 1200:
		nets = row[1:5]
		acts = row[5:9]
		print(nets)
		print(acts)
		image = cv2.imread("SCL/NGCT53_SCL/ngct53_55.png", 1)
		cv2.imshow("i", image)
		cv2.waitKey(0)
		image = cv2.rectangle(image, (nets[0], nets[1]), (nets[2], nets[3]), (0, 0, 255), 2)
		image = cv2.rectangle(image, (acts[0], acts[1]), (acts[2], acts[3]), (0, 255, 0), 2)
		cv2.imwrite("NO_CALCIUM_LAD_EX.png", image)
	if scan == 97 and slce == 61 and r > 1200:
		nets = row[1:5]
		acts = row[5:9]
		image = cv2.imread("SCL/NGCT53_SCL/ngct97_61.png", 1)
		image = cv2.rectangle(image, (nets[0], nets[1]), (nets[2], nets[3]), (0, 0, 255), 2)
		image = cv2.rectangle(image, (acts[0], acts[1]), (acts[2], acts[3]), (0, 255, 0), 2)
		cv2.imwrite("SOME_CALCIUM_LAD_EX.png", image)
	if scan == 78 and slce == 78 and r > 1200:
		nets = row[1:5]
		acts = row[5:9]
		image = cv2.imread("SCL/NGCT78_SCL/ngct78_78.png", 1)
		image = cv2.rectangle(image, (nets[0], nets[1]), (nets[2], nets[3]), (0, 0, 255), 2)
		image = cv2.rectangle(image, (acts[0], acts[1]), (acts[2], acts[3]), (0, 255, 0), 2)
		cv2.imwrite("MUCH_CALCIUM_LAD_EX.png", image)
