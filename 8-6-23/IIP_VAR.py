import csv
import cv2
import numpy as np
from sets_LAD_SCL_NEW import sets_LAD_SCL as SETS
from processes_VAR import process
from mpi4py import MPI

# ptn = 62

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

t = 120 + (rank // 16 * 5)
k = 20 + (rank // 7)

#
#
#

mult = SETS.data()
print(mult)
scans = []
for el in mult:
    scans.append(el[0])
itera = 1
scn = [78, 89, 27, 74, 55, 19, 80, 96, 56, 85, 79, 5, 18, 28, 32, 93, 97, 29, 3, 47, 82, 51, 62]
under = [33, 6, 35, 64, 57, 26]
lsts = []
for st in mult:
    # place = scans.index(ptn)
    # st = mult[place]
    ptn = st[0]

    refc = open("BB_DATA/REVALS_7_UPD_ANNO.csv", "r")
    rewr = csv.reader(refc)
    coords = []
    for row in rewr:
        cr = []
        if int(row[1]) == ptn:
        	for i in range(3, 7):
        		cr.append(int(float(row[i])))
        	coords.append(cr)
    refc.close()
    CAC = []
    for i in range(itera):
        CAC.append(0)
    for i in range(int(st[1]), int(st[2])):
        path = "SCL/NGCT{}_SCL/ngct{}_{}.png".format(ptn, ptn, i)
        image = cv2.imread(path, 0)
        img = cv2.resize(image, (1024, 1024))
        u = int(i - st[1])
        coord = coords[u]
        spliced = img[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
        fov = 0
        act = 0
        with open("NGCT_CAC_Scores.csv", "r") as s:
            sr = csv.reader(s)
            srl = list(sr)
            fov = int(srl[ptn][-2])
            act = round(float(srl[ptn][2]), 3)
        for j in range(itera):
            scr1024 = process.cut_scoring(spliced, fov, t, k)
            scr512 = scr1024 / 4.0
            CAC[j] += scr512
        (h, w) = spliced.shape[:2]
        if h != 0 and w != 0:
            lst = []
            for i in range(h):
                for j in range(w):
                    lst.append(spliced[i, j])
    fin = [ptn, t, k, act]
    for i in range(itera):
        fin.append(CAC[i])
    
    lsts.append(fin)
    #print(fin)
    #with open("resultsFiles/CAC_VARIABLE.csv", "a+") as M:
    #    (csv.writer(M)).writerow(fin)
pdifftot = 0
count = 0
for i in range(lsts):
	act = lsts[i][3]
	calc = lsts[i][4]
	avg = (act + calc) / 2.0
	diff = abs(act - calc)
	pdiff = diff / avg * 100
	pdifftot += pdiff
	count += 1

pdavg = pdifftot / count if count != 0 else 1000

pd = open("pdiffs.csv", "a+")
(csv.writer(pd)).writerow([lsts[0][1], lsts[0][2], pdavg])
pd.close()
