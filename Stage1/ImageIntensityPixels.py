import csv
import cv2
import numpy as np
from sets_LAD_SCL_NEW import sets_LAD_SCL as SETS
from processes_X_M import process

ptn = 27 

mult = SETS.data()
scans = []
for el in mult:
    scans.append(el[0])

place = scans.index(ptn)
st = mult[place]

refc = open("REVALS_FC_2.csv", "r")
rewr = csv.reader(refc)
coords = []
for row in rewr:
    cr = []
    for i in range(7, 11):
        cr.append(int(float(row[i])))
    coords.append(cr)
refc.close()
CAC = 0
for i in range(st[1], st[2]):
    path = "SCL/NGCT{}_SCL/ngct{}_{}.png".format(ptn, ptn, i)
    image = cv2.imread(path, 0)
    img = cv2.resize(image, (1024, 1024))
    u = i - st[1]
    coord = coords[u]
    spliced = img[coord[1]:coord[3], coord[0]:coord[2]]
    fov = 0
    with open("NGCT_CAC_Scores.csv", "r") as s:
        sr = csv.reader(s)
        srl = list(sr)
        fov = int(srl[ptn][-2])
    scr = process.cut_scoring(spliced, fov, 0)
    CAC += scr
    (h, w) = spliced.shape[:2]
    if h != 0 and w != 0:
        lst = []
        for i in range(h):
            for j in range(w):
                lst.append(spliced[i, j] - 1024)
        print(lst)
    #with open("records.csv", "a+") as r:
    #    (csv.writer(r)).writerow(lst)
print(CAC)
