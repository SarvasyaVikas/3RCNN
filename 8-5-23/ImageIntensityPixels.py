import csv
import cv2
import numpy as np
from sets_LAD_SCL_NEW import sets_LAD_SCL as SETS
from processes_X_M import process

# ptn = 62

mult = SETS.data()
scans = []
for el in mult:
    scans.append(el[0])

scn = [78, 89, 27, 74, 55, 19, 80, 96, 56, 85, 79, 5, 18, 28, 32, 93, 97, 29, 3, 47, 82, 51, 62]
for ptn in scn:
    place = scans.index(ptn)
    st = mult[place]

    refc = open("REVALS_FC_2.csv", "r")
    rewr = csv.reader(refc)
    coords = []
    for row in rewr:
        cr = []
        if int(row[1]) == ptn:
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
        act = 0
        with open("NGCT_CAC_Scores.csv", "r") as s:
            sr = csv.reader(s)
            srl = list(sr)
            fov = int(srl[ptn][-2])
            act = round(float(srl[ptn][2]), 3)
        scr = process.cut_scoring(spliced, fov, 101)
        CAC += scr
        (h, w) = spliced.shape[:2]
        if h != 0 and w != 0:
            lst = []
            for i in range(h):
                for j in range(w):
                    lst.append(spliced[i, j])
    print([ptn, CAC, act])
