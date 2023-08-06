import csv
import cv2
import numpy as np
from sets_LAD_SCL_NEW import sets_LAD_SCL as LAD
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

mult = LAD.data()

F = open("scoresCHY.csv", "r")
FR = csv.reader(F)

perdiffs = []
count = []

for i in range(56):
    perdiffs.append(0)
    count.append(0)

for row in FR:
    values = (row[2].replace('"', '').split(","))
    valx = int(float(values[0]))
    valy = int(float(values[1]))
    if valy == rank + 131:
       pd = round(float(row[-1]), 3)
       perdiffs[valx - 75] += pd
       count[valx - 75] += 1

F.close()

AVG = []
for i in range(len(perdiffs)):
    if count[i] != 0:
        AVG.append(perdiffs[i] / count[i])
    else:
        AVG.append(100000000)

with open("scores_parser_2.csv", "a+") as G:
    for i in range(56):
        (csv.writer(G)).writerow([rank, rank + 131, i, i + 75, "LAD", perdiffs[i], count[i], AVG[i]])

if rank != 0:
    comm.send(AVG, dest = 0, tag = rank)
else:
    AVGS = [AVG]
    for i in range(1, size):
        avg = comm.recv(source = i, tag = i)
        AVGS.append(avg)

    minAVGS = []
    for i in range(70):
        minAVG = 1000
        for j in range(56):
            if AVGS[i][j] < minAVG:
                minAVG = AVGS[i][j]
        minAVGS.append(minAVG)

    ind = minAVGS.index(min(minAVGS))
    iAVGS = AVGS[ind]

    print(iAVGS)
    subind = iAVGS.index(min(iAVGS))

    with open("scores_parser_2.csv", "a+") as G:
        (csv.writer(G)).writerow([ind, subind, AVGS[ind][subind]])
