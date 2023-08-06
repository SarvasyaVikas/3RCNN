import csv
import cv2
import numpy as np
from sets_LAD_SCL_NEW import sets_LAD_SCL as LAD
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

mult = LAD.data()

F = open("scoresCH.csv", "r")
FR = csv.reader(F)

perdiffs = 0
count = 0

for row in FR:
    value = int(float((row[2].replace('"', '').split(","))[0]))
    if value == rank + 75:
       pd = round(float(row[-1]), 3)
       perdiffs += pd
       count += 1

F.close()

AVG = perdiffs / count

with open("scores_parser.csv", "a+") as G:
    (csv.writer(G)).writerow([rank, rank + 75, "LAD", perdiffs, count, AVG])

if rank != 0:
    comm.send(AVG, dest = 0, tag = rank)
else:
    AVGS = [AVG]
    for i in range(1, size):
        avg = comm.recv(source = i, tag = i)
        AVGS.append(avg)

    sortedAVG = sorted(AVGS)
    resorted = []
    for i in range(len(sortedAVG)):
        resorted.append(AVGS.index(sortedAVG[i]))

    with open("scores_parser.csv", "a+") as G:
        (csv.writer(G)).writerow(AVGS)
        (csv.writer(G)).writerow(resorted)
