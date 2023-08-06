import csv
import numpy as np

f = open("bb5.csv", "r")
h = open("bb3.csv", "r")
g = open("bb6.csv", "a+")
hr = csv.reader(h)
fr = csv.reader(f)
gw = csv.writer(g)
for row in hr:
    if len(row) == 10:
        scan = int(float((row[0].split("_")[0])[4:]))
        if scan < 20:
            for i in range(4):
                row.append(abs(float(row[1 + i]) - float(row[5 + i])))
            gw.writerow(row)
for row in fr:
    if len(row) == 10:
        for i in range(4):
            row.append(abs(float(row[1 + i]) - float(row[5 + i])))
        gw.writerow(row)

f.close()
g.close()
h.close()
