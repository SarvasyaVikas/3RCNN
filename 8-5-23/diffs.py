import csv
import numpy as np

f = open("bb1.csv", "r")
fr = csv.reader(f)
count10 = 0
tot10 = 0
countU = 0
totU = 0
for row in fr:
    if len(row) == 14:
        d = float(row[10]) + float(row[11]) + float(row[12]) + float(row[13])
        if d > 220:
            count10 += 1
            tot10 += d
        else:
            countU += 1
            totU += d

f.close()
avg10 = tot10 / count10
avgU = totU / countU
print(count10)
print(countU)
print(avgU)
