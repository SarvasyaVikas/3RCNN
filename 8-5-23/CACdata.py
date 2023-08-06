import csv
import numpy as np

sCH = open("scoresCH.csv", "r")
sCHr = csv.reader(sCH)

for row in sCHr:
    fval = int(float((row[2].replace('"', '').split(","))[0]))
    if fval == 130:
        print(row[4:])
