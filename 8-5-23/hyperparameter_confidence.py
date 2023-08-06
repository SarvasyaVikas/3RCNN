import numpy as np
import csv

def softmax(sm):
    tot = 0
    conf = 0
    for i in range(len(sm)):
        val = (np.e ** sm[i])
        tot += val
        if i == (len(sm) - 1):
            conf = val

    scr = float(conf / tot)
    # scr should maybe be above 0.3 to generate values
    return scr

def test(bins, cnfs, T):
    tot = len(bins)
    cor = [0, 0]
    oth = [0, 0]
    prob = 0
    for i in range(tot):
        if (cnfs[i] < T) and (bins[i] == 1):
            cor[1] += 1
        elif (cnfs[i] > T) and (bins[i] == 0):
            cor[0] += 1
        elif (cnfs[i] > T) and (bins[i] == 1):
            oth[1] += 1
        elif (cnfs[i] < T) and (bins[i] == 0):
            oth[0] += 1
        if (bins[i] == 1):
            prob += 1

    otfrac = prob / tot
    if T == .5:
        print(otfrac)
    newfrac = float(sum(oth) / tot)
    frac = float(sum(cor) / tot)
    direc = True
    if frac > newfrac:
        direc = False
    fin = np.divide(cor, tot)
    if direc:
        fin = np.divide(oth, tot)
    # fraction of correct
    return (max(frac, newfrac), direc, fin)

splices = open("splices.csv", "r")
bins = []
cnfs = []
splicereader = csv.reader(splices)
for row in splicereader:
    if float(row[6]) != 0:
        bins.append(1)
    else:
        bins.append(0)
    sm = [float(row[10]), float(row[11])]
    scr = softmax(sm)
    cnfs.append(scr)
splices.close()
maxlst1 = [0, 0]
maxlst2 = [0, 0]
maxlst = [0, 0]
for i in range(1000):
    T = round((i / 1000.0), 3)
    (val, direc, fin) = test(bins, cnfs, T)
    if val > 0.714:
        print([T, val, direc, fin[0], fin[1]])
    if val > maxlst[1]:
        maxlst = [T, val, direc, fin[0], fin[1]]
    if fin[0] > maxlst1[0]:
        maxlst1 = [T, val, direc, fin[0], fin[1]]
    if fin[1] > maxlst2[1]:
        maxlst2 = [T, val, direc, fin[0], fin[1]]
print(maxlst)
print(maxlst1)
print(maxlst2)
