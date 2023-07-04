from sets_RCA import SETS as RCA
import numpy as np
import csv

class RIGHT:
    def __init__(self):
        pass

    def values():
        RCA_head = []
        RCA_full = []
        for i in range(len(RCA.data(1.0))):
            RCA_full.append(RCA.data(1.0)[i])
            RCA_head.append(RCA.data(1.0)[i][0])
        for i in range(len(RCA.test(1.0))):
            RCA_full.append(RCA.test(1.0)[i])
            RCA_head.append(RCA.test(1.0)[i][0])

        arr = []
        data = open("3RCNN_Data_Annotations.csv", "r")
        datareader = csv.reader(data)

        vals = open("RCA_VALUES.csv", "a+")
        valswriter = csv.writer(vals)

        for row in datareader:
            scan = int(row[1])
            ind = -1
            try:
                ind = RCA_head.index(scan)
            except:
                pass

            if ind != -1:
                if int(row[2]) > int(RCA_full[ind][1]) and int(row[2]) <= int(RCA_full[ind][2]):
                    arr.append(row[0:7])

        for i in range(len(arr)):
            valswriter.writerow(arr[i])

        data.close()
        vals.close()

RIGHT.values()
