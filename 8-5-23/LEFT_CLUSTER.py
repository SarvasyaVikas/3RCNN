from sets_LAD import SETS as LAD
from sets_CX import SETS as CX
from sets_LM import SETS as LM
import numpy as np
import csv

class CLUSTER:
    def __init__(self):
        pass

    def LEFT_TRAIN():
        left_train = []
        left_test = []

        LAD_LST = []
        CX_LST = []
        LM_LST = []
        for i in range(len(LAD.data(1.0))):
            LAD_LST.append(LAD.data(1.0)[i][0])
        for i in range(len(CX.data(1.0))):
            CX_LST.append(CX.data(1.0)[i][0])
        for i in range(len(LM.data(1.0))):
            LM_LST.append(LM.data(1.0)[i][0])

        for j in range(len(LAD.test(1.0))):
            LAD_LST.append(LAD.test(1.0)[j][0])
        for j in range(len(CX.test(1.0))):
            CX_LST.append(CX.test(1.0)[j][0])
        for j in range(len(LM.test(1.0))):
            LM_LST.append(LM.test(1.0)[j][0])

        for i in range(1, 101):
            LAD_val = -1
            CX_val = -1
            LM_val = -1
            LAD_full = []
            CX_full = []
            LM_full = []

            for j in range(len(LAD.data(1.0))):
                LAD_full.append(LAD.data(1.0)[j])
            for j in range(len(LAD.test(1.0))):
                LAD_full.append(LAD.test(1.0)[j])
            for j in range(len(CX.data(1.0))):
                CX_full.append(CX.data(1.0)[j])
            for j in range(len(CX.test(1.0))):
                CX_full.append(CX.test(1.0)[j])
            for j in range(len(LM.data(1.0))):
                LM_full.append(LM.data(1.0)[j])
            for j in range(len(LM.test(1.0))):
                LM_full.append(LM.test(1.0)[j])

            try:
                LAD_val = LAD_LST.index(i)
            except:
                pass

            try:
                CX_val = CX_LST.index(i)
            except:
                pass

            try:
                LM_val = LM_LST.index(i)
            except:
                pass

            if (LAD_val == -1) and (CX_val == -1) and (LM_val == -1):
                pass
            else:
                min_val = 1000
                max_val = -1000
                if LAD_val != -1:
                    min_val = LAD_full[LAD_val][1]
                    max_val = LAD_full[LAD_val][2]

                if CX_val != -1:
                    cur_min = CX_full[CX_val][1]
                    cur_max = CX_full[CX_val][2]
                    if cur_min < min_val:
                        min_val = cur_min
                    if cur_max > max_val:
                        max_val = cur_max

                if LM_val != -1:
                    cur_min = CX_full[LM_val][1]
                    cur_max = CX_full[LM_val][2]
                    if cur_min < min_val:
                        min_val = cur_min
                    if cur_max > max_val:
                        max_val = cur_max

                setval = [i, min_val, max_val]
                if i < 81:
                    left_train.append(setval)
                else:
                    left_test.append(setval)

        return [left_train, left_test]

    def TRAIN_NUM(left_train):
        tot = 0
        for i in range(len(left_train)):
            diff = left_train[i][2] - left_train[i][1]
            tot += diff
        print(tot)
        
    def LEFT_TRAIN_ANNOTATIONS(left_train):
        data_annotations = open("3RCNN_Data_Annotations.csv", "r")
        data_reader = csv.reader(data_annotations)
        data_lst = list(data_reader)
        cluster_annotations = open("LEFT_CLUSTER_VALUES.csv", "a+")
        cluster_writer = csv.writer(cluster_annotations)
        for i in range(len(left_train)):
            rows = []
            for row in data_lst:
                if int(row[1]) == int(left_train[i][0]):
                    if int(row[2]) > int(left_train[i][1]) and int(row[2]) <= int(left_train[i][2]):
                        rows.append(row)
            for j in range(len(rows)):
                LXs = [rows[j][7], rows[j][11], rows[j][15]]
                TYs = [rows[j][8], rows[j][12], rows[j][16]]
                RXs = [rows[j][9], rows[j][13], rows[j][17]]
                BYs = [rows[j][10], rows[j][14], rows[j][18]]

                LX0 = [float(i) for i in LXs if float(i) != 0]
                TY0 = [float(i) for i in TYs if float(i) != 0]

                if len(LX0) == 0:
                    LX0.append(0)
                if len(TY0) == 0:
                    TY0.append(0)
                Xmin = min(LX0)
                Ymin = min(TY0)
                Xmax = max(RXs)
                Ymax = max(BYs)

                lst = ["NGCT", left_train[i][0], j + int(left_train[i][1]), Xmin, Ymin, Xmax, Ymax]
                cluster_writer.writerow(lst)

        data_annotations.close()
        cluster_annotations.close()

CLUSTER.TRAIN_NUM(CLUSTER.LEFT_TRAIN()[0])
CLUSTER.TRAIN_NUM(CLUSTER.LEFT_TRAIN()[1])
CLUSTER.LEFT_TRAIN_ANNOTATIONS(CLUSTER.LEFT_TRAIN()[0])
CLUSTER.LEFT_TRAIN_ANNOTATIONS(CLUSTER.LEFT_TRAIN()[1])
