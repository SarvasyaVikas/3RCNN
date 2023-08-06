import csv

class sets_LAD_SCL:
    def __init__(self):
        pass
    
    def data(ex = True):
        mult = []

        revals = open("REVALS_FC_2.csv", "r")
        revalsreader = csv.reader(revals)

        scans = []
        count = []
        for row in revalsreader:
            if float(row[1]) not in scans:
                scans.append(int(row[1]))
                count.append([int(row[2]), int(row[2])])
            else:
                count[-1][1] = int(row[2])

        for i in range(len(scans)):
            lst = [scans[i], count[i][0], count[i][1] + 1]
            if ex:
                if scans[i] not in [6, 33, 64, 26, 57, 59, 94, 24, 30, 31, 37, 38, 39, 42, 43, 50, 63, 65, 70, 71, 83, 84, 88, 90, 95, 99]:
                    mult.append(lst)
            else:
                mult.append(lst)
        
        print(mult)
        return mult

sets_LAD_SCL.data()
