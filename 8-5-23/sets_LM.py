import random

# add values for LAD, CX, LM
# different functions for each
# change function calls in 3RCNN_BASE_()_FIN.py and DRCNN-TESTING_()_FIN.py
# potential left cluster function
# also left cluster function within the run programs to generate it

class SETS:
    def __init__(self):
        pass
    
    def data(prob):
        mult = [[1, 5, 15], [3, 40, 50], [4, 25, 35], [9, 10, 20], [10, 10, 20], [11, 40, 50], [22, 60, 70], [24, 50, 60], [29, 50, 60], [30, 60, 70], [42, 15, 25], [43, 5, 15], [46, 60, 70], [47, 45, 55], [51, 30, 40], [57, 50, 60], [58, 60, 70], [59, 35, 45], [62, 0, 10], [65, 30, 40], [70, 15, 25], [74, 15, 25], [78, 0, 10], [79, 50, 60]]
        
        fin = []
        for i in range(len(mult)):
            if random.random() <= prob:
                fin.append(mult[i])
        
        return fin
    
    def imgs(mult):
        tot = 0
        for i in range(len(mult)):
            diff = mult[i][2] - mult[i][1]
            tot += diff
        print(tot)
    
    def test(prob):
        testing = [[82, 30, 50], [83, 55, 65], [84, 60, 70], [85, 15, 25], [86, 35, 45], [87, 60, 70], [89, 0, 10], [91, 62, 72], [93, 55, 65], [94, 55, 65], [96, 0, 10]]
        fin = []
        for i in range(len(testing)):
            if random.random() <= prob:
                fin.append(testing[i])
        
        return fin
    
    def validation(testing):
        tot = 0
        for i in range(len(testing)):
            diff = testing[i][2] - testing[i][1]
            tot += diff
        print(tot)

SETS.imgs(SETS.data(1.0))
SETS.validation(SETS.test(1.0))
