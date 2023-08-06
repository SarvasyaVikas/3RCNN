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
        mult = [[1, 59, 89], [3, 40, 50], [4, 25, 35], [5, 50, 60], [6, 45, 55], [7, 20, 30], [9, 12, 22], [10, 10, 30], [12, 55, 65], [13, 50, 70], [14, 55, 65], [15, 37, 47], [18, 45, 55], [19, 70, 90], [22, 63, 73], [23, 15, 35], [24, 55, 65], [26, 50, 70], [27, 40, 60], [28, 50, 70], [29, 55, 65], [30, 60, 80], [31, 15, 25], [32, 60, 70], [33, 53, 63], [34, 70, 100], [35, 60, 70], [37, 10, 20], [39, 5, 15], [42, 20, 30], [43, 5, 15], [46, 60, 80], [47, 50, 60], [48, 35, 45], [49, 45, 55], [50, 65, 75], [51, 30, 40], [52, 75, 85], [53, 50, 60], [54, 25, 35], [55, 0, 20], [56, 25, 35], [57, 50, 60], [58, 64, 74], [59, 39, 49], [62, 0, 20], [63, 45, 55], [64, 15, 25], [65, 32, 42], [67, 65, 75], [68, 50, 60], [69, 0, 10], [70, 17, 27], [71, 0, 10], [74, 15, 35], [75, 55, 65], [78, 0, 20], [79, 55, 65], [80, 20, 30]]
        
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
        testing = [[82, 40, 50], [83, 55, 75], [84, 60, 80], [85, 17, 27], [86, 35, 45], [87, 60, 80], [88, 35, 45], [89, 0, 20], [90, 40, 50], [91, 68, 78], [92, 40, 50], [93, 60, 70], [94, 60, 70], [95, 0, 10], [96, 5, 15], [97, 60, 70], [99, 45, 55]]
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
