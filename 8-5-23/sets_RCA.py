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
        mult = [[1, 0, 40], [3, 40, 60], [4, 20, 40], [5, 40, 60], [6, 40, 60], [10, 0, 40], [12, 50, 70], [13, 50, 70], [14, 50, 70], [15, 30, 50], [17, 25, 45], [18, 40, 60], [19, 70, 90], [22, 60, 80], [23, 0, 60], [24, 40, 60], [26, 45, 65], [27, 20, 60], [28, 50, 70], [29, 50, 70], [30, 58, 78], [31, 10, 30], [32, 40, 80], [33, 40, 80], [34, 40, 100], [35, 56, 76], [37, 0, 40], [39, 0, 40], [42, 0, 40], [43, 0, 20], [44, 20, 40], [45, 0, 60], [46, 40, 80], [47, 30, 70], [48, 25, 45], [49, 30, 70], [50, 60, 80], [51, 25, 45], [52, 60, 100], [53, 45, 65], [54, 0, 40], [55, 0, 40], [56, 16, 36], [57, 40, 80], [58, 53, 73], [59, 20, 60], [60, 0, 20], [62, 0, 40], [63, 40, 80], [64, 10, 30], [65, 20, 60], [66, 0, 20], [67, 60, 100], [68, 30, 70], [69, 0, 20], [70, 5, 45], [71, 0, 40], [73, 20, 40], [74, 0, 80], [75, 40, 80], [76, 40, 80], [78, 0, 60], [79, 40, 100], [80, 10, 50]]        
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
        testing = [[82, 0, 80], [83, 55, 75], [84, 50, 90], [85, 0, 60], [86, 0, 80], [87, 40, 100], [88, 0, 80], [89, 0, 80], [90, 35, 75], [91, 50, 90], [92, 30, 70], [93, 20, 100], [94, 40, 80], [95, 0, 20], [96, 0, 40], [97, 20, 100], [98, 30, 90], [99, 30, 90]]
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
