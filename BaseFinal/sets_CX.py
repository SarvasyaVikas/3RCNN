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
        mult = [[1, 10, 20], [3, 45, 55], [4, 25, 35], [5, 50, 60], [7, 20, 30], [10, 10, 30], [11, 40, 60], [12, 55, 65], [13, 50, 60], [14, 50, 60], [15, 35, 55], [17, 25, 35], [18, 45, 55], [19, 70, 80], [22, 60, 70], [23, 15, 35], [24, 55, 65], [26, 55, 65], [27, 45, 55], [28, 55, 65], [29, 55, 65], [30, 58, 78], [31, 20, 30], [32, 60, 70], [33, 55, 65], [34, 75, 95], [35, 65, 75], [39, 5, 15], [42, 20, 30], [43, 5, 15], [46, 65, 75], [47, 45, 55], [48, 35, 45], [49, 47, 57], [50, 65, 75], [51, 30, 40], [52, 80, 90], [53, 45, 65], [54, 25, 35], [55, 5, 15], [56, 25, 35], [57, 50, 60], [58, 65, 75], [62, 0, 10], [63, 50, 60], [64, 15, 25], [65, 35, 45], [66, 0, 20], [67, 65, 75], [68, 50, 60], [69, 10, 20], [70, 15, 35], [71, 0, 20], [74, 15, 35], [75, 55, 65], [78, 0, 20], [79, 55, 65], [80, 20, 30]]
        
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
        testing = [[82, 40, 50], [83, 60, 80], [84, 60, 80], [85, 15, 25], [86, 35, 45], [87, 60, 70], [88, 37, 47], [89, 5, 25], [91, 70, 80], [93, 60, 70], [95, 5, 15], [96, 5, 15], [97, 60, 70], [98, 60, 70], [99, 50, 60]]
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
