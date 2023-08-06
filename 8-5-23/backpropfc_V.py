import numpy as np
from network_FC_CF_IV import network
import csv

class BPFC:
    def __init__(self):
        pass

    def FCBP(layer, inputs, act, alpha, w = 0):
        orig = layer.copy()
        for i in range(len(inputs)):
            for j in range(len(act)):
                w = layer[i][0][j]
                b = layer[i][1]
                x = inputs[i]
                y = act[j]
                yi = y / len(inputs)
                ei = (w * x) + b - yi

                dCdw = 2 * float(ei) * float(x)
                delta = dCdw * alpha

                layer[i][0][j] -= delta

        return (orig, layer)

    def sigmoidINVERSE(x):
        recip = 1 / x
        inside = recip - 1
        if inside == 0:
            inside = 1
        ln = np.log(abs(inside))
        y = -1 * ln
        return y

    def tanhINVERSE(x):
        plus1 = x + 1
        frac = 2 / plus1
        inside = frac - 1
        if inside == 0:
            inside = 1
        minus_two_y = np.log(abs(inside))
        y = -0.5 * minus_two_y
        return y

    def mrelu(val):
        inv = val
        if val < 0:
            inv = val * 10
        return inv

    def NMBP(layer, outputs, activation = 0):
        avgs = []
        for i in range(len(layer)):
            avgs.append(0)
        avg_output = []
        for i in range(len(outputs)):
            deactiv = outputs[i] if outputs[i] > 0 else (10 * outputs[i])
            if activation == 1:
                deactiv = BPFC.mrelu(outputs[i])
            avg_output.append(deactiv / len(layer))

        for i in range(len(layer)):
            for j in range(len(layer[i][0])):
                wx = avg_output[j] - layer[i][1]
                x = wx / layer[i][0][j]
                avgs[i] += x

        res = []
        for i in range(len(avgs)):
            res.append(avgs[i] / len(outputs))

        return res

    def mse(vals1, vals2):
        tot = 0
        for i in range(len(vals1)):
            diff = vals1[i] - vals2[i]
            sq = diff ** 2
            tot += sq
        res = tot / len(vals1)
        return res

    def INIT_PASS(networkFC, init_inputs):
        (fc5, wr5) = network.forward_pass(networkFC[1][0], init_inputs, networkFC[1][1], 0.1, wr = True)
        (fc6, wr6) = network.forward_pass(networkFC[1][1], fc5, networkFC[2], 0, wr = True)
                    
        return (fc5, fc6, wr5, wr6)

    def REG(net):
        for i in range(len(net[1])):
            for j in range(len(net[1][i])):
                for k in range(len(net[1][i][j][0])):
                    if abs(net[1][i][j][0][k]) > 10:
                        sign = -1 if net[1][i][j][0][k] < 0 else 1
                        net[1][i][j][0][k] = np.sqrt(abs(net[1][i][j][0][k])) * sign

        for i in range(len(net[2])):
            for j in range(len(net[2][i][0])):
                if abs(net[2][i][0][j]) > 10:
                    sign = -1 if net[2][i][0][j] < 0 else 1
                    net[2][i][0][j] = np.sqrt(abs(net[2][i][0][j])) * sign

        return net
 
    def FULL(networkFC, init_inputs, fin_outputs, alpha, coords, updated_inputs, overfitted_inputs, prev, loss):
        (fc5, wr5) = network.forward_pass(networkFC[1][0], init_inputs, networkFC[1][1], 0.1, wr = True)
        (fc6, wr6) = network.forward_pass(networkFC[1][1], fc5, networkFC[2], 0, wr = True)

        w5a = sum(wr5) / len(wr5)
        w6a = sum(wr6) / len(wr6)
        # fin outputs is the same as coords
        bp6 = BPFC.NMBP(networkFC[1][1], coords, 0)
        bp5 = BPFC.NMBP(networkFC[1][0], bp6, 0.1)
        filter_loss = bp5

        (_, oFC10) = BPFC.FCBP(networkFC[1][0], init_inputs, bp6, alpha, w = w5a)
        (_, oFC11) = BPFC.FCBP(networkFC[1][1], fc5, fin_outputs, alpha, w = w6a)
        
        (nFC10, networkFC[1][0]) = BPFC.FCBP(networkFC[1][0], init_inputs, bp6, alpha)
        (nFC11, networkFC[1][1]) = BPFC.FCBP(networkFC[1][1], fin_outputs, bp7, alpha)

        (ow5, o5) = network.forward_pass(networkFC[1][0], updated_inputs, networkFC[1][1], 0.1, wr = True)
        (ow6, o6) = network.forward_pass(networkFC[1][1], o5, networkFC[2], 0, wr = True)

        (new5, r5) = network.forward_pass(networkFC[1][0], overfitted_inputs, networkFC[1][1], 0.1, wr = True)
        (new6, r6) = network.forward_pass(networkFC[1][1], new5, networkFC[2], 0, wr = True)

        pred_list = []
        for i in range(4):
            pred_list.append(fc6[0])
            pred_list.append(new6[0])
            pred_list.append(ow6[0])

        vals = open("vals.csv", "a+")
        (csv.writer(vals)).writerow(pred_list)
        vals.close()

        pred1 = [0, 0, 0, 0]
        if fc7[1] > fc7[0]:
            pred1 = fc6
        pred2 = [0, 0, 0, 0]
        if new7[1] > new7[0]:
            pred2 = new6
        pred3 = [0, 0, 0, 0]
        if ow7[1] > ow7[0]:
            pred3 = ow6

        se1 = BPFC.mse(coords, pred1)
        se2 = BPFC.mse(coords, pred2)
        se3 = BPFC.mse(coords, pred3)

        use1 = se1 * (2 ** 14)
        use2 = se2 * (2 ** 14)
        use3 = se3 * (2 ** 14)

        tse1 = use1 + sum(wr5) + sum(wr6) + sum(wr7)
        tse2 = use2 + sum(r5) + sum(r6) + sum(r7)
        tse3 = use3 + sum(o5) + sum(o6) + sum(o7)

        pred_fin = ow6
        fin_network = networkFC.copy()
        code = 3
        if min(tse1, tse2, tse3) == tse1 and tse1 != 0:
            fin_network[1][0] = nFC10
            fin_network[1][1] = nFC11
            fin_network[2] = nFC2
            pred_fin = fc6
            code = 1
        elif min(tse1, tse2, tse3) == tse2:
            fin_network[1][0] = oFC10
            fin_network[1][1] = oFC11
            fin_network[2] = oFC2
            pred_fin = new6
            code = 2
        else:
            pass

        filter_matrix = []
        for k in range(8):   
            filter_row = []
            for j in range(8):
                place = (8 * k) + j
                val = filter_loss[place]
                filter_row.append(val)
            filter_matrix.append(filter_row)
            
        rho = network.direction(prev, loss)

        return (fin_network, min(tse1, tse2, tse3), code, pred_fin)
