import numpy as np
from network_FC_CF_IV import network

class BPFC:
    def __init__(self):
        pass

    def FCBP(layer, inputs, act, alpha):
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

    def NMBP(layer, outputs, activation = 0):
        avgs = []
        for i in range(len(layer)):
            avgs.append(0)
        avg_output = []
        for i in range(len(outputs)):
            deactiv = BPFC.sigmoidINVERSE(outputs[i])
            if activation == 0:
                deactiv = BPFC.tanhINVERSE(outputs[i])
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

    def FULL(networkFC, init_inputs, fin_outputs, alpha, coords):
        try:
            fc5 = network.forward_pass(networkFC[1][0], init_inputs, networkFC[1][1])
        except:
            print(len(init_inputs))
            print(len(networkFC[1][0]))
        fc6 = network.forward_pass(networkFC[1][1], fc5, networkFC[2], 1)
        fc7 = network.forward_pass(networkFC[2], fc6, [0, 0])

        bp7 = BPFC.NMBP(networkFC[2], fin_outputs)
        bp6 = BPFC.NMBP(networkFC[1][1], bp7, 1)

        (nFC10, networkFC[1][0]) = BPFC.FCBP(networkFC[1][0], init_inputs, bp6, alpha)
        (nFC11, networkFC[1][1]) = BPFC.FCBP(networkFC[1][1], fc5, bp7, alpha)
        (nFC2, networkFC[2]) = BPFC.FCBP(networkFC[2], fc6, fin_outputs, alpha)
        
        new5 = network.forward_pass(networkFC[1][0], init_inputs, networkFC[1][1])
        new6 = network.forward_pass(networkFC[1][1], new5, networkFC[2], 1)
        new7 = network.forward_pass(networkFC[2], new6, [0, 0])

        pred1 = [0, 0, 0, 0]
        if fc7[1] > fc7[0]:
            pred1 = fc6
        pred2 = [0, 0, 0, 0]
        if new7[1] > new7[0]:
            pred2 = new6

        mse1 = BPFC.mse(coords, pred1)
        mse2 = BPFC.mse(coords, pred2)

        fin_network = networkFC.copy()
        if mse1 < mse2:
            fin_network[1][0] = nFC10
            fin_network[1][1] = nFC11
            fin_network[2] = nFC2
        elif mse2 < mse1:
            pass
        else:
            diff1 = abs(fin_outputs[0] - fc7[0])
            diff2 = abs(fin_outputs[0] - new7[0])
            if diff1 < diff2:
                fin_network[1][0] = nFC10
                fin_network[1][1] = nFC11
                fin_network[2] = nFC2

        return (fin_network, min(mse1, mse2))
