import numpy as np
import cv2
import random
import math

def generate_layer(prev, nodes):
	layer = []
	for i in range(prev):
		weights = []
		for j in range(nodes):
			weight = random.uniform(-1, 1)
			weights.append(weight)
		bias = random.uniform(-1, 1)
		layer.append([weights, bias])
	return layer

def elu(x):
	if x > 0:
		return x
	else:
		val = ((math.e) ** x) - 1
		return val

def forward_pass(layer, vals, layerNodes, forwardNodes):
	new = []
	for i in forwardNodes:
		tot = 0
		for j in layerNodes:
			added = (layer[j][0][i] * vals[j]) + layer[j][1]
			tot += added
		activ = elu(tot)
		new.append(tot)
	return new

def mse(pred, act):
	diff = abs(pred - act)
	squared = diff ** 2
	return squared

def backpropagation(layer, loss, forward, alpha, old):
	loss_vals = []
	for i in range(layer):
		loss_val = 0
		for j in range(forward):
			val = (loss[j] * layer[i][0][j]) + forward[j][1]
			change = val * old[i]
			layer[i][0][j] -= (change * alpha)
			loss_val += change
		loss_vals.append(loss_val)
	return (layer, loss_vals)

def dropout(layer, frac):
	nodes = []
	for i in range(layer):
		val = random.uniform(0, 1)
		if val > frac:
			nodes.append([i, layer[i]])
	return nodes
	
layerInput = generate_layer(4, 16)
layer1 = generate_layer(16, 32)
layer2 = generate_layer(32, 64)
layer3 = generate_layer(64, 128)
layer4 = generate_layer(128, 32)
layer5 = generate_layer(32, 8)
layer6 = generate_layer(8, 4)
layerOutput = generate_layer(4, 1)

inputs = [] # put inputs here
actuals = [] # put actual values here

results = []

for _ in range(epochs):
	for i in range(len(inputs)):
		new1 = forward_pass(layerInput, inputs[i], dropout(layer1, 0.25))
		new2 = forward_pass(dropout(layer1, 0.25), new1, dropout(layer2, 0.25))
		new3 = forward_pass(dropout(layer2, 0.25), new2, dropout(layer3, 0.25))
		new4 = forward_pass(dropout(layer3, 0.25), new3, dropout(layer4, 0.25))
		new5 = forward_pass(dropout(layer4, 0.25), new4, dropout(layer5, 0.25))
		new6 = forward_pass(dropout(layer5, 0.25), new5, dropout(layer6, 0.25))
		newOUT = forward_pass(dropout(layer6, 0.25), new6, layerOutput)
		
		newLOSS = mse(newOUT[0], actuals[i])
		results.append(newLOSS)
		
		(layer6, loss6) = backpropagation(layer6, newLOSS, layerOutput, alpha, new6)
		(layer5, loss5) = backpropagation(layer5, loss6, layer6, alpha, new5)
		(layer4, loss4) = backpropagation(layer4, loss5, layer5, alpha, new4)
		(layer3, loss3) = backpropagation(layer3, loss4, layer4, alpha, new3)
		(layer2, loss2) = backpropagation(layer2, loss3, layer3, alpha, new2)
		(layer1, loss1) = backpropagation(layer1, loss2, layer2, alpha, new1)
