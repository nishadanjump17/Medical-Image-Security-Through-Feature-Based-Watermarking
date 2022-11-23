from math import exp
from random import seed
from random import random
import numpy as np
import cv2
# Initialize a network


##def initialize_network(n_inputs, n_hidden, n_outputs):
##	network = list()
##	hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]}
##                 for i in range(n_hidden)]
##	network.append(hidden_layer)
##	output_layer = [{'weights': [random() for i in range(n_hidden + 1)]}
##                 for i in range(n_outputs)]
##	network.append(output_layer)
##	return network

#compute NN with hidden and output layers
def initialize_network(x):
	network = list()
	hl=list()
	ol=list()
	for i in range(len(x)):
		hl.append(random()*x[i])
	ol=HER( np.array(x),3).tolist()
	

		    
	hidden_layer = [{'weights': hl}]
	network.append(hidden_layer)
	output_layer = [{'weights': ol}]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input

##def hermite_polynomial(j,x):
##        if j==0:
##                return 1
##        elif j==1:
##                return 2*x
##        else:
##                h=list()
##                h.append(1)
##                h.append(2*x)
##                i=2
##                while i<=j:
##                        h.append(2*x*h[i-1]-2*(i-1)*h[i-2])
##                        i+=1
##                return h[i-1]
def HER(x,n):
    if n==0:
        return 1.0 + 0.0*x
    elif n==1:
        return 2.0*x
    else:
        return 2.0*x*HER(x,n-1) -2.0*(n-1)*HER(x,n-2)

def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation


def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output


def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output


def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons


def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(neuron['output'] - expected[j])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error


def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] -= l_rate * neuron['delta']

# Train a network for a fixed number of epochs


def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i]) **
			                 2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


#Generate chaotic sequence
def chaotic_sequence(mu,n,initial):
        i=1
        seq=list()
        seq.append(initial)
        #print(type(seq))
        while i<=n:
                #print(seq[i-1])
                seq.append(mu*seq[i-1]*(1-seq[i-1]))
                i+=1
        return seq

def scramble(img,x):
        rows,cols = img.shape
        
        h=HER( np.array(x),3)
        oseq = h.tolist()
        seq = h.tolist()
        seq.sort()
        
        ilist=list()

        for i in range(rows):
                for j in range(cols):
                        ilist.append(img[i,j])

        ind=0
        eindices=list()
        while ind<len(seq):
                
                eind=oseq.index(seq[ind])
                #print(eind)
                exist_count = eindices.count(ind)
                if exist_count>0:
                        #print(eind)
                        ind+=1
                        continue
                eindices.append(eind)
                t=ilist[eind]
                ilist[eind]=ilist[ind]
                ilist[ind]=t
                ind+=1
        n=0
        #print(eindices)
        for i in range(rows):
                for j in range(cols):
                        img[i,j]=ilist[n]
                        n+=1
        return img
# Test training backprop algorithm
##seed(1)
##dataset = [[2.7810836, 2.550537003, 0],
##           [1.465489372, 2.362125076, 0],
##           [3.396561688, 4.400293529, 0],
##           [1.38807019, 1.850220317, 0],
##           [3.06407232, 3.005305973, 0],
##           [7.627531214, 2.759262235, 1],
##           [5.332441248, 2.088626775, 1],
##           [6.922596716, 1.77106367, 1],
##           [8.675418651, -0.242068655, 1],
##           [7.673756466, 3.508563011, 1]]
##n_inputs = len(dataset[0]) - 1
##n_outputs = len(set([row[-1] for row in dataset]))
##network = initialize_network(n_inputs, 2, n_outputs)
##train_network(network, dataset, 0.5, 20, n_outputs)
##for layer in network:
##	print(layer)


img = cv2.imread('wm.jpg',0)
rows,cols = img.shape
x=chaotic_sequence(3.5699456,rows*cols-1,0.66)
scramble_img=scramble(img,x)

cv2.imwrite('scrambledimage.jpg', scramble_img)

#cv2.imshow('Chaos', scramble_img)



network=initialize_network(x)

##for layer in network:
##        new_inputs = []
##        for neuron in layer:
##                activation = activate(neuron['weights'], x)
##                print(activation)
##                neuron['output'] = transfer(activation)
##                new_inputs.append(neuron['output'])
##                inputs = new_inputs
##print(new_inputs)

