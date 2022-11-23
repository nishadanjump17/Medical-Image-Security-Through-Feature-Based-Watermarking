# Linear Algebra and Neural Network
# Linear Algebra Learning Sequence

import numpy as np

# Use of np.array() to define an Input Vector
V = np.array([.323,.432])
print("The Vector A as Inputs : ",V)

# defining Weight Vector
VV = np.array([[.3,.66,],[.27,.32]])
W = np.array([.7,.3,])

print("\nThe Vector B as Weights: ",VV)

# defining a neural network for predicting an 
# output value
def neural_network(inputs, weights):
    wT = np.transpose(weights)
    elpro = wT.dot(inputs)
    
    # Tangent Hyperbolic Function for Decision Making
    out = np.tanh(elpro)
    return out

outputi = neural_network(V,VV)

# printing the expected output
print("Expected Value of Hidden Layer Units: ", outputi)

outputj = neural_network(outputi,W)

# printing the expected output
print("Expected Output of the with one hidden layer : ", outputj)
