import math
import numpy as np
import matplotlib.pyplot as plt



#CALCULATION OFV WEIGHTS

# Use of np.array() to define an Input Vector
V = np.array([.323, .432])
print("The Vector A as Inputs : ", V)

# defining Weight Vector
VV = np.array([[.3, .66, ], [.27, .32]])
W = np.array([.7, .3, ])

print("\nThe Vector B as Weights: ", VV)

# defining a neural network for predicting an
# output value


def neural_network(inputs, weights):
    wT = np.transpose(weights)
    elpro = wT.dot(inputs)

    # Tangent Hyperbolic Function for Decision Making
    out = np.tanh(elpro)
    return out


outputi = neural_network(V, VV)

# printing the expected output
print("Expected Value of Hidden Layer Units: ", outputi)

outputj = neural_network(outputi, W)

# printing the expected output
print("Expected Output of the with one hidden layer : ", outputj)


#HERMITE POLYNOMIAL 
def HER(x, n):
    if n == 0:
        return 1.0 + 0.0*x
    elif n == 1:
        return 2.0*x
    else:
        return 2.0*x*HER(x, n-1) - 2.0*(n-1)*HER(x, n-2)


xvals = np.linspace(-np.pi, np.pi, 1000)
for N in np.arange(0, 7, 1):
    sol = HER(xvals, N)
    plt.plot(xvals, sol, label="n = " + str(N))
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
print(sol)
plt.grid()
plt.legend()
plt.show()
