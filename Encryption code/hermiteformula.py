import numpy as np
from sympy import N

r = 3.8

N=10

x = .3+np.zeros(N)
for n in range(N-1):
    x[n+1] = r*x[n]*(1-x[n])
    print(n+1)
    


