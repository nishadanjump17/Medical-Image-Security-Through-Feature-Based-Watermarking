import math
import numpy as np
import matplotlib.pyplot as plt


def HER(x,n):
    if n==0:
        return 1.0 + 0.0*x
    elif n==1:
        return 2.0*x
    else:
        return 2.0*x*HER(x,n-1) -2.0*(n-1)*HER(x,n-2)

xvals = np.linspace(0,1,1000)
print(xvals)
sol = HER(xvals,3)
print(sol[1:1000])
##for N in np.arange(0,7,1):
##    sol = HER(xvals,N)
##    plt.plot(xvals,sol,label = "n = " + str(N))
##plt.xticks(fontsize=14,fontweight="bold")
##plt.yticks(fontsize=14,fontweight="bold")
##plt.grid()
##plt.legend()
##plt.show()
