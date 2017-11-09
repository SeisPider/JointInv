#! /usr/bin/env python                
import numpy as np               
import matplotlib.pyplot as plt 
from mpmath import * 

def f3(c):
    # set params.
    alpha1, alpha2, beta1, beta2 = 6, 8, 3.5, 4.7
    rho1, rho2 = 2.7, 3.3
    H = 40

    # calculate
    mu2overmu1 = (beta2 / beta1)**2
    omega = 2 * np.pi / PERIOD

    return tan(omega * H * sqrt((1/beta1**2)-(1/c**2))) - \
        mu2overmu1 * sqrt((1/c**2)-(1/beta2**2)) \
        /sqrt((1/beta1**2)-(1/c**2))
results = []
for per in np.arange(25,100,0.1):
    PERIOD = per
    if per == 25:
        result = findroot(f3, 3.5)
        results.append(result)
    else:
        result = findroot(f3, results[-1])
        results.append(result)
c = np.array([float(x.real) for x in results])
dcdt = np.gradient(c, 0.1, edge_order=2)
u = 
