#! /usr/bin/env python                
import os, sys               
import numpy as np               
import matplotlib.pyplot as plt               
import obspy 
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
data, alphanorm, betanorm, alpha, beta, lambda_, corr_len = np.loadtxt("./l_curve.csv",
                                                                       unpack=True, usecols=(0,1,2,3,4,5,6),delimiter=",")
ax.plot_trisurf(alphanorm, betanorm, data, cmap=plt.cm.Spectral)
ax.set_xlabel(r"$||F(\mathbf{m})||_2^{2}$")
ax.set_ylabel(r"$||H(\mathbf{m})||_2^{2}$")
ax.set_zlabel(r"$||R(\mathbf{m})||_2^{2}$")
#plt.show()
plt.close()
# choose the proper beta as 8500
msk = beta == 500

alphanormn, datan, alphan = alphanorm[msk], data[msk], alpha[msk]
#alphanorms = np.linspace(alphanormn.min(), alphanormn.max(), 500)
#f2 = interpolate.interp1d(alphanormn, datan)
#datanorms = f2(alphanorms)
#f3 = interpolate.interp1d(alphanormn, alphan)
#alphas = f3(alphanorms)
alphanorms = alphanormn
datanorms = datan
alphas = alphan

rinv = np.zeros(len(datanorms)-3)
for idx, i in enumerate(np.arange(1,len(datanorms)-2)):
    pt1 = np.array([alphanorms[i-1], datanorms[i-1]])
    pt2 = np.array([alphanorms[i],datanorms[i]])
    pt3 = np.array([alphanorms[i+1],datanorms[i+1]])
    a = np.linalg.norm(pt3 - pt2)
    b = np.linalg.norm(pt3 - pt1)
    c = np.linalg.norm(pt2 - pt1)
    p = (a + b + c) / 2 
    r = a * b * c  / (4 * np.sqrt(p * (p-a) * (p-b) * (p-c)))
    rinv[idx] = 1.0 / r
maxdatanorm = datanorms[rinv.argmax()+1]
maxalphanorm = alphanorms[rinv.argmax()+1]
print(maxalphanorm, maxdatanorm)
print(alphas)
plt.scatter(alphanorms, datanorms, c=alphas)
ax = plt.gca()
#ax.loglog()
ax.set_xlabel(r"$||F(\mathbf{m})||_2^{2}$")
ax.set_ylabel(r"$||R(\mathbf{m})||_2^{2}$")
ax.set_title(r"L curve while $\beta=500$")
plt.colorbar()
plt.show()
