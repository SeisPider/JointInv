#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys

mperiod, mgv = np.loadtxt("./mft96.disp", usecols=(16,5),
                          unpack=True)
synperd, syngv = np.loadtxt("./SREGN.ASC", usecols=(2,5), unpack=True,
                           skiprows=1)

# plot comparison
plt.plot(mperiod, mgv, "+", label="Measured")
plt.plot(synperd, syngv, label="Synthetic")
plt.xlabel("Period [s]")
plt.ylabel("Group Velocity [km/s]")
plt.title("Long Period Correction")
plt.legend()
plt.xlim(25,100)
plt.show()
