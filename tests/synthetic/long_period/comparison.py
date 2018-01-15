#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def one_test(dispfile="./B1mft96.disp", dist=2500):
    mperiod, mgv = np.loadtxt(dispfile, usecols=(16,5),
                              unpack=True)
    arr1 = dist / mgv

    synperd, syngv = np.loadtxt("./SREGN.ASC", usecols=(2,5), unpack=True,
                               skiprows=1)
    arr2 = dist / syngv

    msk = (mperiod < 100) * (mperiod > 25)
    f2 = interp1d(synperd, arr2, kind="cubic")
    splinearrs = f2(mperiod[msk])

    cmnper = np.array(mperiod[msk])
    arrdiff = np.array(splinearrs - arr1[msk])
    return mperiod, arr1, synperd, arr2, cmnper, arrdiff

# plot comparison
mperiod, arr1, synperd, arr2B1, cmnperB1, arrdiffB1 = one_test()
mperiod, arr1, synperd, arr2B2, cmnperB2, arrdiffB2 = one_test("./B2mft96.disp", 2850)
mperiod, arr1, synperd, arr2B3, cmnperB3, arrdiffB3 = one_test("./B3mft96.disp", 5000)
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12,6))

axes[0].plot(mperiod, arr1, "+", label="Measured")
axes[0].plot(synperd, arr2B3, label="Synthetic")
axes[0].set_xlabel("Period [s]")
axes[0].set_ylabel(r"$Dist\ /\ V_{g}$ [s]")
axes[0].set_title("Arrival calculation")
axes[0].legend()
axes[0].set_xlim(25,100)


axes[1].plot(cmnperB1, arrdiffB1, "+", label="dist=2500 km")
axes[1].plot(cmnperB2, arrdiffB2, "+", label="dist=2850 km")
axes[1].plot(cmnperB3, arrdiffB3, "+", label="dist=5000 km")
axes[1].set_xlabel("Period [s]")
axes[1].set_ylabel(r"$t_{synthetic}-t_{measured} $   [s]")
axes[1].legend()
axes[1].set_xlim(25,100)
axes[1].set_ylim(-40,25)
axes[1].set_title("Distance-dependent arrival distance")
axes[1].legend()
plt.savefig("Distance-dependent.png")
plt.close()

# investigate influence of alpha
mperiod, arr1, synperd, arr2100, cmnper100, arrdiff100 = one_test(
    "./alpha100mft96.disp", 2500)
mperiod, arr1, synperd, arr2200, cmnper200, arrdiff200 = one_test(
    "./alpha200mft96.disp", 2500)
mperiod, arr1, synperd, arr2500, cmnper500, arrdiff500 = one_test(
    "./alpha500mft96.disp", 2500)
mperiod, arr1, synperd, arr21000, cmnper1000, arrdiff1000 = one_test(
    "./alpha1000mft96.disp", 2500)
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6,6))
ax.plot(cmnper100, arrdiff100, "+", label="alpha=100")
ax.plot(cmnper200, arrdiff200, "+", label="alpha=200")
ax.plot(cmnper500, arrdiff500, "+", label="alpha=500")
ax.plot(cmnper1000, arrdiff1000, "+", label="alpha=1000")
ax.legend()
ax.set_xlabel("Period [s]")
ax.set_ylabel("Arrival Difference [s]")
ax.set_title("Influence of alpha")
ax.set_ylim(-20,5)
plt.savefig("Alpha_nondependent.png")
plt.close()

