#/usr/bin/env python

from JointInv.machinelearn.base import gen_disp_classifier, velomap
from JointInv.machinelearn.twostationgv import interstagv, spline_interpolate
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d
clf = gen_disp_classifier()
#dispfile1 = "./data/rawtraces/20150106220913/XJ.HTB.00.BHZ.mft96.disp"
#dispfile2 = "./data/rawtraces/20150106220913/XJ.HTTZ0.00.BHZ.mft96.disp"
dispfile1 = "./synthetic/0501.mft96.disp"
dispfile2 = "./synthetic/0101.mft96.disp"
refdisp = "../data/info/SREGN.ASC"

roughgvb = velomap(dispinfo=dispfile2, refdisp=refdisp, trained_model=clf,
                   line_smooth_judge=True, digest_type="poly")
roughgva = velomap(dispinfo=dispfile1, refdisp=refdisp, trained_model=clf,
                   line_smooth_judge=True, digest_type="poly")

phdispper, phdispvelo = np.loadtxt("./synthetic/rayl.dsp", usecols=(5,6),
                                   unpack=True)

intersta = interstagv(roughgva, roughgvb)
insta1, insta2, periods, velo1, velo2, velo3 = intersta.inter_gv_measurement()


permin, permax = phdispper.min(), phdispper.max()
wspline = np.arange(permin, permax)

# import synthetic
period, synrc = np.loadtxt("./synthetic/simple.rc", usecols=(0,1), unpack=True)
synru = np.loadtxt("./synthetic/simple.ru", usecols=1)
pru1, mru1 = np.loadtxt("./synthetic/GRN21Z0501.dsp", usecols=(4,5), unpack=True)
pru2, mru2 = np.loadtxt("./synthetic/GRN21Z0101.dsp", usecols=(4,5), unpack=True)

plt.figure(1)
plt.subplot(1,2,1)
plt.plot(insta1, velo1, "+", label="Measured U of {}".format(roughgva.id))
plt.plot(insta2, velo2, "+", label="Measured U of {}".format(roughgvb.id))
plt.plot(pru1, mru1, "^", label="Graph. Measured U")
plt.plot(pru2, mru2, "^", label="Graph. Measured U")
# plt.plot(periods, velo3, "o", label="Interstation Measured U")
plt.plot(period, synru, label="Synthetic U")
plt.xlabel("Period [s]")
plt.ylabel("Velocity [km/s]")
plt.title("{}-{}".format(roughgva.id, roughgvb.id))
plt.xlim(permin, permax)
plt.legend()

f2 = interp1d(phdispper, phdispvelo, kind="cubic")
c = f2(wspline)
dcdt = np.gradient(c, 1, edge_order=2)
u = c / (1  + wspline * dcdt/ c) 

plt.subplot(1,2,2)
plt.plot(phdispper, phdispvelo, "o", label="Interstation Measured C")
plt.plot(period, synrc, label="Synthetic C")
plt.plot(wspline, u, '^', label="Derived U from C")
plt.plot(period, synru, label="Synthetic U")
#plt.plot(np.arange(permin, permax), u, '^', label="Derived U from C")
#plt.plot(np.arange(permin, permax), c, 'o', label="Interpolated C")
#plt.plot(periods, velo3, "+", label="Interstation Measured U")

plt.xlabel("Period [s]")
plt.ylabel("Velocity [km/s]")
plt.title("{}-{}".format(roughgva.id, roughgvb.id))
plt.xlim(permin, permax)
plt.legend()
plt.show()
#plt.savefig("CBC-HEF-self-consistency-test.png")
