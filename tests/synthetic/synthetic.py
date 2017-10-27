#/usr/bin/env python

from JointInv.machinelearn.base import gen_disp_classifier, velomap
from JointInv.machinelearn.twostationgv import interstagv, spline_interpolate
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import os


from scipy.interpolate import interp1d

clf = gen_disp_classifier()
dispfile1 = "./sacmft/B00101Z00.sac.mft96.disp"
dispfile2 = "./sacmft/B00201Z00.sac.mft96.disp"
refdisp = "../../data/info/SREGN.ASC"

roughgva = velomap(dispinfo=dispfile1, refdisp=refdisp, trained_model=clf,
                   line_smooth_judge=True, digest_type="poly")
roughgvb = velomap(dispinfo=dispfile2, refdisp=refdisp, trained_model=clf,
                   line_smooth_judge=True, digest_type="poly")

intersta = interstagv(roughgva, roughgvb)
insta1, insta2, periods, velo1, velo2, velo3 = intersta.inter_gv_measurement()

# Isolate fundamental rayleigh waves
CPSPATH = "~/src/CPS/PROGRAMS.330/bin"

for judgement in [roughgva, roughgvb]:
    outpath = join("./isolation/", ".".join([judgement.id, "d"]))
    judgement.MFT962SURF96(outpath, CPSPATH)
    excfile = join(CPSPATH, "sacmat96")
    if judgement == roughgva:
        judgement.isowithsacmat96(surf96filepath=outpath,
                                  srcsacfile="./B00101Z00.sac", cpspath=CPSPATH)
    else:
        judgement.isowithsacmat96(surf96filepath=outpath,
                                  srcsacfile="./B00201Z00.sac", cpspath=CPSPATH)
    os.system("mv ./*.sac[rs] ./isolation/")


permin, permax = min(insta1.min(), insta2.min()), max(
    insta1.max(), insta2.max())

# import synthetic
period, synrc, synru = np.loadtxt("./SREGN.ASC", usecols=(2, 4, 5),
                                  unpack=True, skiprows=1)
# import measured phase velocity dispersion
velocper, velocms = np.loadtxt("./isolation/rayl.dsp", usecols=(5, 6),
                               unpack=True)
plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(insta1, velo1, "o", label="Measured U of {}".format(roughgva.id))
plt.plot(insta2, velo2, "o", label="Measured U of {}".format(roughgvb.id))
plt.plot(period, synru, label="Synthetic U")
plt.xlabel("Period [s]")
plt.ylabel("Velocity [km/s]")
plt.title("{}-{}".format(roughgva.id, roughgvb.id))
plt.xlim(permin, permax)
plt.legend()


# derive group velocity from measured phase velocity dispersion
plt.subplot(1, 2, 2)
plt.plot(period, synrc, label="Synthetic C")
plt.plot(velocper, velocms, "o", label="Measured C")
plt.xlabel("Period [s]")
plt.ylabel("Velocity [km/s]")
plt.title("{}-{}".format(roughgva.id, roughgvb.id))
plt.xlim(permin, permax)
plt.legend()
plt.show()
