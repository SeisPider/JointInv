#/usr/bin/env python

from JointInv.machinelearn.base import gen_disp_classifier, velomap
from JointInv.machinelearn.twostationgv import interstagv, spline_interpolate
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import os


from scipy.interpolate import interp1d

# pre-defined: interested period region
PERMIN, PERMAX = 25, 100


# 1. Train classifier and import extreme value files
clf = gen_disp_classifier()
dispfile1 = "./sacmft/XJ.HTB.00.BHZ.mft96.disp"
dispfile2 = "./sacmft/XJ.HTTZ0.00.BHZ.mft96.disp"
refdisp = "../../data/info/SREGN.ASC"

# 2. Initialize velomap class and classify this velocity map
roughgva = velomap(dispinfo=dispfile1, refdisp=refdisp, trained_model=clf,
                   line_smooth_judge=True, digest_type="poly")
roughgvb = velomap(dispinfo=dispfile2, refdisp=refdisp, trained_model=clf,
                   line_smooth_judge=True, digest_type="poly")

# 3. visualize all possible points 
def obtain_points(velomap, mode="all"):
    """Give back all possible dispersion points
    """
    if mode == "all":
        insta = np.array([x.instaper for x in velomap.records])
        gvelo = np.array([x.velo for x in velomap.records])
        veloerr = np.array([x.veloerr for x in velomap.records])
    if mode == "1st":
        insta = np.array([x.instaper for x in velomap.midrec])
        gvelo = np.array([x.velo for x in velomap.midrec])
        veloerr = np.array([x.veloerr for x in velomap.midrec])
    if mode == "2nd":
        insta = np.array([x.instaper for x in velomap.disprec])
        gvelo = np.array([x.velo for x in velomap.disprec])
        veloerr = np.array([x.veloerr for x in velomap.disprec])
    if mode == "reference":
        insta = velomap.refperiods
        gvelo = velomap.refgv
        veloerr = None
    return insta, gvelo, veloerr

for idx, velomap in enumerate([roughgva, roughgvb]):
    # import data
    insta, gvelo, veloerr = obtain_points(velomap)
    plt.errorbar(insta, gvelo, fmt="o", yerr=veloerr)
    plt.xlabel("Instantaneous Period [s]")
    plt.ylabel("Group Velocity [km/s]")
    plt.title("Extreme Value of {}".format(velomap.id))
    plt.xlim(PERMIN, PERMAX)
    plt.savefig("{}.ext.val.png".format(velomap.id))
    plt.close()

# 4. Visualize machine-determined seletion
for idx, velomap in enumerate([roughgva, roughgvb]):
    # import all extreme value
    insta, gvelo, veloerr = obtain_points(velomap)
    plt.errorbar(insta, gvelo, fmt="o", yerr=veloerr, 
                 label="All possible points")
    
    # import machine-determined result
    insta, gvelo, veloerr = obtain_points(velomap, mode="1st")
    plt.errorbar(insta, gvelo, fmt="o", yerr=veloerr,
                 label="Machine selected points")
    
    # depict reference dispersion
    refperiod, refvelo, _ = obtain_points(velomap, mode="reference")
    plt.plot(refperiod, refvelo, label="Reference dispersion [ak135]")

    plt.xlabel("Instantaneous Period [s]")
    plt.ylabel("Group Velocity [km/s]")
    plt.title("Machine Determination of {}".format(velomap.id))
    plt.xlim(PERMIN, PERMAX)
    plt.legend()
    plt.savefig("{}.mac.det.png".format(velomap.id))
    plt.close()

# 5. Isolate fundamental rayleigh waves
CPSPATH = "~/src/CPS/PROGRAMS.330/bin"

for judgement in [roughgva, roughgvb]:
    outpath = join("./isolation/", ".".join([judgement.id, "d"]))
    judgement.MFT962SURF96(outpath, CPSPATH)
    excfile = join(CPSPATH, "sacmat96")
    judgement.isowithsacmat96(surf96filepath=outpath, cpspath=CPSPATH,
                              srcpath="./data/")
    os.system("mv ./data/*.SAC[rs] ./isolation/")

# 6. goto folder isolation and use subroutine do_pom

# 7. Visualization of phase velocity
velocper, velocms = np.loadtxt("./isolation/rayl.dsp", usecols=(5,6),
                               unpack=True)
crefper, crefvelo = np.loadtxt("./AK135.ASC", usecols=(2, 4), unpack=True, 
                               skiprows=1)

plt.plot(velocper, velocms, "o", label="Measured C")
plt.plot(crefper, crefvelo, label="Synthetic C [ak135]")
plt.xlabel("Period [s]")
plt.ylabel("Phase Velocity [km/s]")
plt.title("{}-{}".format(roughgva.id, roughgvb.id))
plt.xlim(velocper.min(), velocper.max())
plt.legend()
plt.savefig("{}.{}.mea.pha.png".format(roughgva.id, roughgvb.id))
plt.close()


# 8. self-consistency test part
intersta = interstagv(roughgva, roughgvb)
insta1, insta2, periods, velo1, velo2, velo3 = intersta.inter_gv_measurement()
wspline = np.arange(velocper.min(), velocper.max(), 1) 

## Derive group velocity with measured phase velocity 
f2 = interp1d(velocper, velocms, kind="cubic")
c = f2(wspline)
dcdt = np.gradient(c, 2, edge_order=2)
u = c / (1 + wspline * dcdt / c)


plt.plot(periods, velo3, "o", label="Interstation Measured U")
plt.plot(wspline, u, 'o', label="Derived U from C")
plt.plot(velocper, velocms, '+', label="Measured C")
plt.plot(wspline, c, label="Interpolated C")
plt.xlabel("Period [s]")
plt.ylabel("Velocity [km/s]")
plt.title("{}-{}".format(roughgva.id, roughgvb.id))
plt.xlim(PERMIN, PERMAX)
plt.legend()
plt.savefig("Self-consistency.{}.{}.png".format(roughgva, roughgvb))
