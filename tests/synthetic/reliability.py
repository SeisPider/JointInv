#/usr/bin/env python
from JointInv.machinelearn.base import gen_disp_classifier, velomap
from JointInv.machinelearn.twostationgv import interstagv
import matplotlib.pyplot as plt
import numpy as np
import random as rdn
from glob import glob
from os.path import join
from subprocess import call
import os


def onetest(clf, CPSPATH, clfcv, percent=0.2):
    dispfile1 = "./sacmft/B00101Z00.sac.mft96.disp"
    dispfile2 = "./sacmft/B00201Z00.sac.mft96.disp"
    refdisp = "../../data/info/SREGN.ASC"

    roughgva = velomap(dispinfo=dispfile1, refdisp=refdisp, trained_model=clf,
                       line_smooth_judge=True, digest_type="poly")
    roughgvb = velomap(dispinfo=dispfile2, refdisp=refdisp, trained_model=clf,
                       line_smooth_judge=True, digest_type="poly")

    for judgement in [roughgva, roughgvb]:
        outpath = join("./isolation/", ".".join([judgement.id, "d"]))
        
        # add noise into raw dispersion curve
        for record in judgement.disprec:
            record.velo += rdn.gauss(0,1) * percent * record.velo 

        judgement.MFT962SURF96(outpath, CPSPATH)
        excfile = join(CPSPATH, "sacmat96")
        if judgement == roughgva:
            judgement.isowithsacmat96(surf96filepath=outpath,
                                      srcsacfile="./B00101Z00.sac",
                                      cpspath=CPSPATH)
        else:
            judgement.isowithsacmat96(surf96filepath=outpath,
                                      srcsacfile="./B00201Z00.sac",
                                      cpspath=CPSPATH)

        os.system("mv ./*.sac[rs] ./isolation/")

    # excet pom96
    pmin, pmax = 25, 100
    filelists = glob("./isolation/*.sacs")
    with open("cmdfil", "w") as cmdfil:
        string = "{}\n{}".format(str(filelists[0]), str(filelists[1]))
        cmdfil.writelines(string)
    sacpompath = join(CPSPATH, "sacpom96")
    cmdstring = "{} -C ./cmdfil -pmin {} -pmax {} -V -E -nray 250 -R -S >> log.info".format(
                sacpompath, pmin, pmax)
    call(cmdstring, shell=True)
    os.system("mv pom96.dsp synthetic_event-B001-B002.pom96.dsp")

    phasevelomap = velomap(dispinfo="./synthetic_event-B001-B002.pom96.dsp",
                     refdisp=refdisp, trained_model=clfcv, velotype="clean_cv",
                     line_smooth_judge=True, digest_type="poly", treshold=3)
    os.system("rm ./synthetic_event-B001-B002.pom96.dsp")
    os.system("rm ./isolation/*.sac[rs]")
    return phasevelomap, roughgva, roughgvb

if __name__=="__main__":
    # do monte carlo reliability test
    clf = gen_disp_classifier()

    # Isolate fundamental rayleigh waves
    CPSPATH = "~/src/CPS/PROGRAMS.330/bin"
    clfcv = gen_disp_classifier(mode="clean_cv", weighted=False)
    cvelomap, gvelomapa, gvelomapb = [], [], []
    for i in range(1):
        cvmap, rvmapa, rvmapb = onetest(clf=clf, CPSPATH=CPSPATH, clfcv=clfcv,
                                        percent=0.0)
        cvelomap.append(cvmap)
        gvelomapa.append(rvmapa)
        gvelomapb.append(rvmapb)

    # visualization
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    
    # import synthetic
    period, synrc, synru = np.loadtxt("./SREGN.ASC", usecols=(2, 4, 5),
                                      unpack=True, skiprows=1)
    axes[0].plot(period, synrc)

    print (len(cvelomap))
    for idx, velomap in enumerate(cvelomap):
        period = np.array([x.period for x in velomap.disprec])
        velo = np.array([x.velo for x in velomap.disprec])
        axes[0].plot(period, velo, "o")
        axes[0].set_xlabel("Period [s]")
        axes[0].set_ylabel("Phase Velocity [km/s]")
        axes[0].set_title("Measured Phase Velocity")
        axes[0].set_xlim(period.min(), period.max())

    
    for velomap in gvelomapa:
        period = np.array([x.instaper for x in velomap.disprec])
        velo = np.array([x.velo for x in velomap.disprec])
        axes[1].plot(period, velo, 'ro')
        axes[1].set_xlabel("Period [s]")
        axes[1].set_ylabel("Group Velocity [km/s]")
        axes[1].set_title("Disturbed Raw Group Velocity")
        axes[1].set_xlim(25, 100)
    plt.savefig("reliability_test.png")
    #plt.show()

