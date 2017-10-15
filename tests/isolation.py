#! /usr/bin/env python
import os
from os.path import join
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from JointInv.machinelearn.base import load_disp, velomap
from JointInv.global_var import logger

def event_isolation(eventdir, refgvpath, outpath):
    
    dispfiles = glob.glob(join(eventdir, "*.disp"))
    for dispfile in dispfiles:
        
        # mainly 
        judgement = velomap(dispfile, refgvpath, clf, periodmin=permin, periodmax=permax)
        outfilepath = join(outpath, ".".join([judgement.id, "d"]))
        judgement.MFT962SURF96(outfilepath, CPSPATH)
        judgement.isowithsacmat96(eventdir, outfilepath, cpspath=CPSPATH)

        # move files
        SACsfiles = join(eventdir, "*.SACs")
        cmndstr = "mv {} {}".format(SACsfiles, outpath)
        os.system(cmndstr)


if __name__=="__main__":

    """
    Isolate fundamental rayleigh wave with automatically picked group velocity
    dispersion curve with Decision Tree algorithm
    """
    CPSPATH = "~/src/CPS/PROGRAMS.330/bin"
    
    try:
        rootdir = sys.argv[1]
    except IndexError:
        logger.error("Please input directory of dataset !")
        rootdir = input()

    # set para.
    permax = 100
    permin = 25
    velomin = 2.0
    velomax = 5.0
    disp = load_disp()
    # train model
    n_classes = 2
    pair = [0,1]
    x = disp.data[:, pair]
    y = disp.target
    errweight = 1.0/disp.data[:,-1]
    clf = DecisionTreeClassifier(min_samples_split=20).fit(x, y, sample_weight=errweight)


    eventlist = glob.glob(join(rootdir, "rawtraces", "*"))
    refgvpath = "../data/info/SREGN.ASC"
    allsacmftpath = "../JointInv/scripts/allsacmft.sh"

    for event in eventlist:
        
        # automatically perform sacmft96
        cmndstr = "sh {} {} {} {} {} {}".format(allsacmftpath, event,
                                                permin, permax, velomin, velomax)
        os.system(cmndstr)

        # check and create output dir
        eventid = event.split("/")[-1]
        outputdir = join(rootdir, "isotraces", eventid)
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)

        # isolate traces of this event
        try:
            event_isolation(event, refgvpath, outputdir)
            msg = 'ok'
        except Exception as err:
            # Unhandled exception!
            msg = 'Unhandled error: {}'.format(err)
        continue
    if os.path.isfile("./disp.out"):
        os.remove("./disp.out")
