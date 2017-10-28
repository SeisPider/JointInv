#! /usr/bin/env python
import os
from os.path import join
import sys
import glob
from sklearn.tree import DecisionTreeClassifier as DTC

from JointInv.machinelearn.base import gen_disp_classifier, velomap
from JointInv.psconfig import get_global_param
from JointInv import psutils, logger

# Isolate waveform of an specific event
def event_isolation(eventdir, refgvpath, outpath,
                    cpspath=None, classifier=None, periodmin=25, periodmax=100):

    dispfiles = glob.glob(join(eventdir, "*.disp"))
    for dispfile in dispfiles:

        # mainly
        judgement = velomap(dispfile, refgvpath, trained_model=classifier,
                            treshold=2)
        outfilepath = join(outpath, ".".join([judgement.id, "d"]))
        judgement.MFT962SURF96(outfilepath, cpspath)
        judgement.isowithsacmat96(srcpath=eventdir, surf96filepath=outfilepath,
                                  cpspath=cpspath)
        # move isolated traces and images in current directory to outpath 
        SACsfiles = join(eventdir, "*.SACs")
        cmndstr = "mv {} {}\n mv ./*.png {}".format(SACsfiles, outpath, outpath)
        os.system(cmndstr)

        # move measured dispersion curve to result path
        cmndstr = "mv ./disp.out {}".format(join(outpath, ".".join([judgement.id,
                                                                    "disp.out"])))
        os.system(cmndstr)


if __name__ == "__main__":

    """
    Isolate fundamental rayleigh wave with automatically picked group velocity
    dispersion curve with Decision Tree algorithm
    """
    # import configuration file
    gbparam = get_global_param("../data/configs/")

    # import training data and train model
    clf = gen_disp_classifier() 

    # set para. for extract rough group velocity curve
    try:
        dataset_dir =  gbparam.dataset_dir
    except IndexError:
        logger.error("Please input directory of dataset !")
        rawtracedir = input()

    permin, permax = 25, 100  # set period region
    velomin, velomax = gbparam.signal_window_vmin, gbparam.signal_window_vmax
    eventlist = glob.glob(join(dataset_dir, "rawtraces", "*"))
    refgvpath = "../data/info/SREGN.ASC"

    allsacmftpath = psutils.locate_external_scripts("allsacmft.sh")
    
    for event in eventlist:
        
        # automatically perform sacmft96
        cmndstr = "sh {} {} {} {} {} {}".format(allsacmftpath, event,
                                                permin, permax, velomin, velomax)
        os.system(cmndstr)

        # check and create output dir
        eventid = event.split("/")[-1]
        outputdir = join(dataset_dir, "isotraces", eventid)
        os.makedirs(outputdir, exist_ok=True)
        
        # isolate traces of this event
        try:
            event_isolation(event, refgvpath, outputdir, cpspath=gbparam.cpspath,
                            classifier=clf, periodmin=permin, periodmax=permax)
            msg = 'ok'
        except Exception as err:
            # Unhandled exception!
            msg = 'Unhandled error: {}'.format(err)
        continue
        
        # measure clean phase velocity dispersion curve

