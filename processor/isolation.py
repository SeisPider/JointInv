#! /usr/bin/env python
import os
from os.path import join
import sys
import glob
from sklearn.tree import DecisionTreeClassifier as DTC

from JointInv.machinelearn.base import load_disp, velomap
from JointInv.global_var import logger
from JointInv.psconfig import get_global_param
from JointInv import psutils


# Isolate waveform of an specific event
def event_isolation(eventdir, refgvpath, outpath,
                    cpspath=COMPUTER_PROGRAMS_IN_SEISMOLOGY_DIR,
                    classifier=None, periodmin=25, periodmax=100):

    dispfiles = glob.glob(join(eventdir, "*.disp"))
    for dispfile in dispfiles:

        # mainly
        judgement = velomap(dispfile, refgvpath, classifier, periodmin=permin,
                            periodmax=permax)
        outfilepath = join(outpath, ".".join([judgement.id, "d"]))
        judgement.MFT962SURF96(outfilepath, cpspath)
        judgement.isowithsacmat96(eventdir, outfilepath, cpspath=cpspath)
        # move files
        SACsfiles = join(eventdir, "*.SACs")
        cmndstr = "mv {} {}".format(SACsfiles, outpath)
        os.system(cmndstr)


if __name__ == "__main__":

    """
    Isolate fundamental rayleigh wave with automatically picked group velocity
    dispersion curve with Decision Tree algorithm
    """
    # import configuration file
    GBPARA = get_global_param("../data/configs/")


    cpspath = GBPARA.cpspath

    # import training data and train model
    disp = load_disp()
    n_classes, pair = 2, [0, 1]
    x = disp.data[:, pair]
    y = disp.target
    errweight = 1.0 / disp.data[:, -1]
    clf = DTC(min_samples_split=20).fit(x, y, sample_weight=errweight)

    # set para. for extract rough group velocity curve
    try:
        rootdir = sys.argv[1]
    except IndexError:
        logger.error("Please input directory of dataset !")
        rootdir = input()

    permin, permax = 25, 100  # set period region
    velomin, velomax = GBPARA.signal_window_vmin, GBPARA.signal_window_vmax
    eventlist = glob.glob(join(rootdir, "rawtraces", "*"))
    refgvpath = "../data/info/SREGN.ASC"

    allsacmftpath = psutils.locate_external_scripts("allsacmft.sh")
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
            event_isolation(event, refgvpath, outputdir, cpspath=cpspath,
                            classifier=clf, periodmin=permin, periodmax=permax)
            msg = 'ok'
        except Exception as err:
            # Unhandled exception!
            msg = 'Unhandled error: {}'.format(err)
        continue
    if os.path.isfile("./disp.out"):
        os.remove("./disp.out")
