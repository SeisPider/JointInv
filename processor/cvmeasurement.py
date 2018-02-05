# /usr/bin/env python
# -*- coding:utf-8 -*-

# standard modules
import os
from os.path import join, basename, isfile
from subprocess import call
from copy import copy
from itertools import combinations
from glob import glob
import numpy as np

# self-developed module
from JointInv import teleseis, pserrors, logger
from JointInv.psconfig import get_global_param
from JointInv.pstwostation import StasCombine
from JointInv.machinelearn.base import gen_disp_classifier, velomap


MULTIPLEPROCESSING = {'Initialization': True,
                      'Measure disp': False}

NB_PROCESSING = None
if any(MULTIPLEPROCESSING.values()):
    import multiprocessing as mp
    mp.freeze_support()  # for windows

if __name__=='__main__':
    # Parameters Determination
    PARAM = get_global_param("../data/Configs")

    outbasename = ['teleseismic', '{}-{}'.format(PARAM.fstday.year,
                                                 PARAM.endday.year), 'XJ']
    # import catalog and stations
    catalogs = teleseis.get_catalog(catalog=PARAM.catalog_dir, fstday=PARAM.fstday,
                                    endday=PARAM.endday)
    stations = teleseis.scan_stations(dbdir=PARAM.stationinfo_dir,
                                      sacdir=join(PARAM.dataset_dir, "Isotraces"),
                                      fstday=PARAM.fstday, endday=PARAM.endday,
                                      dbtype="iso")
    # combine station pairs
    station_pairs = list(combinations(stations, 2))

    # -------------------------------------------------------------------------
    # select common line station_pairs and events
    # each element in judgment indicates a judgement result
    # and filter them
    # -------------------------------------------------------------------------
    judgements = [teleseis.common_line_judgement(event, station_pair)
                  for event in catalogs for station_pair in station_pairs]
    judgements = filter(lambda v: v is not None, judgements)


    def get_useable_combine(judgement):
        """
        Initializing func that return instance of clas Tscombine
        Function is ready to be parallelized

        Parameter
        =========
        judgement : tuple
            contains two `Station`s and a event dict.
        """
        station1 = judgement[0]
        station2 = judgement[1]
        event = judgement[2]
        try:
            stascombine = StasCombine(sta1=station1, sta2=station2, event=event)
            errmsg = None
        except pserrors.TracesNotCorrected as err:
            # cannot initialize class as response of traces area not removed
            stascombine = None
            errmsg = '{} -> skipping'.format(err)
        except Exception as err:
            # Unhandled exception
            stascombine = None
            errmsg = 'Unhandled error -> {}'.format(err)
        if errmsg:
            # print error message
            logger.error("{}.{}-{}.{}[{}]".format(station1.network, station1.name,
                                                  station2.network, station2.name,
                                                  errmsg))
        return stascombine


    # class initialization and waveform import
    if MULTIPLEPROCESSING['Initialization']:
        # multipleprocessing turned on: one process per combination
        pool = mp.Pool(NB_PROCESSING)
        combinations = pool.map(get_useable_combine, judgements)
        pool.close()
        pool.join()
    else:
        combinations = [get_useable_combine(s) for s in judgements]
    
    # Export possible combination
    with open("./combination.info", 'w') as f:
        for combination in combinations:
            f.writelines(combination.id + " \n")


    def get_possible_dispersion_point(combination, cpspath, dataset_dir, clf,
                                      refmodel="../data/Info/AK135SREGN.ASC",
                                      pmin=20, pmax=200):
        """
        Export possible dispersion points with sacpom96 and handle name of files

        Parameter
        =========

        combination : class `StasCombine`
            Class combination stations in the same line and event information
        cpspath : string or path-like object
            Indeicate location of computer program in seismology version <- 3.30
        dataset_dir : str
            Directory of the data base
        clf : `sklearn.classifier`
            classifier trained with hand-picked dispersion curves
        refmodel : str
            directory of the reference model, if not given, use AK135
        pmin : int or float
            Minimum period for dispersion curve extraction
        pmax : int or float
            Maximum period for dispersion curve extraction
        """
        cbn = copy(combination)
        eventid = cbn.event['id']
        err = None
        # judge whether combination exist
        logger.info("Handling combination -> {}".format(cbn.id))
        tr1pathdir = join(cbn.sta1.basedir, eventid, cbn.sta1.file)
        tr2pathdir = join(cbn.sta2.basedir, eventid, cbn.sta2.file)
        tr1path, tr2path = glob(tr1pathdir), glob(tr2pathdir)
        if not tr1path or not tr2path:
            logger.info("No file {} or {}".format(tr1pathdir, tr2pathdir))
            return

        # operate with sacpom96
        with open("cmdfil", "w") as cmdfil:
            string = "{}\n{}".format(str(tr1path[0]), str(tr2path[0]))
            cmdfil.writelines(string)
        sacpompath = join(cpspath, "sacpom96")
        cmdstring = "{} -C ./cmdfil -pmin {} -pmax {} -V -E -nray 500 -R \
                     -S >> log.info".format(sacpompath, pmin, pmax)
        call(cmdstring, shell=True, stderr=err)

        # move files to output directory
        for filename in glob("./[Pp][Oo][Mm]96*"):

            filebasename = basename(filename)
            # set the base dirname
            basedir = join(dataset_dir, "Isotraces")
            filebasename = ".".join([combination.id, filebasename])
            desdirname = join(basedir, eventid, filebasename)

            # check existence of log.info, if not, create it
            if not isfile("./log.info"):
                os.system("touch ./log.info")

            cmdstring = "mv {} {} >> log.info".format(filename, desdirname)
            call(cmdstring, shell=True, stderr=err)
        
        # estimate the phase velocity dispersion curve
        dspfilename = join(basedir, eventid, ".".join([combination.id, 
                                                       "pom96.dsp"]))

        # export interpolated dispersion curve
        try:
            clean_cv = velomap(dispinfo=dspfilename, refdisp=refmodel,
                               trained_model=clf, velotype="clean_cv",
                               line_smooth_judge=True, digest_type="poly",
                               treshold=3)
            periods = np.array([x.period for x in clean_cv.disprec])
            velos = np.array([x.velo for x in clean_cv.disprec])
            outputfilename = ".".join([dspfilename, "cv"])
            np.savetxt(outputfilename, np.matrix([periods, velos]).T, fmt="%.5f")

            # handle files
            tobe_moved = basename(dspfilename).split(".")[0] + "*"
            desdir = join(basedir, eventid)
            cmdstring = "mv {} {} >> log.info".format(tobe_moved, desdir)
            call(cmdstring, shell=True, stderr=err)
        except Exception as err:
            # Unhandled exception
            errmsg = 'Unhandled error -> {}'.format(err)
            logger.info("{} [] ".format(combination.id, errmsg))
        return None
    
    clf = gen_disp_classifier(mode="clean_cv", weighted=False)
    for combination in combinations:
        get_possible_dispersion_point(combination, PARAM.cpspath,
                                      PARAM.dataset_dir, clf=clf)
