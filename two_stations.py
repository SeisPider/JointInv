# /usr/bin/env python
# -*- coding:utf-8 -*-
from pysismo import teleseis, pstwostation, pserrors
from pysismo.global_var import logger
from pysismo.pstwostation import Tscombine
from pysismo.psconfig import (FTAN_ALPHA, FIRSTDAY, LASTDAY,
                              TELESEISMIC_DISPERSION_DIR,
                              COMPUTER_PROGRAMS_IN_SEISMOLOGY_DIR)

from itertools import combinations
import itertools as it
import numpy as np
import dill
import os
import subprocess
import shutil
import glob

MULTIPLEPROCESSING = {'Initialization': False,
                      'Measure disp': False}


DATADIR = "./DATA"
ISOLATION_DIR = DATADIR + "/Isolation"
EVENT_DIR = DATADIR + "/GoodTrace"

NB_PROCESSING = None
if any(MULTIPLEPROCESSING.values()):
    import multipleprocessing as mp
    mp.freeze_support()  # for windows

OUTBASENAME_PARTS = ['teleseismic', '{}-{}'.format(FIRSTDAY.year, LASTDAY.year),
                     'XJ']
OUTFILEPATH = os.path.join(TELESEISMIC_DISPERSION_DIR,
                           '_'.join(OUTBASENAME_PARTS))

# import catalog and stations
catalogs = teleseis.get_catalog()
stations = teleseis.scan_stations()
# combine station pairs
station_pairs = list(combinations(stations, 2))

# select common line station_pairs and events
# each element in judgment indicates a judgement result
# and filter them
judgements = [pstwostation.common_line_judgement(event, station_pair)
             for event in catalogs for station_pair in station_pairs]
judgements = filter(lambda v: v is not None, judgements)

def get_useable_combine(judgement):
    """
    Initializing func that return instance of clas Tscombine
    Function is ready to be parallelized
    """
    station1 = judgement[0]
    station2 = judgement[1]
    event = judgement[2]
    try:
        tscombine = Tscombine(sta1=station1, sta2=station2, event=event)
        errmsg = None
    except pserrors.TracesNotCorrected as err:
        # cannot initialize class as response of traces area not removed
        tscombine = None
        errmsg = '{} -> skipping'.format(err)
    except Exception as err:
        # Unhandled exception
        tscombine = None
        errmsg = 'Unhandled error -> {}'.format(err)
    if errmsg:
        # print error message
        logger.error("{}.{}-{}.{}[{}]".format(station1.network, station1.name,
                                              station2.network, station2.name,
                                              errmsg))
    return tscombine


# class initialization and waveform import
if MULTIPLEPROCESSING['Initialization']:
    # multipleprocessing turned on: one process per combination
    pool = mp.Pool(NB_PROCESSING)
    combinations = pool.map(get_useable_combine, judgements)
    pool.close
    pool.join()
else:
    combinations = [get_useable_combine(s) for s in judgements]


# ouput matched two stationsi
with open("./LOG.INFO", 'a') as f:
    for combination in combinations:
        f.writelines(combination.id+" \n")
    f.close()

# Isolate fundamental rayleigh wave with computer programs in seismology
def Isolate_Fundamental_Rayleigh_Wave(eventfolder, datadir=DATADIR,
                                      eventdir=EVENT_DIR,
                                      isolationdir=ISOLATION_DIR):
    """
    Measure Group velocity of traces and Cut fundamental rayleigh wave out
    """
    # obtain catalog
    os.chdir(os.path.join(EVENT_DIR, eventfolder))
    p = subprocess.Popen(['sh'], stdin=subprocess.PIPE)
    s = "do_mft *.SAC"
    p.communicate(s.encode())
    
    os.chdir("../../../")

    # move files to Isolated part
    suffixs = ["*.SACs", "*SACr", "*.dsp"]
    for suffix in suffixs: 

        sourcefiles = glob.glob(os.path.join(EVENT_DIR, eventfolder, suffix))
        destination = os.path.join(isolationdir, eventfolder)
        if not os.path.exists(destination):
            os.mkdir(destination)
        
        for sourcefile in sourcefiles:
            shutil.move(sourcefile, destination)
    return None

events = glob.glob(os.path.join(EVENT_DIR, "*"))

for event in events:
    eventfolder = os.path.split(event)[-1]
    Isolate_Fundamental_Rayleigh_Wave(eventfolder)

# Export data
with open('{}.dill'.format(OUTFILEPATH), 'wb') as f:
    msg = "Exporting dispersion curves calculated to -> {}".format(f.name)
    logger.info(msg)
    dill.dump(tscombinations, f, protocol=4)
