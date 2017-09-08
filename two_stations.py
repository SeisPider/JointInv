# /usr/bin/env python
# -*- coding:utf-8 -*-
from pysismo import teleseis, pstwostation, pserrors
from pysismo.global_var import logger
from pysismo.pstwostation import Tscombine
from pysismo.psconfig import (FTAN_ALPHA, FIRSTDAY, LASTDAY,
                              TELESEISMIC_DISPERSION_DIR)

from itertools import combinations
import itertools as it
import numpy as np
import dill
import os

MULTIPLEPROCESSING = {'Initialization': False,
                      'Measure disp': False}

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

# process traces and measure dispersion curves
def measure_teleseismic_dispersion(tscombine, periods, alpha=FTAN_ALPHA,
                                   shift_len=500.0, demoT=80):
    """
    Measure teleseismic fundamental Rayleigh wave dispersion and handle errors

    :type tscombines: class `.pysismo.pstwostation.Tscombine`
    :param tscombines: contains information and bounded with method to calculate
                       Rayleigh wave dispersion curves
    :type periods: class `numpy.array`
    :param periods: contains periods band we are interested
    :type alpha: float default `FTAN_ALPHA` from `.pysismo.psconfig`
    :param alpha: factor in Gaussian filter
    :type shift_len: float default `500.0`
    :param shift_len: time shift in cross-correlation [unit is second]
    """
    logger.info("Measure dispersion curve of {}".format(tscombine.id))

    # debug
    logger.debug("Periods -> {}".format(len(periods)))
    logger.debug("alpha -> {}".format(alpha))
    logger.debug("shift -> {}".format(shift_len))
    try:
        tscombine.measure_dispersion(periods=periods, alpha=alpha,
                                     shift_len=shift_len)
        errmsg = None
    except pserrors.CannotMeasureDispersion as err:
        # can not measure
        errmsg = "{} -> skipping".format(err)
    except Exception as err:
        # Unhandled exception
        errmsg = 'Unhandled error -> {}'.format(err)
    if errmsg:
        # print error message
        logger.error("{}.{}-{}.{}[{}]".format(tscombine.sta1.network,
                                              tscombine.sta1.name,
                                              tscombine.sta2.network,
                                              tscombine.sta2.name,
                                              errmsg))
        tscombine = None
    return tscombine

# Import periods that we interested in
periods = np.arange(20, 100)
# Filter None combination
combinations = filter(lambda v: v is not None, combinations)

# Measure dispersion
if MULTIPLEPROCESSING['Measure disp']:
    # multipleprocessing turned on: one process per tscombine instance
    pool = mp.Pool(NB_PROCESSING)
    tscombinations = pool.starmap(measure_teleseismic_dispersion, list(
                                  zip(combinations, it.repeat(periods))))
    pool.close()
    pool.join()
else:
    tscombinations = [measure_teleseismic_dispersion(tscombine, periods)
                      for tscombine in combinations]
# Export data
with open('{}.dill'.format(OUTFILEPATH), 'wb') as f:
    msg = "Exporting dispersion curves calculated to -> {}".format(f.name)
    logger.info(msg)
    dill.dump(tscombinations, f, protocol=4)
