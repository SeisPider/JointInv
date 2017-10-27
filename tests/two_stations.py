# /usr/bin/env python
# -*- coding:utf-8 -*-
from JointInv import teleseis, pserrors, logger
from JointInv.psconfig import  get_global_param
from JointInv.pstwostation import StasCombine 

from itertools import combinations
import os
MULTIPLEPROCESSING = {'Initialization': False,
                      'Measure disp': False}


NB_PROCESSING = None
if any(MULTIPLEPROCESSING.values()):
    import multipleprocessing as mp
    mp.freeze_support()  # for windows

# Parameters Determination
gbparam = get_global_param("../data/configs")

outbasename = ['teleseismic', '{}-{}'.format(gbparam.fstday.year,
                                             gbparam.endday.year), 'XJ']
outfilepath = os.path.join(gbparam.teledisp_dir, '_'.join(outbasename))

# import catalog and stations
catalogs = teleseis.get_catalog(catalog=gbparam.catalog_dir, fstday=gbparam.fstday,
                                endday=gbparam.endday)
stations = teleseis.scan_stations(dbdir=gbparam.stationinfo_dir, 
                                  sacdir=gbparam.isolation_output_dir,
                                  fstday=gbparam.fstday, endday=gbparam.endday,
                                  dbtype="iso")
# combine station pairs
station_pairs = list(combinations(stations, 2))

# select common line station_pairs and events
# each element in judgment indicates a judgement result
# and filter them
judgements = [teleseis.common_line_judgement(event, station_pair)
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
