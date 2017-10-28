# /usr/bin/env python
# -*- coding:utf-8 -*-
from JointInv import teleseis, pserrors, logger
from JointInv.psconfig import  get_global_param
from JointInv.pstwostation import StasCombine 

from itertools import combinations, repeat
import os
from os.path import join, basename
from subprocess import call
from copy import copy
from glob import glob
MULTIPLEPROCESSING = {'Initialization': True,
                      'Measure disp': False}


NB_PROCESSING = None
if any(MULTIPLEPROCESSING.values()):
    import multiprocessing as mp
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
                                  sacdir=join(gbparam.dataset_dir, "isotraces"),
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

def get_possible_dispersion_point(combination, cpspath, dataset_dir, pmin=25,
                                  pmax=100):
    """
    Export possible dispersion points with sacpom96 and handle name of files

    Parameter
    =========

    combination : class `StasCombine`
        Class combination stations in the same line and event information
    cpspath : string or path-like object
        Indeicate location of computer program in seismology version <- 3.30
    pmin : int or float
        Minimum period for dispersion curve extraction
    pmax : int or float
        Maximum period for dispersion curve extraction
    """
    cbn = copy(combination)
    eventid = cbn.event['id']
    err= None 
    # judge whether combination exist
    logger.info("Handling combination -> {}".format(cbn.id))
    tr1pathdir = join(cbn.sta1.basedir, eventid, cbn.sta1.file)
    tr2pathdir = join(cbn.sta2.basedir, eventid, cbn.sta2.file)
    tr1path = glob(tr1pathdir)
    tr2path = glob(tr2pathdir)
    if not tr1path or not tr2path:
        logger.info("No file {} or {}".format(tr1pathdir, tr2pathdir))
        return
    
    # operate with sacpom96
    with open("cmdfil", "w") as cmdfil:
        string = "{}\n{}".format(str(tr1path[0]), str(tr2path[0]))
        cmdfil.writelines(string)
    sacpompath = join(cpspath, "sacpom96")
    cmdstring = "{} -C ./cmdfil -pmin {} -pmax {} -V -E -nray 250 -R -S >> log.info".format(
                sacpompath, pmin, pmax)
    call(cmdstring, shell=True, stderr=err)

    # move files to output directory
    for filename in glob("./[Pp][Oo][Mm]96*"):
       
        filebasename = basename(filename) 
        # set the base dirname
        basedir = join(dataset_dir, "isotraces")
        filebasename = ".".join([combination.id, filebasename])
        desdirname = join(basedir, eventid, filebasename)

        cmdstring = "mv {} {} >> log.info".format(filename, desdirname)
        call(cmdstring, shell=True, stderr=err)
    return err

for combination in combinations:
    get_possible_dispersion_point(combination, gbparam.cpspath, 
                                   gbparam.dataset_dir)

# ouput matched two stationsi
with open("./combination.info", 'w') as f:
    for combination in combinations:
        f.writelines(combination.id+" \n")
