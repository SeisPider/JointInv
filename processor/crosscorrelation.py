#!/usr/bin/python -u
import os
import gc
import sys
import warnings
import datetime as dt
import itertools as it
import pickle
import obspy.signal.cross_correlation
from termcolor import colored
from os.path import join

# Personal lib.
from JointInv.psconfig import get_global_param
from JointInv import pscrosscorr, pserrors, psstation, trimmer
from JointInv import logger
from JointInv.psrespider import SourceResponse


# #############################################################################
# Configuration and info. output
# #############################################################################

MULTIPROCESSING = {'merge trace': False,
                   'process trace': True,
                   'cross-corr': True}

# Set parallel computation
NB_PROCESSES = None
if any(MULTIPROCESSING.values()):
    import multiprocessing as mp
    mp.freeze_support()

PARAM = get_global_param("../data/Configs/")

# Notices fundamental parameters
logger.info("processing parameters")
if "{}".format(PARAM.mseed_dir) != "null":
    msg = "dir of miniseed -> {}".format(PARAM.mseed_dir)
    logger.info(msg)
if "{}".format(PARAM.dataless_dir) != "null":
    msg = "dir of dataless seed -> {}".format(PARAM.dataless_dir)
    logger.info(msg)
if "{}".format(PARAM.stationxml_dir) != "null":
    msg = "dir of stationxml -> {}".format(PARAM.stationxml_dir)
    logger.info(msg)
if "{}".format(PARAM.stationinfo_dir) != "null":
    msg = "dir of station database -> {}".format(PARAM.stationinfo_dir)
    logger.info(msg)
if "{}".format(PARAM.catalog_dir) != "null":
    msg = "dir of catalog -> {}".format(PARAM.catalog_dir)
    logger.info(msg)
if "{}".format(PARAM.sacpz_dir) != "null":
    msg = "dir of SAC PZ file -> {}".format(PARAM.sacpz_dir)
    logger.info(msg)
if "{}".format(PARAM.crosscorr_dir) != "null":
    msg = "output dir -> {}".format(PARAM.crosscorr_dir)
    logger.info(msg)
msg = "bandpass -> {:.1f}-{:.1f} s".format(1.0 / PARAM.freqmax,
                                           1.0 / PARAM.freqmin)
logger.info(msg)


# Set normalization parameters
if PARAM.onebit_norm:
    logger.info("Temporal normalization -> one-bit normalization")
else:
    s = ("Running mean normalization -> earthquake band ({:.1f}-{:.1f} s)")
    logger.info(s.format(1.0 / PARAM.freqmin_eq, 1.0 / PARAM.freqmax_eq))

fmt, s = '%d/%m/%Y', "stacked length -> {}-{}"
logger.info(s.format(PARAM.fstday.strftime(fmt), PARAM.endday.strftime(fmt)))

# Set filter notification
subset = PARAM.crosscorr_stations_subset
if subset:
    logger.info("Only for stations: {}".format(', '.join(subset)))

# Check Source of response file
responsefrom = []
if PARAM.use_datalesspaz:
    responsefrom.append('datalesspaz')
if PARAM.use_stationxml:
    responsefrom.append('xmlresponse')
if PARAM.use_response_spider:
    responsefrom.append('sacpz')

OUTBASENAME_PARTS = [
    'xcorr',
    '-'.join(s for s in subset) if subset else None,
    '{}-{}'.format(PARAM.fstday.year, PARAM.endday.year),
    '1bitnorm' if PARAM.onebit_norm else None,
    '+'.join(responsefrom)]

OUTFILESPATH = join(PARAM.crosscorr_dir, '_'.join(p for p in OUTBASENAME_PARTS if p))
msg = 'Default name of output files (no ext.):\n\t\t\t\t"{}"\n'.format(OUTFILESPATH)
logger.info(msg)


# #############################################################################
# Main progr
# #############################################################################

# Reading inventories in dataless seed and/or StationXML files
dataless_inventories = []
if PARAM.use_datalesspaz:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dataless_inventories = psstation.get_dataless_inventories(PARAM.dataless_dir,
                                                                  verbose=True)

xml_inventories = []
if PARAM.use_stationxml:
    xml_inventories = psstation.get_stationxml_inventories(PARAM.stationxml_dir,
                                                           verbose=True)
# #############################################################################
# Import stations info. and response files
# #############################################################################

responses_spider = {}
if PARAM.use_response_spider:
    # import response files from CENC
    responses_spider.update({"CENC": SourceResponse(join(PARAM.sacpz_dir,
                                                         "CENC"), "CENC")})

# Initializing Trimer and scan stations 
trimmer = trimmer.Trimmer(stationinfo=PARAM.stationinfo_dir,
                          catalog=PARAM.catalog_dir,
                          sacdir=join(PARAM.dataset_dir, "RawTraces"),
                          velomin=PARAM.velomin, velomax=PARAM.velomax)

stations = psstation.get_stations(mseed_dir=PARAM.mseed_dir,
                                  xml_inventories=xml_inventories,
                                  dataless_inventories=dataless_inventories,
                                  database=trimmer.stations,
                                  networks=PARAM.network_subset,
                                  channels=PARAM.channel_subset,
                                  startday=PARAM.fstday,
                                  endday=PARAM.endday,
                                  verbose=True)

# #############################################################################
# Processing traces
# #############################################################################

# Initializing collection of cross-correlations
xc = pscrosscorr.CrossCorrelationCollection()

# Loop on days
nday = (PARAM.endday - PARAM.fstday).days + 1
dates = [PARAM.fstday + dt.timedelta(days=i) for i in range(nday)]

# Stacking CCFs monthly
for date in dates:

    # Exporting the collection of cross-correlations after the end of each
    # processed month (allows to restart after a crash from that date)

    # export all ccfs in one file
    lastdate = date - dt.timedelta(days=1)
    monthdelta = date.month - lastdate.month
    if monthdelta != 0 and xc:
        OUTFILEPATH = "{}_{}".format(OUTFILESPATH, lastdate.strftime("%Y%m"))
        msg = "Exporting cross-correlations calculated to: \n " + OUTFILEPATH
        logger.info(msg)
        xc.export(OUTFILEPATH, onlypickle=True)

        # export ccfs into many sac files
        xc.export2sac(crosscorr_dir=PARAM.crosscorr_dir)

        # Finish one month computation, exit script
        logger.info("Congraducation for finishing computation !")
        sys.exit()


    logger.info("Processing data of day {}".format(date))

    # loop on stations appearing in subdir corresponding to current month
    date_subdir = date.strftime("%Y%m%d")
    date_stations = sorted(
        sta for sta in stations if date_subdir in sta.subdirs)

    # subset if stations (if provided)
    if PARAM.crosscorr_stations_subset:
        date_stations = [sta for sta in date_stations
                         if sta.name in PARAM.crosscorr_stations_subset]

    # =============================================================
    # preparing functions that get one merged trace per station
    # and pre-process trace, ready to be parallelized (if required)
    # =============================================================

    def get_merged_trace(station):
        """
        Preparing func that returns one trace from selected station,
        at current date. Function is ready to be parallelized.
        """
        try:
            SKIPLOCS, MINFILL = PARAM.crosscorr_skiplocs, PARAM.minfill
            trace = pscrosscorr.get_merged_trace(station=station,
                                                 date=date,
                                                 skiplocs=SKIPLOCS,
                                                 minfill=MINFILL)
            errmsg = None
        except pserrors.CannotPreprocess as err:
            # cannot preprocess if no trace or daily fill < *minfill*
            trace = None
            errmsg = '{}: skipping'.format(err)
        except Exception as err:
            # unhandled exception!
            trace = None
            errmsg = 'Unhandled error: {}'.format(err)

        if errmsg:
            # printing error message
            logger.error('{}.{} [{}] '.format(
                station.network, station.name, errmsg))
        return trace

    def preprocessed_trace(trace, response, PARAM, trimmer=None,
                           responses_spider=None,  debug=False):
        """
        Preparing func that returns processed trace: processing includes
        removal of instrumental response, band-pass filtering, demeaning,
        detrending, downsampling, time-normalization and spectral whitening
        (see pscrosscorr.preprocess_trace()'s doc)

        Function is ready to be parallelized.
        """
        if not (trace):
            return
        if not (response or responses_spider):
            return

        logger.info("Preprocessing.{}".format(trace.id))

        if debug:
            print(trace, response, trimmer, responses_spider)

        network = trace.stats.network
        station = trace.stats.station
        try:
            pscrosscorr.preprocess_trace(
                trace=trace,
                trimmer=trimmer,
                paz=response,
                responses_spider=responses_spider,
                freqmin=PARAM.freqmin,
                freqmax=PARAM.freqmax,
                freqmin_earthquake=PARAM.freqmin_eq,
                freqmax_earthquake=PARAM.freqmax,
                corners=PARAM.corners,
                zerophase=PARAM.zerophase,
                period_resample=PARAM.period_resample,
                onebit_norm=PARAM.onebit_norm,
                window_time=PARAM.window_time,
                window_freq=PARAM.window_freq)
            msg = 'ok'
        except pserrors.CannotPreprocess as err:
            # cannot preprocess if no instrument response was found,
            # trace data are not consistent etc. (see function's doc)
            trace = None
            msg = '{}: skipping'.format(err)
        except Exception as err:
            # unhandled exception!
            trace = None
            msg = 'Unhandled error: {}'.format(err)
            # printing output (error or ok) message
            logger.error('{}.{} [{}] '.format(network, station, msg))
        # although processing is performed in-place, trace is returned
        # in order to get it back after multi-processing
        return trace

    # ====================================
    # getting one merged trace per station
    # ====================================
    if MULTIPROCESSING['merge trace']:
        # multiprocessing turned on: one process per station
        pool = mp.Pool(NB_PROCESSES)
        traces = pool.map(get_merged_trace, date_stations)
        pool.close()
        pool.join()
    else:
        # multiprocessing turned off: processing stations one after another
        traces = [get_merged_trace(s) for s in date_stations]

    # =====================================================
    # getting or attaching instrumental response
    # (parallelization is difficult because of inventories)
    # =====================================================
    responses = []
    for tr in traces:
        if not tr:
            responses.append(None)
            continue

    # responses elements can be (1) dict of PAZ if response found in
    # dataless inventory, (2) None if response found in StationXML
    # inventory (directly attached to trace) or (3) False if no
    # response found
        try:
            response = pscrosscorr.get_or_attach_response(
                trace=tr,
                dataless_inventories=dataless_inventories,
                xml_inventories=xml_inventories,
                responses_spider=responses_spider)
            errmsg = None
        except pserrors.CannotPreprocess as err:
            # response not found
            response = False
            errmsg = '{}: skipping'.format(err)
        except Exception as err:
            # unhandled exception!
            response = False
            errmsg = 'Unhandled error: {}'.format(err)

        responses.append(response)
        if errmsg:
            # printing error message
            net, name = tr.stats.network, tr.stats.station
            logger.error('{}.{} [{}] '.format(net, name, errmsg))
    # =================
    # processing traces
    # =================
    if MULTIPROCESSING['process trace']:
        # multiprocessing turned on: one process per station
        comblist = list(zip(traces, responses, it.repeat(PARAM),
                            it.repeat(trimmer), it.repeat(responses_spider)))

        pool = mp.Pool(NB_PROCESSES)
        traces = pool.starmap(preprocessed_trace, comblist)
        pool.close()
        pool.join()
    else:
        # multiprocessing turned off: processing stations one after another
        comblist = list(zip(traces, responses, it.repeat(trimmer)))
        traces = [preprocessed_trace(tr, res, PARAM, trim, responses_spider)
                  for tr, res, trim in comblist]

    # setting up dict of current date's traces, {station: trace}
    tracedict = {s.name: trace for s, trace in zip(
        date_stations, traces) if trace}

    # TODO: I ensured the waveform can be preprocessed properly
    #       verify the EGF result is right
    # ==============================================
    # stacking cross-correlations of the current day
    # ==============================================

    if len(tracedict) < 2:
        logger.error("No cross-correlation for this day")
        continue


    xcorrdict = {}
    if MULTIPROCESSING['cross-corr']:
        # if multiprocessing is turned on, we pre-calculate cross-correlation
        # arrays between pairs of stations (one process per pair) and feed
        # them to xc.add() (which won't have to recalculate them)
        logger.info("Pre-calculating cross-correlation arrays")

        def xcorr_func(pair):
            """
            Preparing func that returns cross-correlation array
            beween two traces
            """
            (s1, tr1), (s2, tr2) = pair
            logger.debug('{}-{} '.format(s1, s2))
            shift = int(PARAM.crosscorr_tmax / PARAM.period_resample)
            xcorr = obspy.signal.cross_correlation.correlate(
                tr1, tr2, shift=shift)
            return xcorr

        pairs = list(it.combinations(sorted(tracedict.items()), 2))
        pool = mp.Pool(NB_PROCESSES)
        xcorrs = pool.map(xcorr_func, pairs)
        pool.close()
        pool.join()
        xcorrdict = {(s1, s2): xcorr for ((s1, _), (s2, _)),
                     xcorr in zip(pairs, xcorrs)}

    logger.info("Stacking cross-correlations")
    xc.add(tracedict=tracedict,
           stations=stations,
           xcorr_tmax=PARAM.crosscorr_tmax,
           xcorrdict=xcorrdict,
           verbose=not MULTIPROCESSING['cross-corr'])
