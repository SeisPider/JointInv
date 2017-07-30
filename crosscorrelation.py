#!/usr/bin/python -u
"""
[Advice: run this script using python with unbuffered output:
`python -u crosscorrelation.py`]

This script reads seismic waveform data from a set of stations, and
calculates the cross-correlations between all pairs of stations. The
data (in miniseed format) must be located in folder *MSEED_DIR*. The
stations information (coordinates, instrument response) can be read
from dataless seed files (if *USE_DATALESSPAZ* = True) located in
folder *DATALESS_DIR*, and/or stationXML files (if *USE_STATIONXML* =
True) located in folder *STATIONXML_DIR*. Note that two different
stations MUST HAVE DIFFERENT NAMES, even if they do not belong to
the same network. Also, one given station cannot have several
sets of coordinates: if so, it will be skipped.

In the current version of the program, miniseed files MUST be
organized inside their directory as:
<year>-<month>/<network>.<station>.<channel>.mseed, e.g.:
1988-10/BL.JFOB.BHZ.mseed
So, there is one sub-directory per month, and inside it, one miniseed
file  per month and per station.

The implemented algorithm follows the lines of Bensen et al.,
"Processing seismic ambient noise data to obtain reliable broad-band
surface wave dispersion measurements", Geophys. J. Int. (2007).

The procedure consists in stacking daily cross-correlations between
pairs of stations, from *FIRSTDAY* to *LASTDAY* and, in each given day,
rejecting stations whose data fill is < *MINFILL*. Define a subset of
stations to cross-correlate in *CROSSCORR_STATIONS_SUBSET* (or let it
empty to cross-correlate all stations). Define a list of locations to
skip in *CROSSCORR_SKIPLOCS*, if any. The cross-correlations are
calculated between -/+ *CROSSCORR_TMAX* seconds.

Several pre-processing steps are applied to the daily seismic waveform
data, before the daily cross-correlation is calculated and stacked:

(1) removal of the instrument response, the mean and the trend;

(2) band-pass filter between *PERIODMIN* and *PERIODMAX* sec

(3) down-sampling to sampling step = *PERIOD_RESAMPLE* sec

(4) time-normalization:

    - if *ONEBIT_NORM* = False, normalization of the signal by its
      (smoothed) absolute amplitude in the earthquake period band,
      defined as *PERIODMIN_EARTHQUAKE* - *PERIODMIN_EARTHQUAKE* sec.
      The smoothing window is *PERIODMAX_EARTHQUAKE* / 2;

    - if *ONEBIT_NORM* = False, one-bit normalization, wherein
      only the sign of the signal is kept (+1 or -1);

(5) spectral whitening of the Fourier amplitude spectrum: the Fourier
    amplitude spectrum of the signal is divided by a smoothed version
    of itself. The smoonthing window is *WINDOW_FREQ*.

Note that all the parameters mentioned above are defined in the
configuration file.

When all the cross-correlations are calculated, the script exports
several files in dir *CROSSCORR_DIR*, whose name (without extension)
is:

xcorr[_<stations of subset>]_<first year>-<last year>[_1bitnorm] ...
      _[datalesspaz][+][xmlresponse][_<suffix>]

where <suffix> is provided by the user. For example:
"xcorr_1996-2012_xmlresponse"

The files, depending on their extension, contain the following data:

- .pickle       = set of all cross-correlations (instance of
                  pscrosscorr.CrossCorrelationCollection) exported in binary
                  format with module pickle;

- .txt          = all cross-correlations exported in ascii format
                  (one column per pair);

- .stats.txt    = general information on cross-correlations in ascii
                  format: stations coordinates, number of days, inter-
                  station distance etc.

- .stations.txt = general information on the stations: coordinates,
                  nb of cross-correlations in which it appears, total
                  nb of days it has been cross-correlated etc.

- .png          = figure showing all the cross-correlations (normalized to
                  unity), stacked as a function of inter-station distance.
"""

from pysismo import pscrosscorr, pserrors, psstation
from pysismo.global_var import logger
import os
import sys
import warnings
import datetime as dt
import itertools as it
import pickle
import obspy.signal.cross_correlation
from termcolor import colored

# turn on multiprocessing to get one merged trace per station?
# to preprocess trace? to stack cross-correlations?
MULTIPROCESSING = {'merge trace': False,
                   'process trace': True,
                   'cross-corr': True}
VERBOSE = False
# how many concurrent processes? (set None to let multiprocessing module decide)
NB_PROCESSES = None
if any(MULTIPROCESSING.values()):
    import multiprocessing as mp
    mp.freeze_support()  # for Windows...

# ====================================================
# parsing configuration file to import some parameters
# ====================================================

from pysismo.psconfig import (
    MSEED_DIR, DATALESS_DIR, STATIONXML_DIR, RESP_DIR,CROSSCORR_DIR,SACPZ_DIR,
    USE_DATALESSPAZ, USE_STATIONXML,USE_COMBINATION,CROSSCORR_STATIONS_SUBSET,
    CROSSCORR_SKIPLOCS,USE_COMBINATION_RESP,CROSS_STATIONS_DELETE,
    FIRSTDAY, LASTDAY, MINFILL, FREQMIN, FREQMAX, CORNERS, ZEROPHASE, PERIOD_RESAMPLE,
    ONEBIT_NORM, FREQMIN_EARTHQUAKE, FREQMAX_EARTHQUAKE, WINDOW_TIME, WINDOW_FREQ,
    CROSSCORR_TMAX)

print("\nProcessing parameters:")
print("- dir of miniseed data: " + MSEED_DIR)
print("- dir of dataless seed data: " + DATALESS_DIR)
print("- dir of stationXML data: " + STATIONXML_DIR)
print("- dir of RESP files:" + RESP_DIR)
print("- dir of POLEZERO files:" + SACPZ_DIR)
print("- output dir: " + CROSSCORR_DIR)
print("- band-pass: {:.1f}-{:.1f} s".format(1.0 / FREQMAX, 1.0 / FREQMIN))
if ONEBIT_NORM:
    print("- normalization in time-domain: one-bit normalization")
else:
    s = ("- normalization in time-domain: "
         "running normalization in earthquake band ({:.1f}-{:.1f} s)")
    print(s.format(1.0 / FREQMAX_EARTHQUAKE, 1.0 / FREQMIN_EARTHQUAKE))
fmt = '%d/%m/%Y'
s = "- cross-correlation will be stacked between {}-{}"
print(s.format(FIRSTDAY.strftime(fmt), LASTDAY.strftime(fmt)))
subset = CROSSCORR_STATIONS_SUBSET
if subset:
    logger.info("  for stations: {}".format(', '.join(subset)))

# ========================================
# Name of output files (without extension).
# E.g., "xcorr_2000-2012_xmlresponse"
# ========================================
tstart = dt.datetime.now()

responsefrom = []
if USE_DATALESSPAZ:
    responsefrom.append('datalesspaz')
if USE_STATIONXML:
    responsefrom.append('xmlresponse')
if USE_COMBINATION:
    # Choose which one to append
    if USE_COMBINATION_RESP:
        responsefrom.append("RESPresponse")

    responsefrom.append("POLEZEROresponse")
OUTBASENAME_PARTS = [
    'xcorr',
    '-'.join(s for s in CROSSCORR_STATIONS_SUBSET) if CROSSCORR_STATIONS_SUBSET else None,
    '{}-{}'.format(FIRSTDAY.year, LASTDAY.year),
    '1bitnorm' if ONEBIT_NORM else None,
    '+'.join(responsefrom)
]
OUTFILESPATH = os.path.join(CROSSCORR_DIR, '_'.join(p for p in OUTBASENAME_PARTS if p))
msg = 'Default name of output files (without extension):\n"{}"\n'.format(OUTFILESPATH)
logger.info(msg)

# import arguments
suffix=sys.argv[1]
if suffix:
    OUTFILESPATH = '{}_{}'.format(OUTFILESPATH, suffix)
msg = 'Results will be exported to files:\n"{}"(+ extension)\n'.format(OUTFILESPATH)
logger.info(msg)

# ============
# Main program
# ============

# Reading inventories in dataless seed and/or StationXML files
dataless_inventories = []
if USE_DATALESSPAZ:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dataless_inventories = psstation.get_dataless_inventories(DATALESS_DIR,
                                                                  verbose=True)

xml_inventories = []
if USE_STATIONXML:
    xml_inventories = psstation.get_stationxml_inventories(STATIONXML_DIR,
                                                           verbose=True)

resp_file_path = []
if USE_COMBINATION:
    xml_inventories=[]
    xml_inventories = psstation.get_stationxml_inventories(STATIONXML_DIR,
                                                            verbose=False)

    # Reading RESP or SACPZ filenames into one list
    if USE_COMBINATION_RESP:
        resp_file_path = psstation.get_RESP_filelists(RESP_DIR,verbose=False)
    else:
        resp_file_path = psstation.get_SACPZ_filelists(SACPZ_DIR,verbose=False)

# Getting list of stations
# -------------------------------
# COMP: Using stationxml file to
#   support location information
# -------------------------------
stations_all = psstation.get_stations(mseed_dir=MSEED_DIR,
                                  xml_inventories=xml_inventories,
                                  dataless_inventories=dataless_inventories,
                                  database=True,
                                  startday=FIRSTDAY,
                                  endday=LASTDAY,
                                  verbose=True)

# Filter stations to delete subset of stations in *CROSS_STATIONS_DELETE*
if CROSS_STATIONS_DELETE:
    # initial stations list
    stations = []
    # check all stations wheather they are in list *CROSS_STATIONS_DELETE*
    for station in stations_all:

        Exist = station.name in CROSS_STATIONS_DELETE
        if Exist:
            logger.info(colored("Delete {} ".format(station.name),'red'))
            continue
        stations.append(station)
else:
    # Don't delete any station
    stations = stations_all

logger.info(colored("ALL {} stations".format(str(len(stations))),'green'))



# Initializing collection of cross-correlations
xc = pscrosscorr.CrossCorrelationCollection()

if VERBOSE:
    logger.info(colored("Initialization time {}".format(dt.datetime.now() - tstart),
                        'red'))
# Loop on day
nday = (LASTDAY - FIRSTDAY).days + 1
dates = [FIRSTDAY + dt.timedelta(days=i) for i in range(nday)]

# construct monthly stacking
for date in dates:

    # exporting the collection of cross-correlations after the end of each
    # processed month (allows to restart after a crash from that date)
    if date.day == 1:
        with open('{}.part.pickle'.format(OUTFILESPATH), 'wb') as f:

            msg = "Exporting cross-correlations calculated until now to: " + f.name
            logger.info(msg)
            pickle.dump(xc, f, protocol=2)

    logger.info("Processing data of day {}".format(date))

    # loop on stations appearing in subdir corresponding to current month
    date_subdir = date.strftime("%Y%m%d")
    date_stations = sorted(sta for sta in stations if date_subdir in sta.subdirs)

    # subset if stations (if provided)
    if CROSSCORR_STATIONS_SUBSET:
        date_stations = [sta for sta in date_stations
                          if sta.name in CROSSCORR_STATIONS_SUBSET]

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
            trace = pscrosscorr.get_merged_trace(station=station,
                                                 date=date,
                                                 skiplocs=CROSSCORR_SKIPLOCS,
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
            logger.error('{}.{} [{}] '.format(station.network, station.name, errmsg))
        return trace

    def preprocessed_trace(trace, response, resp_file_path=None):
        """
        Preparing func that returns processed trace: processing includes
        removal of instrumental response, band-pass filtering, demeaning,
        detrending, downsampling, time-normalization and spectral whitening
        (see pscrosscorr.preprocess_trace()'s doc)

        Function is ready to be parallelized.
        """
        if not trace:
            return
        if not (response or resp_file_path):
            return
        logger.info("Preprocessing.{}".format(trace.id))

        network = trace.stats.network
        station = trace.stats.station
        try:
            pscrosscorr.preprocess_trace(
                trace=trace,
                paz=response,
                resp_file_path=resp_file_path,
                freqmin=FREQMIN,
                freqmax=FREQMAX,
                freqmin_earthquake=FREQMIN_EARTHQUAKE,
                freqmax_earthquake=FREQMAX_EARTHQUAKE,
                corners=CORNERS,
                zerophase=ZEROPHASE,
                period_resample=PERIOD_RESAMPLE,
                onebit_norm=ONEBIT_NORM,
                window_time=WINDOW_TIME,
                window_freq=WINDOW_FREQ)
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
    t0 = dt.datetime.now()
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
                resp_file_path=resp_file_path,
                dataless_inventories=dataless_inventories,
                xml_inventories=xml_inventories)
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
            logger.error('{}.{} [{}] '.format(tr.stats.network, tr.stats.station,
                                                                errmsg))
    if VERBOSE:
        logger.info(colored("import data {}".format(dt.datetime.now() - t0), 'red'))
    t0 = dt.datetime.now()
    # =================
    # processing traces
    # =================
    if not USE_COMBINATION:
        if MULTIPROCESSING['process trace']:
            # multiprocessing turned on: one process per station
            pool = mp.Pool(NB_PROCESSES)
            traces = pool.starmap(preprocessed_trace, list(zip(traces, responses)))
            pool.close()
            pool.join()
        else:
            # multiprocessing turned off: processing stations one after another
            traces = [preprocessed_trace(tr, res)
                                 for tr, res in list(zip(traces, responses))]
    else:
        if MULTIPROCESSING['process trace']:
            #multiprocess truned on: one process per station
            logger.info("MULTIPROCESSING")
            pool = mp.Pool(NB_PROCESSES)
            traces = pool.starmap(preprocessed_trace,list(zip(traces, responses, it.repeat(resp_file_path))))
            pool.close()
            pool.join()
        else:
            traces = [preprocessed_trace(tr,responses,
                                resp_file_path=resp_file_path) for tr in traces]

    # setting up dict of current date's traces, {station: trace}
    tracedict = {s.name: trace for s, trace in zip(date_stations, traces) if trace}

    delta = (dt.datetime.now() - t0).total_seconds()
    logger.info(colored("Processed stations in {:.1f} seconds".format(delta), 'green'))

    # ==============================================
    # stacking cross-correlations of the current day
    # ==============================================

    if len(tracedict) < 2:
        logger.error("No cross-correlation for this day")
        continue
    if VERBOSE:
        logger.info(colored("trace preprocess {}".format(dt.datetime.now() - t0),
                            'red'))

    t0 = dt.datetime.now()
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
            logger.info('{}-{} '.format(s1, s2))
            shift = int(CROSSCORR_TMAX / PERIOD_RESAMPLE)
            xcorr = obspy.signal.cross_correlation.correlate(
                tr1, tr2, shift=shift)
            return xcorr

        pairs = list(it.combinations(sorted(tracedict.items()), 2))
        pool = mp.Pool(NB_PROCESSES)
        xcorrs = pool.map(xcorr_func, pairs)
        pool.close()
        pool.join()
        xcorrdict = {(s1, s2): xcorr for ((s1, _), (s2, _)), xcorr in zip(pairs, xcorrs)}

    logger.info("Stacking cross-correlations")
    xc.add(tracedict=tracedict,
           stations=stations,
           xcorr_tmax=CROSSCORR_TMAX,
           xcorrdict=xcorrdict,
           verbose=not MULTIPROCESSING['cross-corr'])

    delta = (dt.datetime.now() - t0).total_seconds()
    msg = "Calculated and stacked cross-correlations in {:.1f} seconds".format(delta)
    logger.info(msg)

t0 = dt.datetime.now()
# exporting cross-correlations
if not xc.pairs():
    logger.info("No cross-correlation could be calculated: nothing to export!")
else:
    # exporting to binary and ascii files
    xc.export(outprefix=OUTFILESPATH, stations=stations, verbose=True)

logger.info(colored("After stacking process {}".format(dt.datetime.now() - t0), 'red'))
logger.info(colored("Second / month * pair {}".format(dt.datetime.now() - tstart), 'red'))
# removing file containing periodical exports of cross-corrs
try:
    os.remove('{}.part.pickle'.format(OUTFILESPATH))
except:
    pass
