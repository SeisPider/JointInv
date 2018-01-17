"""
Module that parses global parameters from a configuration
file at first import, to make them available to the other
parts of the program.
"""

import configparser
import os
import glob
import json
import datetime as dt
import numpy as np
from os.path import dirname, join

from . import Bunch


def select_and_parse_config_file(basedir='./', ext='cnf', verbose=True):
    """
    Reads a configuration file and returns an instance of ConfigParser:

    First, looks for files in *basedir* with extension *ext*.
    Asks user to select a file if several files are found,
    and parses it using ConfigParser module.

    @rtype: L{ConfigParser.ConfigParser}
    """
    config_files = glob.glob(os.path.join(basedir, '*.{}'.format(ext)))

    if not config_files:
        raise Exception("No configuration file found!")

    if len(config_files) == 1:
        # only one configuration file
        config_file = config_files[0]
    else:
        print("Select a configuration file:")
        for i, f in enumerate(config_files, start=1):
            print("{} - {}".format(i, f))
        res = int(input(''))
        config_file = config_files[res - 1]

    if verbose:
        print("Reading configuration file: {}".format(config_file))

    conf = configparser.ConfigParser()
    conf.read(config_file)

    return conf

# ==========================
# parsing configuration file
# ==========================


def get_global_param(configdirname, ext='cnf', verbose=True):
    """
    Returns global parameters with a Dictionary-like object
    """

    config = select_and_parse_config_file(basedir=configdirname,
                                          ext='cnf', verbose=True)

    # -----
    # paths
    # -----

    # input dirs
    MSEED_DIR = config.get('paths', 'MSEED_DIR')
    STATIONXML_DIR = config.get('paths', 'STATIONXML_DIR')
    DATALESS_DIR = config.get('paths', 'DATALESS_DIR')
    RESP_DIR = config.get('paths', 'RESP_DIR')
    SACPZ_DIR = config.get('paths', 'SACPZ_DIR')
    ALTERNATIVE_SACPZ_DIR = config.get('paths', 'ALTERNATIVE_SACPZ_DIR')
    STATIONINFO_DIR = config.get('paths', 'STATIONINFO_DIR')
    CATALOG_DIR = config.get('paths', 'CATALOG_DIR')

    # output dirs
    CROSSCORR_DIR = config.get('paths', 'CROSSCORR_DIR')
    #TRIMMER_OUTPUT_DIR = config.get('paths', 'TRIMMER_OUTPUT_DIR')
    DATASET_DIR = config.get('paths', 'DATASET_DIR')
    TELESEISMIC_DISPERSION_DIR = config.get(
        'paths', 'TELESEISMIC_DISPERSION_DIR')

    # dir of the Computer Programs in Seismology (can be None)
    COMPUTER_PROGRAMS_IN_SEISMOLOGY_DIR = config.get('paths',
                                                     'COMPUTER_PROGRAMS_IN_SEISMOLOGY_DIR')

    # --------------------------------------
    # cross-correlation / spectra parameters
    # --------------------------------------

    # use dataless files or stationXML files to remove instrument response?
    USE_DATALESSPAZ = config.getboolean('cross-correlation', 'USE_DATALESSPAZ')
    USE_STATIONXML = config.getboolean('cross-correlation', 'USE_STATIONXML')
    USE_RESPONSE_SPIDER = config.getboolean('cross-correlation',
                                            'USE_RESPONSE_SPIDER')

    # subset of stations to cross-correlate
    CROSSCORR_STATIONS_SUBSET = config.get('cross-correlation',
                                           'CROSSCORR_STATIONS_SUBSET')
    CROSSCORR_STATIONS_SUBSET = json.loads(CROSSCORR_STATIONS_SUBSET)

    CROSS_STATIONS_DELETE = config.get('cross-correlation',
                                       'CROSS_STATIONS_DELETE')
    CROSS_STATIONS_DELETE = json.loads(CROSS_STATIONS_DELETE)

    NETWORKS_SUBSET = config.get('cross-correlation', 'NETWORKS_SUBSET')
    NETWORKS_SUBSET = json.loads(NETWORKS_SUBSET)

    CHANNELS_SUBSET = config.get('cross-correlation', 'CHANNELS_SUBSET')
    CHANNELS_SUBSET = json.loads(CHANNELS_SUBSET)
    # locations to skip
    CROSSCORR_SKIPLOCS = json.loads(config.get('cross-correlation',
                                               'CROSSCORR_SKIPLOCS'))

    # first and last day, minimum data fill per day
    FIRSTDAY = config.get('cross-correlation', 'FIRSTDAY')
    FIRSTDAY = dt.datetime.strptime(FIRSTDAY, '%d/%m/%Y').date()
    LASTDAY = config.get('cross-correlation', 'LASTDAY')
    LASTDAY = dt.datetime.strptime(LASTDAY, '%d/%m/%Y').date()
    MINFILL = config.getfloat('cross-correlation', 'MINFILL')

    # band-pass parameters
    PERIODMIN = config.getfloat('cross-correlation', 'PERIODMIN')
    PERIODMAX = config.getfloat('cross-correlation', 'PERIODMAX')
    FREQMIN = 1.0 / PERIODMAX
    FREQMAX = 1.0 / PERIODMIN
    CORNERS = config.getint('cross-correlation', 'CORNERS')
    ZEROPHASE = config.getboolean('cross-correlation', 'ZEROPHASE')
    # resample period (to decimate traces, after band-pass)
    PERIOD_RESAMPLE = config.getfloat('cross-correlation', 'PERIOD_RESAMPLE')

    # Time-normalization parameters:
    ONEBIT_NORM = config.getboolean('cross-correlation', 'ONEBIT_NORM')
    # earthquakes period bands
    PERIODMIN_EARTHQUAKE = config.getfloat(
        'cross-correlation', 'PERIODMIN_EARTHQUAKE')
    PERIODMAX_EARTHQUAKE = config.getfloat(
        'cross-correlation', 'PERIODMAX_EARTHQUAKE')
    FREQMIN_EARTHQUAKE = 1.0 / PERIODMAX_EARTHQUAKE
    FREQMAX_EARTHQUAKE = 1.0 / PERIODMIN_EARTHQUAKE
    # time window (s) to smooth data in earthquake band
    # and calculate time-norm weights
    WINDOW_TIME = 0.5 * PERIODMAX_EARTHQUAKE

    # frequency window (Hz) to smooth ampl spectrum
    # and calculate spect withening weights
    WINDOW_FREQ = config.getfloat('cross-correlation', 'WINDOW_FREQ')

    # Max time window (s) for cross-correlation
    CROSSCORR_TMAX = config.getfloat('cross-correlation', 'CROSSCORR_TMAX')

    # Min and Max surface wave velocity in trimmer
    VELOMAX = config.getfloat('cross-correlation', 'VELOMAX')
    VELOMIN = config.getfloat('cross-correlation', 'VELOMIN')

    # default parameters to define the signal and noise windows used to
    # estimate the SNR:
    # - the signal window is defined according to a min and a max velocity as:
    #   dist/vmax < t < dist/vmin
    # - the noise window has a fixed size and starts after a fixed trailing
    #   time from the end of the signal window
    PERIOD_BANDS = json.loads(config.get('FTAN', 'PERIOD_BANDS'))

    SIGNAL_WINDOW_VMIN = config.getfloat('FTAN', 'SIGNAL_WINDOW_VMIN')
    SIGNAL_WINDOW_VMAX = config.getfloat('FTAN', 'SIGNAL_WINDOW_VMAX')
    SIGNAL2NOISE_TRAIL = config.getfloat('FTAN', 'SIGNAL2NOISE_TRAIL')
    NOISE_WINDOW_SIZE = config.getfloat('FTAN', 'NOISE_WINDOW_SIZE')

    # smoothing parameter of FTAN analysis
    FTAN_ALPHA = config.getfloat('FTAN', 'FTAN_ALPHA')

    # periods and velocities of FTAN analysis
    RAWFTAN_PERIODS_STARTSTOPSTEP = config.get(
        'FTAN', 'RAWFTAN_PERIODS_STARTSTOPSTEP')
    RAWFTAN_PERIODS_STARTSTOPSTEP = json.loads(RAWFTAN_PERIODS_STARTSTOPSTEP)
    RAWFTAN_PERIODS = np.arange(*RAWFTAN_PERIODS_STARTSTOPSTEP)

    CLEANFTAN_PERIODS_STARTSTOPSTEP = config.get(
        'FTAN', 'CLEANFTAN_PERIODS_STARTSTOPSTEP')
    CLEANFTAN_PERIODS_STARTSTOPSTEP = json.loads(
        CLEANFTAN_PERIODS_STARTSTOPSTEP)
    CLEANFTAN_PERIODS = np.arange(*CLEANFTAN_PERIODS_STARTSTOPSTEP)

    FTAN_VELOCITIES_STARTSTOPSTEP = config.get(
        'FTAN', 'FTAN_VELOCITIES_STARTSTOPSTEP')
    FTAN_VELOCITIES_STARTSTOPSTEP = json.loads(FTAN_VELOCITIES_STARTSTOPSTEP)
    FTAN_VELOCITIES = np.arange(*FTAN_VELOCITIES_STARTSTOPSTEP)
    FTAN_VELOCITIES_STEP = FTAN_VELOCITIES_STARTSTOPSTEP[2]

    # relative strength of the smoothing term in the penalty function that
    # the dispersion curve seeks to minimize
    STRENGTH_SMOOTHING = config.getfloat('FTAN', 'STRENGTH_SMOOTHING')

    # replace nominal frequancy (i.e., center frequency of Gaussian filters)
    # with instantaneous frequency (i.e., dphi/dt(t=arrival time) with phi the
    # phase of the filtered analytic signal), in the FTAN and dispersion curves?
    # See Bensen et al. (2007) for technical details.
    USE_INSTANTANEOUS_FREQ = config.getboolean(
        'FTAN', 'USE_INSTANTANEOUS_FREQ')

    # if the instantaneous frequency (or period) is used, we need to discard bad
    # values from instantaneous periods. So:
    # - instantaneous periods whose relative difference with respect to
    #   nominal period is greater than ``MAX_RELDIFF_INST_NOMINAL_PERIOD``
    #   are discarded,
    # - instantaneous periods lower than ``MIN_INST_PERIOD`` are discarded,
    # - instantaneous periods whose relative difference with respect to the
    #   running median is greater than ``MAX_RELDIFF_INST_MEDIAN_PERIOD`` are
    #   discarded; the running median is calculated over
    #   ``HALFWINDOW_MEDIAN_PERIOD`` points to the right and to the left
    #   of each period.

    MAX_RELDIFF_INST_NOMINAL_PERIOD = config.getfloat('FTAN',
                                                      'MAX_RELDIFF_INST_NOMINAL_PERIOD')
    MIN_INST_PERIOD = config.getfloat('FTAN', 'MIN_INST_PERIOD')
    HALFWINDOW_MEDIAN_PERIOD = config.getint(
        'FTAN', 'HALFWINDOW_MEDIAN_PERIOD')
    MAX_RELDIFF_INST_MEDIAN_PERIOD = config.getfloat('FTAN',
                                                     'MAX_RELDIFF_INST_MEDIAN_PERIOD')

    # --------------------------------
    # Tomographic inversion parameters
    # --------------------------------

    # Default parameters related to the velocity selection criteria

    # min spectral SNR to retain velocity
    MINSPECTSNR = config.getfloat('tomography', 'MINSPECTSNR')
    # min spectral SNR to retain velocity if no std dev
    MINSPECTSNR_NOSDEV = config.getfloat('tomography', 'MINSPECTSNR_NOSDEV')
    # max sdt dev (km/s) to retain velocity
    MAXSDEV = config.getfloat('tomography', 'MAXSDEV')
    # min nb of trimesters to estimate std dev
    MINNBTRIMESTER = config.getint('tomography', 'MINNBTRIMESTER')
    # max period = *MAXPERIOD_FACTOR* * pair distance
    MAXPERIOD_FACTOR = config.getfloat('tomography', 'MAXPERIOD_FACTOR')

    # Default internode spacing of grid
    LONSTEP = config.getfloat('tomography', 'LONSTEP')
    LATSTEP = config.getfloat('tomography', 'LATSTEP')

    # Default correlation length of the smoothing kernel:
    # S(r,r') = exp[-|r-r'|**2 / (2 * correlation_length**2)]
    CORRELATION_LENGTH = config.getfloat('tomography', 'CORRELATION_LENGTH')

    # Default strength of the spatial smoothing term (alpha) and the
    # weighted norm penalization term (beta) in the penalty function
    ALPHA = config.getfloat('tomography', 'ALPHA')
    BETA = config.getfloat('tomography', 'BETA')

    # Default parameter in the damping factor of the norm penalization term,
    # such that the norm is weighted by exp(- lambda_*path_density)
    # With a value of 0.15, penalization becomes strong when path density < ~20
    # With a value of 0.30, penalization becomes strong when path density < ~10
    LAMBDA = config.getfloat('tomography', 'LAMBDA')
    return Bunch(
        mseed_dir=MSEED_DIR, stationxml_dir=STATIONXML_DIR,
        dataless_dir=DATALESS_DIR, resp_dir=RESP_DIR,
        sacpz_dir=SACPZ_DIR, alternative_sacpz=ALTERNATIVE_SACPZ_DIR,
        stationinfo_dir=STATIONINFO_DIR,
        crosscorr_dir=CROSSCORR_DIR,
        teleseismic_dispersion_dir=TELESEISMIC_DISPERSION_DIR,
        #trimmer_output_dir=TRIMMER_OUTPUT_DIR,
        dataset_dir=DATASET_DIR,
        catalog_dir=CATALOG_DIR,

        cpspath=COMPUTER_PROGRAMS_IN_SEISMOLOGY_DIR,
        use_datalesspaz=USE_DATALESSPAZ, use_stationxml=USE_STATIONXML,
        use_response_spider=USE_RESPONSE_SPIDER,
        crosscorr_stations_subset=CROSSCORR_STATIONS_SUBSET,
        cross_stations_delete=CROSS_STATIONS_DELETE,
        network_subset=NETWORKS_SUBSET,
        channel_subset=CHANNELS_SUBSET,
        crosscorr_skiplocs=CROSSCORR_SKIPLOCS,
        fstday=FIRSTDAY,
        endday=LASTDAY, minfill=MINFILL, freqmin=FREQMIN,
        freqmax=FREQMAX, corners=CORNERS, zerophase=ZEROPHASE,
        period_resample=PERIOD_RESAMPLE, onebit_norm=ONEBIT_NORM,
        freqmin_eq=FREQMIN_EARTHQUAKE,
        freqmax_eq=FREQMAX_EARTHQUAKE,
        window_time=WINDOW_TIME,
        window_freq=WINDOW_FREQ,
        crosscorr_tmax=CROSSCORR_TMAX,

        velomax=VELOMAX, velomin=VELOMIN,

        period_bands = PERIOD_BANDS,
        signal_window_vmin=SIGNAL_WINDOW_VMIN,
        signal_window_vmax=SIGNAL_WINDOW_VMAX,
        signal2noise_tail=SIGNAL2NOISE_TRAIL,
        noise_window_size=NOISE_WINDOW_SIZE,

        ftan_alpha=FTAN_ALPHA,
        rawftan_periods_startstopstep=RAWFTAN_PERIODS_STARTSTOPSTEP,
        rawftan_periods=RAWFTAN_PERIODS,
        cleanftan_periods_startstopstep=CLEANFTAN_PERIODS_STARTSTOPSTEP,
        cleanftan_periods=CLEANFTAN_PERIODS,

        ftan_velocities_startstopstep=FTAN_VELOCITIES_STARTSTOPSTEP,
        ftan_velocities=FTAN_VELOCITIES,
        ftan_velocities_step=FTAN_VELOCITIES_STEP,

        strength_smooth=STRENGTH_SMOOTHING,
        use_instantaneous_freq=USE_INSTANTANEOUS_FREQ,

        max_reldiff_inst_norminal_period=MAX_RELDIFF_INST_NOMINAL_PERIOD,
        min_inst_period=MIN_INST_PERIOD,
        halfwindow_median_period=HALFWINDOW_MEDIAN_PERIOD,
        max_reldiff_inst_median_period=MAX_RELDIFF_INST_MEDIAN_PERIOD,

        minspectsnr=MINSPECTSNR,
        minspectsnr_nosdev=MINSPECTSNR_NOSDEV,
        maxsdev=MAXSDEV,
        minnbtrimester=MINNBTRIMESTER,
        maxperiod_factor=MAXPERIOD_FACTOR,

        lonstep=LONSTEP,
        latstep=LATSTEP,
        correlation_length=CORRELATION_LENGTH,

        alpha=ALPHA,
        beta=BETA,
        lambdap=LAMBDA
    )
