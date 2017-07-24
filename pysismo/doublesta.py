#!/usr/bin/env python
#-*- coding:utf8 -*-
"""
Scripts for measuring dispersion curves with two-station method

Procedures:
===========
    1. removal of instrumental response
    2. mft (multiple filter technique)
    3. linear-phase FIR bandpass digital filter (Kaiser window)
    4. 3-spline interpolation to transform the cross-correlation amplitude
       image to a phase velocity image.
    5. image analysis technique
"""
import os
import glob
import math
import datetime as dt

from obspy import UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.cross_correlation import xcorr
import numpy as np
from numpy.fft import rfft, irfft, fft, ifft, fftfreq
from itertools import combinations
from scipy.signal import convolve as sig_convolve
from scipy.signal import firwin
from scipy import interpolate

from .global_var import logger
from .distaz import DistAz
import .psutils
from .pscrosscorr import Get_paz_remove, FTAN

from .psconfig import (PERIOD_RESAMPLE, FTAN_ALPHA)
class TsEvnt(object):
    """
    Hashable class holding imformation of matched stations and earthquakes
    which contains:
    - a pair of stations
    - a pair of sets of locations
    - a pair of ids
    """
    def __init__(self, catalog, refdispersion):
        # import catalog
        self.events = self._read_catalog(catalog)
        self.refdisp = self._read_dispers(refdispersion)

    def matchtsevnt(self, client):
        """
        function for matching two stations and events

        Parameters:
        ===========
            client: class holding station information
        """
        self.matched_pairs = []
        for event in self.events:
            logger.info("matching for %s", event['origin'])
            eventdir = event['origin'].strftime("%Y%m%d%H%M%S")
            outdir = os.path.join(client.sacdir, eventdir)

            # check if directory and files inside exist
            if not os.path.exists(outdir) and not glob.glob(outdir + "/*"):
                logger.error("FileNotFoundError")
                continue
            self.matched_pairs.append(self.comlineselector(event, client))
    def _read_dispers(self, refdispersion):
        """
        read reference dispersion curve
        """
        if not os.path.exists(refdispersion):
            logger.error("reference dispersion curve not found")
        vc = np.loadtxt(refdispersion, skiprows=1, usecols=(4,))
        periods = np.loadtxt(refdispersion, skiprows=1, usecols=(2,))
        return (periods, vc)

    def _read_catalog(self, catalog):
        '''
        Read event catalog.

        Format of event catalog:

            origin  latitude  longitude  depth  magnitude  magnitude_type

        Example:

            2016-01-03T23:05:22.270  24.8036   93.6505  55.0 6.7  mww
        '''
        events = []
        with open(catalog) as f:
            for line in f:
                origin, latitude, longitude, depth, magnitude = line.split()[0:5]
                event = {
                    "origin": UTCDateTime(origin),
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "depth": float(depth),
                    "magnitude": float(magnitude),
                }
                events.append(event)
        return events


    def _combine_stas(self, stations):
        """
        Combine station and return station pairs

        Parameters
        ----------
        sta_pairs: list
            station pairs container
        """
        sta_pairs = list(combinations(stations, 2))
        return sta_pairs

    def _linecriterion(self, sta_pair, event, minangle, verbose=False):
        """
        Judge if station pair share the same great circle with event.
        (angle between line connecting stations and events is less than
        three degrees)

        """

        # calculate azimuth and back-azimuth
        sta1, sta2 = sta_pair
        intersta = DistAz(sta1["stla"], sta1["stlo"],
                          sta2["stla"], sta2["stlo"])
        sta2event = DistAz(sta1["stla"], sta1["stlo"],
                          event["latitude"], event["longitude"])
        # sta1-sta2-event
        if np.abs(intersta.baz - sta2event.baz) < minangle:
            if verbose:
                logger.info("Common line: %s-%s-%s", sta1['name'],
                            sta2['name'], event['origin'])
                return {'sta1': sta1, 'sta2': sta2, 'event': event,
                        'diff':(intersta.baz - sta2event.baz)}
        elif np.abs(np.abs(intersta.baz - sta2event.baz) - 180) < minangle:
            if verbose:
                logger.info("Common line: %s-%s-%s", sta2['name'],
                            sta1['name'], event['origin'])
                return {'sta1': sta2, 'sta2': sta1,
                        'event': event,
                        'diff':(np.abs(intersta.baz-sta2event.baz)-180)}
        else:
            return {}

    def comlineselector(self, event, client, minangle=3):
        """
        Select matched station pair and event

        """
        # check if stations exist
        eventdir = event['origin'].strftime("%Y%m%d%H%M%S")
        outdir = os.path.join(client.sacdir, eventdir)
        sta_list = [sta for sta in client.stations
                    if glob.glob(outdir+'/*'+sta['name']+'*')]
        sta_pairs = self._combine_stas(sta_list)
        matched_pair = [self._linecriterion(sta_pair, event, minangle,
                                            verbose=True)
                        for sta_pair in sta_pairs
                        if self._linecriterion(sta_pair, event, minangle)]

        return matched_pair
    def dataprocess(self, client):
        """
        Pre-process the traces data including follow steps:

            1. Removal of the instrument response and downsampling
        """
        return

    def preprocess_traces(self, client, period_resample=1):
        """
        Preprocess traces of two-stations event pairs

        1. remove instrument response with POLEZERO files demeaningm and
           detrending
        """
        self.tseventdata = []
        for matched_pair in self.matched_pairs:
            for pair in matched_pair:
                event = pair['event']

                # this should be removed in the future
                traceid = pair['sta1']['name'] + ".." + "?HZ"
                tr1 = client.read_sac(event, traceid)
                traceid = pair['sta2']['name'] + ".." + "?HZ"
                tr2 = client.read_sac(event, traceid)

                if not tr1 or tr2:
                    logger.error("defeatly import data")
                    continue
                # removal of instrument response of each traces
                procetr1 = trace_process(tr1, client)
                procetr2 = trace_process(tr2, client)

                if not procetr1 or procetr2:
                    logger.error("defeatly remove instrument response")
                    continue
                pairid = "-".join([pair['sta1']['name'], pair['sta2']['name']])
                self.tsevndata.append({pairid:
                                         {'sta1':procetr1, 'sta2':procetr2}
                                       })


    def isolate_rayleigh_train(self, periods, alpha=FTAN_ALPHA):
        """
        isolate rayleigh train with variable width movement window

        ref:
            huajian yao, 2004: A quick tracing method based on image amalysis
            technique for the determination of dual stations phase velocities
            dispersion curve of surface wave
        """

        # handle station pairs one by one
        for sta_pair in self.tseventdata:


def staspair_process(sta_pair, periods, alpha, shift_len=500):
    """
    estimate raw group velocity dispersion curves with FTAN and cross-correlation
    technique

    1. apply FTAN method on each traces and return amplitude time series of
       analytical signal
    2. estimate inter-station group velocity  of various periods via
       cross-correlation of analytical signal
    """
    logger.info("processing station pair %s", str(list(sta_pair.keys())[0]))
    # initialization amplitude/phase matrix: each column = amplitude
    # function of time for a given Faussian filter centered around a period
    tr1 = list(sta_pair.values())[0]['sta1']
    tr2 = list(sta_pair.values())[0]['sta2']

    # measure dispersion curves of group velocity
    reftime1, group_arr1 = measure_group_arrival(tr1, periods, alpha)
    reftime2, group_arr2 = measure_group_arrival(tr2, periods, alpha)
    if reftime1 != reftime2:
        logger.error("ReferenceTimeNotMatch")
        return None
    if not group_arr1 or not group_arr2:
        logger.error("defeatly obtain arrival of group wave")
        return None
    reftime = reftime1


    # apply isolation in time domain
    isotr1 = movement_windows_construction(tr1, periods, group_arr1, reftime1)
    isotr2 = movement_windows_construction(tr2, periods, group_arr2, reftime2)

    # filter traces separately based on kaiser windowed FIR filter
    filttr1 = kaiser_windows_filter(tr1, isotr1, periods)
    filttr2 = kaiser_windows_filter(tr2, isotr2, periods)

    # apply cross-correlation technique and transfer period-delay
    # cross-correlation functions matrix to period-velocity matrix
    vmatrix = intersta_t_v_construct(tr1, tr2, filttr1, filttr2)


    # extrct the dispersion curves

def extract_dispersion_curves(tr1, tr2, vmatrix):
    """
    extract dispersion curves from period-velocity matrix
    """


def intersta_t_v_construct(tr1, tr2, filttr1, filttr2):
    """
    construct inter-station period-velocity matrix

    :type tr1 : class:`obspy.Trace`
    :param tr1 : first trace data corresponding to this station pair
    :type tr2 : class:`obspy.Trace`
    :param tr2 : second trace data corresponding to this station pair
    :type filttr1: `numpy.array`
    :param filttr1: isolated, normalized and filtered trace data
    :type filttr2: `numpy.array`
    :param filttr2: isolated, normalized and filtered trace data
    """
    if tr1.stats.sampling_rate != tr2.stats.sampling_rate:
        logger.error("sampling rate of traces are different!")
        return None

    # calculate delay timescale
    npoints = shift_len * tr1.stats.samplinmg_rate
    timescale  = shift_len * np.arange(-npoints, npoints+1) / float(npoints)

    # calculate inter-station distance
    dist, _, _ = gps2dist_azimuth(tr1.stats.sac.stla, tr1.stats.sac.stlo,
                                  tr2.stats.sac.stla, tr2.stats.sca.stlo)
    dist /= 1000.0 # transfer meter into kilometer
    veloscale = dist / timescale

    # interested velocity scale
    intersveloscale = np.arange(rminv, rmaxv, deltav)

    cmatrix = np.zeros(shape=(len(periods), len(timescale)))
    vmatrix = np.zeros(shape=(len(periods), len(intersveloscale)))
    for iperiod, T0 in enumerate(periods):
        _, _, cmatrix[iperiod, :] = xcorr(filttr1[iperiod, :], filttr2[iperiod, :],
                                    shift_len=npoints, full_xcorr=True)
        # transfer period-shiftlen matrix into period-velocity matrix
        maskarray = (vmatrix < rmaxv) * (vmatrix > rminv)
        splvector = cmatrix[iperiod, :][maskarray]
        # normalize single cross-correlation functions
        splvector /= splvector.max()

        # interpolate cross-correlation function with cubic-spline method
        tck = interpolate.splrep(splvector, veloscale[maskarray], s=0)
        interpvector = interpolate.splev(intersveloscale, tck, der=0)
        vmatrix[iperiod, :] = interpvector

    return vmatrix

def movement_windows_construction(tr, periods, group_arr, reftime, halfwidth=3,
                                  deltav=0.01, rmaxv=7, rminv=2):
    """
    apply movement windows based on eauqtion described in Huajian Yao, 2004
    # movement window construction
    #             | 1               # tgi(Tc) - nTc < t < tgi(Tc) + nTc
    # w(t, Tc) =  | cos(pi * {abs(t-tgi(Tc))-nTc} / Tc)
    #             |                 # -Tc/2 < |t - tgi(Tc)|-nT<-Tc/2
    #             | 0               # else
    """

    w = np.zeros(shape=(len(periods), len(tr.data)))
    isotr = np.zeros(shape=(len(periods), len(tr.data)))
    timescale = [tr.stats.starttime + x / tr.stats.sampling_rate
                  x in range(tr.stats.npts)]
    for iperiod, T0 in enumerate(periods):
        #  mask array construction
        Tturna = group_arr[iperiod] - halfwidth * T0
        Tturnb = group_arr[iperiod] + halfwidth * T0
        Tjumpa = group_arr[iperiod] - halfwidth * T0 - T0 / 2
        Tjumpb = group_arr[iperiod] + halfwidth * T0 + T0 / 2
        w[iperiod,:][(timescale < Tturnb) and (timescale > Tturna)] = 1
        for index, time in enumerate(timescale):
            if (time < Tturna ) and (time > Tjumpa):
                w[iperiod, index] = np.cos(np.pi *
                  (np.abs(time - group_arr[iperiod]) - halfwidth * T0) / T0
                                          )
            if (time < Tjumpb ) and (time > Tturnb):
                w[iperiod, index] = np.cos(np.pi *
                  (np.abs(time - group_arr[iperiod]) - halfwidth * T0) / T0
                                          )
        isotr[iperiod, :] = tr.data * w[iperiod,:]
    return isotr

def kaiser_windows_filter(tr, isotr, periods, window = ('kaiser', 9.0), deltaT=1):
    """
    apply FIR filter with kaiser window
    """
    conv_result = np.zeros(shape=(len(periods), len(tr.data)))

    nyq = tr.stats.sampling_rate / 2.0
    ntaps = 2 **math.ceil(math.log(tr.stats.sac.npts, 2))
    for iperiod, T0 in enumerate(periods):
        lowcut = 1.0 / (T0 + 0.2 * deltaT / 2.0)
        highcut = 1.0 / (T0 - 0.2 * deltaT / 2.0)
        width = (highcut - lowcut) / 2.0
        window = firwin(btaps, [lowcut, highcut], width=width, window=window,
                        nyq = nyq, pass_zero=False)
        conv_result[iperiod, :] = sig_convolve(isotr[iperiod, :], b, mode='valide')
    return conv_result


def trace_process(tr, client, period_resample=1, frmin=0.0067, frcora=0.007,
                  frcorb=0.3, frmax=0.5):
    """
    Preprocess single trace including instrument response removal,
    demeaningm and detrending

    @type trace: trace
    @type resp_filist: list of strings
    @type frmin: float
    @type frcora: float
    @type frcorb: float
    @type frmax: float
    """
    trid = ".".join([tr.stats.network, tr.stats.station,
                             tr.stats.channel])

    # find resp file corresponding to this trace
    resp_file = glob.glob(os.path.join( client.responsedir, ("*."+trid) ))[0]
    # output POLEZERO file number
    if not resp_file:
        logger.error("ResponseFileNotFoundError--%s", trid)
        return None
    # deconvolution
    pre_filt = [freqmin,freqcora,freqcorb,freqmax]
    paz_remove = Get_paz_remove(filename=resp_file)
    # downsampling
    pysismo.psutils.resample = (tr, period_resample)

    df = tr.stats.sampling_rate
    # add a function to Read SACPZ file
    tr.data = simulate_seismometer(tr.data, df,
                                   paz_remove=paz_remove, pre_filt=pre_filt)
    tr.detrend(type='constant')
    tr.detrend(type='linear')

    # normalization with the maximum amplitude
    tr.data /= np.abs(tr.data).max()
    return tr


def measure_group_arrival(tr, periods, alpha):
    """
    estimate dispersion curve of group velocity and isolate the stations
    """
    # estimate the group velocity
    dt = 1.0 / tr.stats.sampling_rate
    amp, phi= FTAN(tr1.data, dt, periods, alpha)

    starttime = tr.stats.starttime
    reftime = UTCDateTime(year=tr.stats.sac.nzyear, julday=tr.stats.sac.nzjday,
                          hour=tr.stats.sac.nzhour, minute=tr.stats.sac.nzmin,
                          second=tr.stats.sac.sec,
                          microsecond=tr.stats.sac.nzmsec*1000)

    # estimate the group velocity dispersion curve with cross-correlation
    for iperiod in range(len(periods)):
        group_arrival[iperiod] = starttime + amp[iperiod].argmax() * dt
    return reftime, group_arrival


