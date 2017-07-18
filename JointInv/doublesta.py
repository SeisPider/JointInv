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
from obspy import UTCDateTime
import numpy as np
from itertools import combinations

from JointInv.global_var import logger
from JointInv.distaz import DistAz
import JointInv.psutils

class TsEvnt(object):
    """
    Hashable class holding imformation of matched stations and earthquakes
    which contains:
    - a pair of stations
    - a pair of sets of locations
    - a pair of ids
    """
    def __init__(self, catalog):
        # import catalog
        self.events = self._read_catalog(catalog)

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
            return {'sta1': sta2, 'sta2': sta1, 'event': event,
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
        2.
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
                self.tsevndata.append({pair:{'sta1':procetr1, 'sta2':procetr2}})


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
    JointInv.psutils.resample = (tr, period_resample)

    df = tr.stats.sampling_rate
    # add a function to Read SACPZ file
    tr.data = simulate_seismometer(tr.data, df,
                                   paz_remove=paz_remove, pre_filt=pre_filt)
    tr.detrend(type='constant')
    tr.detrend(type='linear')
    return tr

def Get_paz_remove(filename=None):
    """
    Scan files from the XJ and return dicts fit for paz_remove

    filename @str : full director and name of response file
    instrument @dict: dict of sensitivity,poles,zeros and gain
    """
    XJ_file = open(filename,'r')
    All_info = XJ_file.readlines()

    poles = []
    zeros = []
    for line in All_info:

        line_str = "".join(line)
        # obtain sensitivity
        if re.search("SENSITIVITY",line_str):
            sensitivity = float( "".join( line_str.split()[1:2] ) )

        # obtain gain
        if re.search("AO",line_str):
            gain = float( "".join( line_str.split()[1:2] ) )

        # obtain poles
        if re.search('POLE_\d',line_str):
            real_part = float( "".join( line_str.split()[1:2] ) )
            imag_part = float( "".join( line_str.split()[2:3] ) )
            poles.append((complex(real_part,imag_part)))

        # obtain zeros
        if re.search('ZERO_\d',line_str):
            real_part = float( "".join( line_str.split()[1:2] ) )
            imag_part = float( "".join( line_str.split()[2:3] ) )
            zeros.append(complex(real_part,imag_part))

    instrument = {'gain': gain, 'poles': poles, 'sensitivity': sensitivity,
                  'zeros': zeros}
    XJ_file.close()

    return instrument






