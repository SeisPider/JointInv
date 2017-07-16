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





