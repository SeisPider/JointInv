#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
Script to trim earthquake event waveform from continuous waveform.

miniSEED should be organized as:

    |--- YYYYMMDD
    |    |-- NET.STA.LOC.CHA.STARTTIME.mseed
    |    |-- NET.STA.LOC.CHA.STARTTIME.mseed
    |    `-- ...
    |--- YYYYMMDD
    |    |-- NET.STA.LOC.CHA.STARTTIME.mseed
    |    |-- NET.STA.LOC.CHA.STARTTIME.mseed
    |    `-- ...
    |--- ...

For example:

    |--- 20160101
    |    |-- NET1.STA1.00.BHE.20160101000000.mseed
    |    |-- NET1.STA1.00.BHN.20160101000000.mseed
    |    |-- NET1.STA1.00.BHZ.20160101000000.mseed
    |    `-- ...
    |--- 20160102
    |    |-- NET1.STA1.00.BHE.20160102000000.mseed
    |    |-- NET1.STA1.00.BHN.20160102000000.mseed
    |    |-- NET1.STA1.00.BHZ.20160102000000.mseed
    |    `-- ...
    |--- ...

Output SAC files are organized as:

    |--- YYYYMMDDHHMMSS
    |    |-- YYYY.JDAY.HH.MM.SS.0000.NET.STA.LOC.CHA.M.SAC
    |    |-- YYYY.JDAY.HH.MM.SS.0000.NET.STA.LOC.CHA.M.SAC
    |    `-- ...
    |--- YYYYMMDDHHMMSS
    |    |-- YYYY.JDAY.HH.MM.SS.0000.NET.STA.LOC.CHA.M.SAC
    |    |-- YYYY.JDAY.HH.MM.SS.0000.NET.STA.LOC.CHA.M.SAC
    |    `-- ...
    |--- ...

For example:


    |--- 20160103230522
    |    |-- 2016.003.23.05.22.0000.AH.ANQ.00.BHE.M.mseed
    |    |-- 2016.003.23.05.22.0000.AH.ANQ.00.BHN.M.mseed
    |    |-- 2016.003.23.05.22.0000.AH.ANQ.00.BHZ.M.mseed
    |    `-- ...
    |--- 20160105140322
    |    |-- 2016.005.14.03.22.0000.AH.ANQ.00.BHE.M.mseed
    |    |-- 2016.005.14.03.22.0000.AH.ANQ.00.BHE.M.mseed
    |    |-- 2016.005.14.03.22.0000.AH.ANQ.00.BHN.M.mseed
    |    |-- 2016.005.14.03.22.0000.AH.ANQ.00.BHZ.M.mseed
    |    `-- ...

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
"""
import os
import re
import copy

import numpy as np
from obspy import read, Stream, UTCDateTime
from obspy.io.sac import SACTrace
from obspy.geodetics.base import gps2dist_azimuth
from . import logger


class Trimmer(object):
    def __init__(self, stationinfo, catalog, sacdir, velomin, velomax):
        """
        Initialize trimmer class

        :param stationinfo: station database
        :type stationinfo : str, path-like object
        :param catalog: events database
        :type catalog : str, path-like object
        :param sacdir: output dir
        :type sacdir : str, path-like object
        :param velomin: minimum velocity of trimmed wave train
        :type velomin : int or float
        :param velomax: maximum velocity of trimmed wave train
        :type velomax : int or float
        """
        self.sacdir = sacdir
        self.stations = read_stations(stationinfo)
        self.events = read_catalog(catalog)
        self.by_speed = {"maximum": velomax, "minimum": velomin}
    
    def __repr__(self):
        """Representation of class trimmer
        """
        return "<Waveform trimmer for teleseismic surface wave>"


    def _writesac(self, stream, event, station, outdir):
        """
        Write data with SAC format with event and station information.


        :param stream: stream contains traces to be written
        :type stream :  class `~obspy.core.stream.Stream`
        :param event: dict contains information of event
        :type event :  dict
        :param station: dict contains stations' information
        :type station :  dict
        :param outdir: output dir of traimmed SAC files
        :type outdir :  str or path-like object
        """
        for trace in stream:
            # transfer obspy trace to sac trace
            sac_trace = SACTrace.from_obspy_trace(trace=trace)

            # set station related headers
            sac_trace.stla = station["stla"]
            sac_trace.stlo = station["stlo"]
            sac_trace.stel = station["stel"]

            if trace.stats.channel[-1] == "E":
                sac_trace.cmpaz = 90
                sac_trace.cmpinc = 90
            elif trace.stats.channel[-1] == "N":
                sac_trace.cmpaz = 0
                sac_trace.cmpinc = 90
            elif trace.stats.channel[-1] == "Z":
                sac_trace.cmpaz = 0
                sac_trace.cmpinc = 0
            else:
                logger.warning("Not E|N|Z component")

            # set event related headers
            sac_trace.evla = event["latitude"]
            sac_trace.evlo = event["longitude"]
            sac_trace.evdp = event["depth"]
            sac_trace.mag = event["magnitude"]

            # 1. SACTrace.from_obspy_trace automatically set Trace starttime
            #    as the reference time of SACTrace, when converting Trace to
            #    SACTrace. Thus in SACTrace, b = 0.0.
            # 2. Set SACTrace.o as the time difference in seconds between
            #    event origin time and reference time (a.k.a. starttime).
            # 3. Set SACTrace.iztype to 'io' change the reference time to
            #    event origin time (determined by SACTrace.o) and also
            #    automatically change other time-related headers
            #    (e.g. SACTrace.b).

            # 1.from_obspy_trace
            #   o
            #   |
            #   b----------------------e
            #   |=>   shift  <=|
            # reftime          |
            #               origin time
            #
            # 2.sac_trace.o = shift
            #   o:reset to be zero
            #   |
            #   b---------------------e
            #   |            |
            #   | refer(origin) time
            # -shift
            sac_trace.o = event["origin"] - sac_trace.reftime
            sac_trace.iztype = 'io'
            sac_trace.lcalda = True

            # SAC file location
            sac_flnm = ".".join([event["origin"].strftime("%Y.%j.%H.%M.%S"),
                                 "0000", trace.id, "M", "SAC"])
            sac_fullname = os.path.join(outdir, sac_flnm)
            sac_trace.write(sac_fullname)
        return

    def read_sac(self, event, traceid):
        """
        read trimmed traces in SAC format
        """
        eventdir = event['origin'].strftime("%Y%m%d%H%M%S")
        outdir = os.path.join(self.sacdir, eventdir)
        sac_flnm = ".".join([event["origin"].strftime("%Y.%j.%H.%M.%S"),
                             "0000", traceid, "M", "SAC"])
        sac_fullname = os.path.join(outdir, sac_flnm)
        try:
            trace = read(sac_fullname)
            return trace
        except:
            return None

    def get_waveform(self, trace):
        """
        Get waveform of events

        :param trace : continuous waveform containing events
        :type trace  : `~obspy.core.trace.Trace`
        """
        # abandon influence of in-place change
        tr_ori = copy.copy(trace)

        # check the destination
        trstart = tr_ori.stats.starttime
        trend = tr_ori.stats.endtime

        trdatestart = trstart.strftime("%Y%m%d")
        trdateend = trend.strftime("%Y%m%d")

        # initiation
        events = []
        if (trdatestart == trdateend) and (trdateend in self.events.keys()):
            events += self.events[trdatestart]
        else:
            if trdatestart in self.events.keys():
                events += self.events[trdatestart]
            if trdateend in self.events.keys():
                events += self.events[trdateend]
        stationid = ".".join([trace.stats.network, trace.stats.station])

        logger.debug("{} events in -> {}-{}".format(len(events), trdatestart,
                                                   trdateend))

        msg = "trdatestart-trdateend -> {}-{}".format(trdatestart, trdateend)
        logger.debug(msg)
        logger.debug("events -> \n {}".format(events))
        logger.debug("stationid -> {}".format(stationid))
        for event in events:
            st = Stream()
            # every time, we should copy trace as it is in place change
            tr_copy = copy.copy(tr_ori)


            time_list = self.stations[stationid]
            for subdict in time_list:
                starttime = subdict["starttime"]
                endtime = subdict["endtime"]
                if trstart < endtime and trstart > starttime:
                    station = subdict
            stlo, stla = station['stlo'], station['stla']
            dist = gps2dist_azimuth(event["latitude"], event["longitude"],
                                    stla, stlo)[0]
            dist_km = dist / 1000.0
            logger.debug("dist_km -> {}".format(dist_km))

            eventdir = event['origin'].strftime("%Y%m%d%H%M%S")
            outdir = os.path.join(self.sacdir, eventdir)
            if not os.path.exists(outdir):
                os.makedirs(outdir, exist_ok=True)

            # determine starttime and endtime by maximum and minimum velocity
            starttime = event['origin'] + dist_km / self.by_speed["maximum"]
            endtime = event['origin'] + dist_km / self.by_speed["minimum"]
            # ignore some outlier
            if (starttime > trend) or (endtime < trstart):
                continue
            tr_trim = tr_copy.trim(starttime=starttime, endtime=endtime)
            st.append(tr_trim)
            # no previous output
            if not self.read_sac(event, tr_trim.id):
                self._writesac(st, event, station, outdir)
                continue
            # there is previous output
            for tr_pre in self.read_sac(event, tr_trim.id):
                logger.debug(
                    "type of previous trace {}".format(tr_pre.data.dtype))
                logger.debug(
                    "type of current trace {}".format(st[0].data.dtype))
                st.append(tr_pre)
            logger.debug("Stream -> {}".format(st))
            st.merge()
            print(station)
            self._writesac(st, event, station, outdir)
            logger.info("trim data of event -> {}".format(event['origin']))

    def view_station(self, sta_nm):
        """
        View information of station based on provided station name
        """
        return [sta for sta in self.stations if re.search(sta_nm, sta['name'])]

def read_catalog(catalog):
    '''
    Read event catalog.

    Format of event catalog:
        origin  latitude  longitude  depth  magnitude  magnitude_type

    Example:

        2016-01-03T23:05:22.270  24.8036   93.6505  55.0 6.7  mww
    '''
    events = {}
    with open(catalog) as f:
        for line in f:
            origin, latitude, longitude, depth, magnitude = line.split()[
                0:5]
            origindate = UTCDateTime(origin).strftime("%Y%m%d")

            if origindate not in events.keys():
                subevent = {"origin": UTCDateTime(origin),
                            "latitude": float(latitude),
                            "longitude": float(longitude),
                            "depth": float(depth),
                            "magnitude": float(magnitude),
                            "id": UTCDateTime(origin).strftime("%Y%m%d%H%M%S")}

                event = {origindate: [subevent]}
                events.update(event)
            else:
                subevent = {"origin": UTCDateTime(origin),
                            "latitude": float(latitude),
                            "longitude": float(longitude),
                            "depth": float(depth),
                            "magnitude": float(magnitude),
                            "id": UTCDateTime(origin).strftime("%Y%m%d%H%M%S")}
                events[origindate].append(subevent)
    return events

def read_stations(stationinfo):
    """
    Read station information from station metadata file.

    Format of station information:

        NET.STA  latitude  longitude  elevation
    """
    stations = {}
    with open(stationinfo, "r") as f:
        for line in f:
            name, stla, stlo, stel, starttime, endtime = line.split()[0:6]
            
            # handle the time
            try:
                starttime = UTCDateTime(starttime)
                endtime   = UTCDateTime(endtime)
            except:
                starttime = UTCDateTime("20090101")
                endtime   = UTCDateTime("20500101")

            if name not in stations.keys():
                station = {
                            name: [{
                                    "stla": float(stla),
                                    "stlo": float(stlo),
                                    "stel": float(stel),
                                    "starttime": starttime,
                                    "endtime": endtime
                                  }]
                          }
                stations.update(station)
            else:
                subdict=  {
                            "stla": float(stla),
                            "stlo": float(stlo),
                            "stel": float(stel),
                            "starttime": starttime,
                            "endtime": endtime
                          }
                stations[name].append(subdict)

    logger.info("%d stations in database.", len(stations))
    return stations
