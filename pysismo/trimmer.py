#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
Script to trim earthquake event waveform from continues waveform.

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

"""
import os
import re
import copy

from obspy import read, Stream, UTCDateTime
from obspy.io.sac import SACTrace
from obspy.geodetics.base import gps2dist_azimuth
from .global_var import logger


class Trimmer(object):
    def __init__(self, stationinfo, catalog, sacdir, velomin, velomax):
        self.sacdir = sacdir
        self.stations = self._read_stations(stationinfo)
        self.events = self._read_catalog(catalog)
        self.by_speed = {"maximum":velomax, "minimum":velomin}

    def _read_catalog(self, catalog):
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
                                "magnitude": float(magnitude)}

                    event = {origindate: [subevent]}
                    events.update(event)
                else:
                    subevent = {"origin": UTCDateTime(origin),
                                "latitude": float(latitude),
                                "longitude": float(longitude),
                                "depth": float(depth),
                                "magnitude": float(magnitude)}
                    events[origindate].append(subevent)
        return events

    def _read_stations(self, stationinfo):
        """
        Read station information from station metadata file.

        Format of station information:

            NET.STA  latitude  longitude  elevation
        """
        stations = {}
        with open(stationinfo, "r") as f:
            for line in f:
                name, stla, stlo, stel = line.split()[0:4]
                station = {
                    name: {
                        "stla": float(stla),
                        "stlo": float(stlo),
                        "stel": float(stel)
                    }
                }
                stations.update(station)
        logger.info("%d stations found.", len(stations))
        return stations

    def _writesac(self, stream, event, station, outdir):
        """
        Write data with SAC format with event and station information.
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
        read trimed trace
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
        """
        # abandon influence of in-place change
        tr_copy = copy.copy(trace)

        # check the destination
        trstart = tr_copy.stats.starttime
        trend = tr_copy.stats.endtime

        tracedatestart = trstart.strftime("%Y%m%d")
        tracedateend = trend.strftime("%Y%m%d")

        events = self.events[tracedatestart] + self.events[tracedateend]
        stationid = ".".join([trace.stats.network, trace.stats.station])


        for event in events:
            st = Stream()
            dist = gps2dist_azimuth(event["latitude"], event["longitude"],
                                    self.stations[stationid]["stla"],
                                    self.stations[stationid]["stlo"])[0]
            dist_km = dist / 1000.0

            eventdir = event['origin'].strftime("%Y%m%d%H%M%S")
            outdir = os.path.join(self.sacdir, eventdir)
            if not os.path.exists(outdir):
                os.makedirs(outdir, exist_ok=True)
            # determine starttime and endtime

            starttime = event['origin'] + dist_km / self.by_speed["maximum"]
            endtime = event['origin'] + dist_km / self.by_speed["minimum"]
            # ignore some outlier
            if (starttime > trend) or (endtime < trstart):
                continue
            tr_trim = tr_copy(starttime=starttime, endtime=endtime)
            st.append(tr_trim)

            # no previous output
            if not self.read_sac(event, tr_trim.id):
                self._writesac(st, event, self.stations[stationid], outdir)
                continue
            # there is previous output
            st.append(self.read_sac(event, tr_trim.id))
            st.merge()
            self._writesac(st, event, self.stations[stationid], outdir)
            logger.info("trim data of event -- {}", event['origin'])

    def view_station(self, sta_nm):
        """
        View information of station based on provided station name
        """
        return [sta for sta in self.stations if re.search(sta_nm, sta['name'])]
