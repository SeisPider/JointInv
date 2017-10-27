#! /usr/bin/python -u
# -*- coding:utf-8 -*-
"""Module for teleseismic dispersion curves measurements
"""
from .psstation import Station
from . import trimmer, psutils, logger
import os
from copy import copy
import itertools as it
import numpy as np


def get_catalog(catalog, fstday, endday):
    """Import catalog

    Import catalog-like files where events are recored as 
         <Origin> <latitude> <longitude> <depth> <magnitude> <magnitude type>
         UTC Time    deg         deg        km        --            --
    e.g. 2012-01-05T01:13:40.430 -17.6910 -173.5430  35.0 5.6  mwc

    Parameters
    ----------
    catalog : str or path-like object
        catalog indicates directory of file containing events info

    fstday : datetime or UTCDateTime class
        Date for beginning to scan

    endday : datetime or UTCDateTime class
        Date for endding to scan
     """

    logger.info("scanning events in dir -> %s", catalog)
    # Import catalog
    catalogs = []
    for key, events in trimmer.read_catalog(catalog).items():

        # filter within time
        year, month, day = int(key[0:4]), int(key[4:6]), int(key[6:8])
        if [year, month, day] < [fstday.year, fstday.month, fstday.day]:
            continue
        if [year, month, day] > [endday.year, endday.month, endday.day]:
            continue

        for event in events:
            catalogs.append(event)
    return catalogs

def scan_stations(dbdir, sacdir, fstday, endday, networks=None, channels=None,
                  dbtype="raw", coord_tolerance=1E-4):
    """Import stations

    Scan stations during research time

    Parameters
    ----------
    dbdir : str or path-like object
        directory of stations information database file
    sacdir : str or path-like object
        sacdir indicates directory of trimmed/isolated SAC files
    networks : list
        networks indicates networks' code to be calculated
    channels : list
        channels indicates channels' code to be calculated
    fstday : `~obspy.UTCDateTime`
        firstday is first day to find the stations
    lastday : `~obspy.UTCDateTime`
        lastday is last day to find the stations
    dbtype: str
        database type, if fundamental rayleigh wave are inisolated, it should
        be `raw` or else `iso`
    """
    logger.info("scanning stations under dir -> %s", sacdir)

    # import station database
    stationdb = trimmer.read_stations(dbdir)
   
    
    # initialization list of stations by scanning
    if dbtype == "raw":
        # scan raw contineous waveform
        files = psutils.filelist(sacdir, startday=fstday, endday=endday,
                                 ext='SAC', subdirs=True)
    elif dbtype == "iso":
        # scan isolated waveform
        files = psutils.filelist(sacdir, startday=fstday, endday=endday,
                                 ext='SACs', subdirs=True)

    stations = []
    for f in files:
        # splitting subdir/filename
        subdir, filename = os.path.split(f)
        # subdir = e.g., 20141122130818
        yyyy, mm, dd = int(subdir[0:4]), int(subdir[4:6]), int(subdir[6:8])
        # checking that day is within selected intervals
        if fstday and (yyyy, mm, dd) < (fstday.year, fstday.month, fstday.day):
            continue
        if endday and (yyyy, mm, dd) > (endday.year, endday.month, endday.day):
            continue

        # filter with channel and network
        network, name, location, channel = filename.split(".")[-6:-2]
        if networks and network not in networks:
            continue
        if channels and channel not in channels:
            continue
        # looking for station in list
        try:
            def match(s): return [s.network, s.name, s.channel] == [
                network, name, channel]
            station = next(s for s in stations if match(s))
        except StopIteration:
            # appending new station, with current subdir
            station = Station(name=name, network=network, channel=channel,
                              filename=filename, basedir=sacdir, subdirs=[subdir])
            stations.append(station)
        else:
            # appending subdir to list of subdirs of station
            station.subdirs.append(subdir)

    logger.info('Found {0} stations'.format(len(stations)))
    # adding lon/lat of stations from inventories
    logger.info("Inserting coordinates to stations from inventories")

    for sta in copy(stations):
        # coordinates of station in database
        staid = ".".join([sta.network, sta.name])
        try:
            coords_set = [(stationdb[staid]['stlo'], stationdb[staid]['stla'],
                           stationdb[staid]['stel'])]
        except KeyError:
            coords_set = ()

        if not coords_set:
            # no coords found: removing station
            logger.warning(
                "skipping {} as no coords were found".format(repr(sta)))
            stations.remove(sta)
        elif len(coords_set) == 1:
            # one set of coords found
            sta.coord = list(coords_set)[0]
        else:
            # several sets of coordinates: calculating max diff
            lons = [lon for lon, _ in coords_set]
            lons_combinations = list(it.combinations(lons, 2))
            lats = [lat for _, lat in coords_set]
            lats_combinations = list(it.combinations(lats, 2))
            maxdiff_lon = np.abs(np.diff(lons_combinations)).max()
            maxdiff_lat = np.abs(np.diff(lats_combinations)).max()
            if maxdiff_lon <= coord_tolerance and maxdiff_lat <= coord_tolerance:
                # coordinates differences are within tolerance:
                # assigning means of coordinates
                s = ("{} has several sets of coords within "
                     "tolerance: assigning mean coordinates")
                logger.info(s.format(repr(sta)))
                sta.coord = (np.mean(lons), np.mean(lats))
            else:
                # coordinates differences are not within tolerance:
                # removing station
                s = ("WARNING: skipping {} with several sets of coords not "
                     "within tolerance (max lon diff = {}, max lat diff = {})")
                logger.info(s.format(repr(sta), maxdiff_lon, maxdiff_lat))
                stations.remove(sta)
    return stations

def common_line_judgement(event, station_pair, minangle=2.0):
    """Select matched station pair and event
    Parameters
    ----------
    event : dict
        event contain info of event ['origin', 'latitude', 'longitude', 'depth'
                                     'magnitude']
    station_pair : tuple
        contain two class Station `.psstation.Station`
    minangle : float
        the minimum angle between two station-event lines
    """
    sta1, sta2 = station_pair
    # calculate azimuth and back-azimuth
    intersta = DistAz(sta1.coord[1], sta1.coord[0],
                      sta2.coord[1], sta2.coord[0])
    sta2event = DistAz(sta1.coord[1], sta1.coord[0],
                       event["latitude"], event["longitude"])
    # sta1-sta2-event
    if np.abs(intersta.baz - sta2event.baz) < minangle:
        logger.debug("Commonline -> %s.%s-%s.%s-%s", sta1.network, sta1.name,
                     sta2.network, sta2.name, event['origin'])
        return (sta1, sta2, event)
    elif np.abs(np.abs(intersta.baz - sta2event.baz) - 180) < minangle:
        logger.debug("Commonline -> %s.%s-%s.%s-%s", sta2.network, sta2.name,
                     sta1.network, sta2.name, event['origin'])
        return (sta2, sta1, event)
    else:
        return None
