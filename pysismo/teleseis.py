#! /usr/bin/python -u
# -*- coding:utf-8 -*-
"""
Module for extracting dispersion curves of Rayleigh wave from teleseismic records

"""
from .psstation import Station
from . import trimmer, psutils
from .global_var import logger
from .psconfig import (STATIONINFO_DIR, CATALOG_DIR, EQWAVEFORM_DIR, FIRSTDAY,
                       LASTDAY, NETWORKS_SUBSET, CHANNELS_SUBSET)
import os
from copy import copy
import itertools as it
import numpy as np

def get_catalog(catalog=CATALOG_DIR, fstday=FIRSTDAY, endday=LASTDAY):
    """
    Import catalog

    :type catalog str or path-like object
    :parm catalog indicates directory of file containing events info
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


def scan_stations(dbdir=STATIONINFO_DIR, sacdir=EQWAVEFORM_DIR,
                  networks=NETWORKS_SUBSET, channels=CHANNELS_SUBSET,
                  fstday=FIRSTDAY, endday=LASTDAY, coord_tolerance=1E-4):
    """
    Import stations

    :type sacdir str or path-like object
    :parm sacdir indicates directory of trimmed SAC files
    :type networks list
    :parm networks indicates networks' code to be calculated
    :type channels list
    :parm channels indicates channels' code to be calculated
    :type firstday `~obspy.UTCDateTime`
    :parm firstday is first day to find the stations
    :type lastday `~obspy.UTCDateTime`
    :parm lastday is last day to find the stations
    """
    logger.info("scanning stations under dir -> %s", sacdir)

    # import station database
    stationdb = trimmer.read_stations(dbdir)
    # initialization list of stations by scanning
    files = psutils.filelist(sacdir, startday=fstday, endday=endday, ext='SAC',
                             subdirs=True)

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
