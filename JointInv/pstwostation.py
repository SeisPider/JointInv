#!/usr/bin/env python
#-*- coding:utf8 -*-
"""
Scripts for measuring dispersion curves with two-station method

Procedures:
===========
"""


class StasCombine(object):
    """
    Class holds imformation of matched stations and earthquakes
    which contains:
    - pairs of stations
    - pairs of waveforms
    """

    def __init__(self, sta1, sta2, event):
        # import fundamental information
        self.event = event
        self.sta1 = sta1
        self.sta2 = sta2
        # import waveform
        self.get_merged_traces()
        # calculate dist
        self.dist = self.sta1.dist(self.sta2)
        self.id = '{0}.{1}-{2}.{3}-{4}'.format(self.sta1.network, self.sta1.name,
                                               self.sta2.network, self.sta2.name,
                                               self.event['origin'].strftime(
                                               "%Y%m%d%H%M%S"))
        # debug
        logger.debug("Dist -> {}".format(self.dist))
        logger.debug("Traces -> {}-{}".format(self.tr1.id, self.tr2.id))
        logger.debug("CombinationId -> {}".format(self.id))
    def __repr__(self):
        """
        e.g. <BL.10.NUPB <-> BL.00.NUPB <-> 2014050112013>
        """
        return '<Combination {0}.{1} <-> {2}.{3} <-> {4}>'.format(
            self.sta1.network, self.sta1.name, self.sta2.network,
            self.sta2.name, self.event['origin'].strftime("%Y%m%d%H%M%S"))
