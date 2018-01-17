#! /usr/bin/env python
import numpy as np
from obspy import UTCDateTime
from os.path import join, basename
from glob import glob
import datetime as dt

from . import logger


class SourceResponse(object):
    """class to handle all response files
    """

    def __init__(self, subdir="./", source="CENC"):
        """initialize and import response files

        Parameter
        =========
        subdir : str or path-like obj.
            subdir of response files
        """
        # initialize files location
        self.subdir = subdir

        # set source code
        if source == "CENC":
            self.source = "CENC"
            suffix = "SACPZs"
            prefix = "PZs"

        # scan stations
        self.response_scanner(suffix=suffix, prefix=prefix)

    def __repr__(self):
        """representation
        """
        return "<Response files of {}>".format(self.source)

    def _cencresponse_scanner(self, suffix="SACPZs", prefix="PZs"):
        """scan response files from CENC 
        """
        # scan networks names
        networkfolders = glob(join(self.subdir, "_".join(["*", suffix])))

        def obtain_networks(folders):
            """obtain networks' name from glob list

            Parameter
            =========
            folders : list
                folders of networks
            """
            return [(basename(folder).split("_")[0], folder) for folder in folders]

        # obtain all networks
        networks = obtain_networks(networkfolders)

        self.response, self.networks = {}, []
        # obtain all network responses
        for network in networks:
            name, folder = network
            self.networks.append(name)
            self.response.update(
                {name: NetworkResponse(network, prefix=prefix)})

    def response_scanner(self, suffix="SACPZs", prefix="PZs"):
        """scan response files 
        """
        # scan networks names
        if self.source == "CENC":
            self._cencresponse_scanner(suffix=suffix, prefix=prefix)
    
    def _check_network_inornot(self, network):
        """Check if this network is included in this SourceResponse obj.
        
        Parameter
        =========
        network : str
            network name to be checked
        """
        return True if network in self.networks else False
    
    def trace_response_spider(self, tr):
        """Check the response file for this trace

        Parameter
        =========
        trace : `~obspy.Trace` obj.
            trace to find its response file 
        """
        
        # check if network belongs to this
        try:
            networkresponse = self.response[tr.stats.network]
        except KeyError:
            return None
        
        # check if response file of this trace exist?
        try:
            traceresponse = networkresponse.responses[tr.id]
        except KeyError:
            return None

        # get  response file location of this trace
        return traceresponse.get_response(tr.stats.starttime)


class NetworkResponse(object):
    """class to handle response file of an entire network
    """

    def __init__(self, network, prefix="PZs"):
        """initialization
        """
        self.network = network
        self.prefix = prefix
        self.responses = self.import_responsefiles()

    def __repr__(self):
        """representation
        """
        return "<Response files of network {}>".format(self.network[0])

    def import_responsefiles(self):
        """Initialization of a network response
        """
        # obtain files list
        name, folder = self.network
        responsefiles = glob(join(folder, "_".join([self.prefix, name, "*"])))

        response = {}
        # obtain response files
        for respfile in responsefiles:
            spliter = basename(respfile).split("_")
            if len(spliter) == 6:
                _, net, sta, cha, startt, endt = spliter
                endt = UTCDateTime(endt + "01")

            elif len(spliter) == 5:
                _, net, sta, cha, startt = spliter
                endt = UTCDateTime(dt.datetime.now())
            starttime, endtime = UTCDateTime(startt + "01"), endt

            # channel id
            trid = ".".join([net, sta, "00", cha])
            if trid not in response.keys():
                response.update({trid: TraceResponse(trid)})
            response[trid].update_periods(starttime, endtime, respfile)
        return response


class TraceResponse(object):
    """class to handle response file of a particular trace
    """

    def __init__(self, trid):
        """response class initialization of this trace
        """
        self.trace = trid
        self.periods = []

    def __repr__(self):
        """representation
        """
        return "Response file of trace {}".format(self.trace)

    def update_periods(self, starttime, endtime, filedirname):
        """obtain periods and response file of this period

        Parameter
        =========
        starttime : `~obspy.UTCDateTime`
            starttime of this period
        endtime : `~obspy.UTCDateTime`
            endtime of this period
        filedirname : str or path-like obj.
            response file dirname of at this period
        """
        self.periods.append((TimePeriod(starttime, endtime), filedirname))

    def get_response(self, time):
        """Return location of response file base on inputted time

        Parameter
        =========
        time : `~obspy.UTCDateTime`
           time to obtain response file 
        """
        timediffs = np.zeros(len(self.periods))
        for index, period in enumerate(self.periods):
            time_period, filename = period
            if time_period.includeornot(time):
                return filename
            else:
                timediffs[index] = time_period.obtain_timediff(time)

        # find a nearest time
        indexmin = timediffs.argmin()

        # if no file is finded
        logger.info("Choose Nearest time period for {}".format(time))
        return self.periods[indexmin][1]


class TimePeriod(object):
    """class to indicate a time period
    """

    def __init__(self, starttime, endtime):
        """initilization

        Parameter
        =========
        starttime : `~obspy.UTCDateTime`
            starttime of this period
        endtime : `~obspy.UTCDateTime`
            endtime of this period
        """
        self.starttime = starttime
        self.endtime = endtime

    def __repr__(self):
        """representation
        """
        return "<Time period {}-{}>".format(self.starttime.strftime("%Y%m%d"),
                                            self.endtime.strftime("%Y%m%d"))

    def includeornot(self, time):
        """To judge if time in this time period

        Parameter
        =========
        time : `~obspy.UTCDateTime`
           time to judge 
        """
        if time >= self.starttime and time <= self.endtime:
            return True
        else:
            return False

    def obtain_timediff(self, time):
        """calculate difference between time and time period

        Parameter
        =========
        time : `~obspy.UTCDateTime`
           time to calculate 
        """
        return min(abs(time - self.starttime), abs(time - self.endtime))


if __name__ == "__main__":
    sourceresponse = SourceResponse(subdir="./CENC")
    AH = sourceresponse.response['AH']
    trresp = AH.responses['AH.TOL.00.BHZ']
    trresp.get_response(UTCDateTime("20170501"))
