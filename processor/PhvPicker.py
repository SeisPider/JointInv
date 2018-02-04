# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 16:46h, 03/02/2018
#        Usage: 
#               python PhvPicker.py
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2017 Xiao Xiao
#-------------------------------------------------------------------------------
"""
"""
# import modules outside
import os
from glob import glob
from os.path import join, exists
from obspy.io.sac import SACTrace
from obspy import UTCDateTime

# Import self-developed lib
from JointInv.psconfig import get_global_param
from JointInv import logger

class InterStationCcfs(object):
    """Handle CCFs stored with SAC format
    """
    def __init__(self, pairid=None, monthccfs=[]):
        """Initialization

        Parameter
        =========
        pairid : str
            id of this station pair, named as : <traceid1>_<traceid2>
        monthesccfs : list
            list of all monthesccfs
        """
        self.pairid = pairid
        self.monthccfs = monthccfs
    
    def stacking_all_monthes(self, export_dir="./"):
        """Return stacked CCFs of all monthly CCFs

        Parameter
        =========
        export_dir : str or path-like obj.
            directory to export this stacked waveform
        """
        logger.info("Stacking all monthly CCFs of %s", self.pairid)
        stacked_ccf = CcfsListSummation(self.monthccfs)

        # export this stacked waveform
        subdir = join(export_dir, "all", self.pairid)
        if not exists(subdir):
            os.makedirs(subdir, exist_ok=True)
        outfilenamedir = join(subdir, self.pairid + ".SAC")
        stacked_ccf.write(outfilenamedir)
    
    def trmester_stacking(self, export_dir="./"):
        """Return trimeseter stacked ccfs 
    
        Parameter
        =========
        export_dir : str or path-like obj.
            directory to export this stacked waveform
        """
        logger.info("Stacking trimester CCFs of %s", self.pairid)
        mmccf = self.monthccfs
        for idx in range(1, len(mmccf)-1):
            # summation of nearest three monthes' waveform
            to_be_stacked = [mmccf[idx-1], mmccf[idx], mmccf[idx+1]]
            stacked_ccf = CcfsListSummation(to_be_stacked)
            
            # export files
            trimester_folder = "trimester_{}".format(mmccf[idx]['yearmonth'])
            subdir = join(export_dir, trimester_folder, self.pairid)
            if not exists(subdir):
                os.makedirs(subdir, exist_ok=True)
            outfilenamedir = join(subdir, self.pairid + ".SAC")
            stacked_ccf.write(outfilenamedir)


def CcfsListSummation(monthccflist):
    """Return summation result of serveral monthly CCFs

    Parameter
    =========
    monthccflist : list
        monthly stacked ccf list, in which ccfs are definded as dict. including
        fileloc and yearmonth
    """
    for idx, monthccf in enumerate(monthccflist):
        # summation of two monthes' CCFs
        MonthlySac = SACTrace.read(monthccf['fileloc'])
        if idx == 0:
            stacked_ccf = MonthlySac
        else:
            stacked_ccf = SacCCFSumation(stacked_ccf, MonthlySac)
    return stacked_ccf


def SacCCFSumation(tr1, tr2):
    """Stacking two SAC formated CCFs

    Parameter
    =========
    tr1 : class `obspy.io.sac.SACTrace`
        sactrace includes one-month stacked ccf
    tr2 : class `obspy.io.sac.SACTrace`
        sactrace includes one-month stacked ccf
    """
    # TODO: check can we sum them up
    tr1.data += tr2.data
    
    # rewrite head info.
    starttime = min(UTCDateTime(tr1.kt0), UTCDateTime(tr2.kt0))
    endtime = max(UTCDateTime(tr1.kt1), UTCDateTime(tr2.kt1))
    tr1.kt0, tr1.kt1 = starttime.strftime("%Y%m%d"), endtime.strftime("%Y%m%d")
    tr1.user0 += tr2.user0
    tr1.reftime = starttime
    return tr1

def search_station_pairs(folder):
    """Find all station pairs included in folder

    Parameter
    =========
    folder : str or path-like obj.
        folder includes all CCFs
    """
    # find all SAC files
    pattern = join(folder, "*/*/*.SAC")
    files = glob(pattern)
    
    # resolve station pair info.
    pairs = {}
    for sacfile in files:
        yearmonth, sta1, filename = sacfile.split("/")[-3:]
        pairid = filename.replace(".SAC", "")
        
        subdict = {
                   "yearmonth": yearmonth,
                   "fileloc"  : sacfile,
                  }
        if pairid not in pairs.keys():
            pair = {

                    pairid:[subdict]
                   
                   }
            pairs.update(pair)
        else:
            pairs[pairid].append(subdict)

    return pairs

if __name__ == '__main__':
    PARAM = get_global_param("../data/Configs")
    SacFolder = join(PARAM.crosscorr_dir, "sac")
    files = search_station_pairs(SacFolder)
    
    # stacking wavforms
    stacked_sac_dir = join(PARAM.crosscorr_dir, "stacked")
    for key, value in files.items():
        stapair = InterStationCcfs(pairid=key, monthccfs=value)
        stapair.stacking_all_monthes(export_dir=stacked_sac_dir)
        stapair.trmester_stacking(export_dir=stacked_sac_dir)
    logger.info("Finish stacking")
