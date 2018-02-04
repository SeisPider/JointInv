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
# import standard module
from subprocess import call
import subprocess
import os, sys
from glob import glob
from os.path import join, exists, split
from shutil import copy2
from itertools import repeat

# import open-source module
from obspy.io.sac import SACTrace
from obspy import UTCDateTime
import numpy as np
from scipy import interpolate

# import self-developed lib
from JointInv.psconfig import get_global_param
from JointInv import logger

MULTIPROCESSING = False 
if MULTIPROCESSING:
    NBPROCESSES = 10
    import multiprocessing as mp
    mp.freeze_support()

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
        self.all_stacked = outfilenamedir

    def trimester_stacking(self, export_dir="./"):
        """Return trimeseter stacked ccfs

        Parameter
        =========
        export_dir : str or path-like obj.
            directory to export this stacked waveform
        """
        logger.info("Stacking trimester CCFs of %s", self.pairid)
        mmccf = self.monthccfs
        self.trimester_dirname = []
        for idx in range(1, len(mmccf) - 1):
            # summation of nearest three monthes' waveform
            to_be_stacked = [mmccf[idx - 1], mmccf[idx], mmccf[idx + 1]]
            stacked_ccf = CcfsListSummation(to_be_stacked)

            # export files
            trimester_folder = "trimester_{}".format(mmccf[idx]['yearmonth'])
            subdir = join(export_dir, trimester_folder, self.pairid)
            if not exists(subdir):
                os.makedirs(subdir, exist_ok=True)
            outfilenamedir = join(subdir, self.pairid + ".SAC")
            stacked_ccf.write(outfilenamedir)
            self.trimester_dirname.append(outfilenamedir)

    def get_final_dispesrion(self, piover4=-1.0, vmin=1.5, vmax=5, tmin=4,
                             tmax=45, tresh=20, ffact=1.0, taperl=1.0,
                             snr=0.2, fmatch=1.0, aftanloc=None,
                             refmodeldir=None):
        """Perform ftan to obtain group and phase velocity dispersion curves

        Parameter
        =========
        piover4 : float
            phase shift = piover4 * pi / 4, for cross-correlationm,
            it should be -1.0
        vmin    : float
            minimal group velocity, km/s
        vmax    : float
            maximal group velocity, km/s
        tmin    : float
            minimal period, s
        tmax    : float
            maximal period, s
        tresh   : float
            treshold for jump detection, usally 10, can be adjusted
        ffact   : float
            factor to automatic filter parameter, usally =1
        taperl  : float
            factor for the left and seismogram tapering, taper = taperl * tmax
        snr     : float
            phase match filter parameter, spectra ratio to determin cutting
            point for phase matched filter
        fmatch  : float
            factor to length of phase matching window
        aftanloc: str
            location of executable aftan file
        refmodeldir: str
            directory of the reference model
        """
        # --------------------------------------------------------------------
        # Add aftan parameter to one dict. and bounch it to self
        # --------------------------------------------------------------------
        piover4, vmin, vmax = str(piover4), str(vmin), str(vmax)
        tmin, tmax, tresh, snr = str(tmin), str(tmax), str(tresh), str(snr)
        ffact, taperl, fmatch = str(ffact), str(taperl), str(fmatch)

        self.aftanparm = {
            'piover4': piover4,
            'vmin': vmin,
            'vmax': vmax,
            'tmin': tmin,
            'tmax': tmax,
            'tresh': tresh,
            'ffact': ffact,
            'taperl': taperl,
            'snr': snr,
            'fmatch': fmatch
        }
        # --------------------------------------------------------------------
        # set parameter controling file
        # --------------------------------------------------------------------
        workwd, CcfFilename = split(self.all_stacked)
        param_str = " ".join([piover4, vmin, vmax, tmin, tmax, tresh, ffact,
                              taperl, snr, fmatch, CcfFilename, "\n"])
        AftanResult = aftan_performer(workwd, refmodeldir, CcfFilename,
                                      param_str, tmin, tmax, aftanloc=aftanloc)
        self.periods, self.gv, self.cv, self.snr = AftanResult

        # --------------------------------------------------------------------
        # Estimate uncertainty of dispersion curves based on trimesters'
        # stacked waveforms
        # --------------------------------------------------------------------
        StdResult = self._get_measure_unceratinty(refmodeldir, aftanloc)
        self.tri_gvs, self.tri_cvs, self.gvstd, self.cvstd = StdResult
        logger.debug("Finish dispersion measurement of %s", self.pairid)

        # --------------------------------------------------------------------
        # export measure phase velocity and group velocity dispersion curve
        # --------------------------------------------------------------------
        # export phase velocity
        outphdir = join(workwd, "_".join([CcfFilename, "ph"]))
        np.savetxt(outphdir, np.matrix([self.periods, self.cv, self.cvstd]).T,
                   fmt="%.5f")

        # export group velocity
        outphdir = join(workwd, "_".join([CcfFilename, "gr"]))
        np.savetxt(outphdir, np.matrix([self.periods, self.gv, self.gvstd]).T,
                   fmt="%.5f")

    def _get_measure_unceratinty(self, modeldir=None, aftanloc=None):
        """Obtain uncertainty of dispersion curve by applying FTAN to all
        trimesters and get their uncertainty distribution

        Parameter
        =========
        aftanloc: str
            location of executable aftan file
        modeldir: str
            directory of the reference model
        """
        if not hasattr(self, "aftanparm"):
            logger.error("No parameter dict. for aftan")

        cvs, gvs = [], []
        for monthfile in self.trimester_dirname:

            # set parameter controling file
            workwd, CcfFilename = split(monthfile)
            param = " ".join([self.aftanparm['piover4'], self.aftanparm['vmin'],
                              self.aftanparm['vmax'], self.aftanparm['tmin'],
                              self.aftanparm['tmax'], self.aftanparm['tresh'],
                              self.aftanparm['ffact'], self.aftanparm['taperl'],
                              self.aftanparm['snr'], self.aftanparm['fmatch'],
                              CcfFilename, "\n"])
            AftanResult = aftan_performer(workwd, modeldir, CcfFilename,
                                          param, self.aftanparm['tmin'],
                                          self.aftanparm['tmax'],
                                          aftanloc=aftanloc)
            gvs.append(AftanResult[1])
            cvs.append(AftanResult[2])
        gvs, cvs = np.array(gvs), np.array(cvs)
        gvstd, cvstd = gvs.std(axis=0), cvs.std(axis=0)
        return gvs, cvs, gvstd, cvstd


def aftan_performer(workdir, modeldir, filename, param_str, tmin,
                    tmax, aftanloc=None):
    """Perform aftan in particular directory

    Parameter
    =========
    workdir   : string
        work dir including stacked waveform
    modeldir   : string
        dir stores reference model
    filename  : string
        file name of stacked waveform stored with SAC format
    param_str : string
        aftan parameters
    aftanloc  : string
        location of aftan executable file
    """
    # change directory to where the SAC file locates
    maind = os.getcwd()
    os.chdir(workdir)

    with open("paramc.dat", "w") as f:
        f.write(param_str)

    # create the reference velocity model
    refmodel_generator(direct=modeldir, CcfFilename=filename)

    # ---------------------------------------------------------------------
    # write the parameter file and perform AFTAN with micheal's code
    # Be careful!, we should add aftan software into PATH if we ignore this
    # variable
    # ---------------------------------------------------------------------
    if not aftanloc:
        call(['aftan_c_test', "./paramc.dat"], stdout=subprocess.PIPE)
    else:
        call([aftanloc, "./paramc.dat"], stdout=subprocess.PIPE)
    logger.debug("Measured disp. of all-monthes stacked waveform")

    # obtain dispesrion curves
    dispfile = filename + "_2_DISP.1"
    normper, instper, gv, cv, _, snr = np.loadtxt(dispfile, unpack=True,
                                                  usecols=(1, 2, 3, 4, 5, 6))
    # use cubic spline interpolation to obtain dispersion

    def CubicSplineInterp(per, var, interper):
        """Return interpolated variables

        Parameter
        =========
        per : numpy array
            raw period band
        var : numpy array
            raw variable array
        interper : numpy array
            period to interpolate
        """
        tck = interpolate.splrep(per, var)
        intervar = interpolate.splev(interper, tck, der=0)
        mask = (interper <= per.max()) * (interper >= per.min())
        interped = np.array([intervar[idx] if mask[idx] else np.nan
                             for idx in range(len(intervar))])
        return interped
    # interpolation
    interpper = np.arange(float(tmin), float(tmax))
    interpgv = CubicSplineInterp(instper, gv, interpper)
    interpcv = CubicSplineInterp(instper, cv, interpper)
    interpsnr = CubicSplineInterp(instper, snr, interpper)

    # back to previous working directory
    os.chdir(maind)
    return interpper, interpgv, interpcv, interpsnr


def refmodel_generator(direct=None, CcfFilename=None, model="AK135"):
    """Generate reference model to refer in selecting phase velocity branch

    Parameter
    =========
    dicrect : str or path-like obj.
        directory of reference model
    CcfFilename : str
        filename of SAC file
    model : str
        determine model 
    """
    modelfilename = "_".join([model, "PHP"])
    modelfiledirname = join(direct, modelfilename)

    # import model
    exportname = "_".join([CcfFilename, "PHP"])
    copy2(modelfiledirname, exportname)
    logger.debug("Generate Reference Model Suc. for {}".format(CcfFilename))


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


def search_station_pairs(dirlog=None):
    """Find all station pairs included in folder

    Parameter
    =========
    dirlog : str
        log of all SAC file location
    """
    # find all SAC files
    with open(dirlog) as f:
        files = f.readlines()

    # resolve station pair info.
    pairs = {}
    for sacfile in files:
        sacfile = sacfile.strip()
        
        yearmonth, sta1, filename = sacfile.split("/")[-3:]
        pairid = filename.replace(".SAC", "")

        subdict = {
            "yearmonth": yearmonth,
            "fileloc": sacfile,
        }
        if pairid not in pairs.keys():
            pair = {

                pairid: [subdict]

            }
            pairs.update(pair)
        else:
            pairs[pairid].append(subdict)

    return pairs


if __name__ == '__main__':
    PARAM = get_global_param("../data/Configs")
    
    try:
        files = search_station_pairs(sys.argv[1])
    except:
        logger.error("Usage: python PhvPicker.py SAC_file_log\n  \
                     eg. ./sac/201701/XJ.ALS/XJ.ALS.00.BHZ_XJ.HTB.00.BHZ.SAC")
        sys.exit()

    refmodeldir = "/home/seispider/Projects/TianShan/extended/JointInv/data/Info"

    # stacking wavforms
    stacked_sac_dir = join(PARAM.crosscorr_dir, "stacked")
    def handle_sta_pair(pairid, value, stacked_sac_dir, refmodeldir):
        """Handle waveform stacking, dispersion curves measurements

        Parameter
        =========
        pairid : str
            string ID to identify a station pair
        value  : dict
            dictionay including SAC location info. of this station pair
        stacked_sac_dir : str or path-like obj.
            directory of the stacked SAC files
        refmodeldir     : str or path-like obj.
            directory of the reference model file
        """
        logger.info("Processing pair {}".format(pairid))
        try:
            stapair = InterStationCcfs(pairid=pairid, monthccfs=value)
            stapair.stacking_all_monthes(export_dir=stacked_sac_dir)
            stapair.trimester_stacking(export_dir=stacked_sac_dir)
            stapair.get_final_dispesrion(refmodeldir=refmodeldir)
        except Exception as err:
            msg = "Unhandled errors : {}".format(err)
            logger.error('{} [{}]'.format(pairid, msg))
    
    if MULTIPROCESSING:
        comblist = zip(files.keys(), files.values(), repeat(stacked_sac_dir), 
                       repeat(refmodeldir))
        with mp.Pool(NBPROCESSES) as p:
            p.starmap(handle_sta_pair, comblist)
    else:
        for key, value in files.items():
            handle_sta_pair(key, value, stacked_sac_dir, refmodeldir)


