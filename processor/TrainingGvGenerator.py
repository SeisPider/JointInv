# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 21:07h, 05/02/2018
#        Usage: 
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2017 Xiao Xiao
#-------------------------------------------------------------------------------
"""
"""
import os
from glob import glob
import numpy as np

from os.path import join
from subprocess import call

def mftpertrace(sacfile, index, export_dir):
    """Perform mft with particular trace and handle files

    Parameter
    =========
    sacfile : str
        location of sac file
    index : int
        number of this operation
    export_dir : str
        location of exported files
    """
    # perform mft and pick data
    sys_str = "do_mft {}".format(sacfile)
    os.system(sys_str)

    # handle files
    outlierdir = join(export_dir, "outliers")
    inlinedir = join(export_dir, "inliners")
    os.makedirs(outlierdir, exist_ok=True)
    os.makedirs(inlinedir, exist_ok=True)
    
    # move files
    outlierdirname = join(outlierdir, "outlier.d.{}".format(index))
    cmdstr = "cp outlier.d {}".format(outlierdirname)
    call(cmdstr, shell=True)

    inlinerdirname = join(inlinedir, "inliner.dsp.{}".format(index))
    cmdstr = "cp *.dsp {}".format(inlinerdirname)
    call(cmdstr, shell=True)

    # delete else things
    cmdstr = "rm outlier* mft96* MFT* disp* *.dsp"
    call(cmdstr, shell=True)


if __name__ == '__main__':
    def gv_training_data():
        eventdir = glob("../data/RawTraces_china/201701*/*.SAC")
        for idx in range(100):
            sacf = eventdir[int(np.random.rand(1) * len(eventdir))]
            mftpertrace(sacf, idx, "../data/Training")
    gv_training_data()

    
