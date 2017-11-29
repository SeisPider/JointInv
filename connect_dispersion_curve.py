#! /usr/bin/env python                
import os, sys               
import numpy as np               
import matplotlib.pyplot as plt               
import obspy
import pickle

from JointInv import logger
from os.path import basename, join, exists

pickle_file = "../output/FTAN/FTAN_whitenedxc_minSNR=7_2015-2017_POLEZEROresponse_ALPHA20.pickle"
with open(pickle_file, 'rb') as f:
    # open file and load dispersionc curves
    curves = pickle.load(f)
    f.close()

# use imported data to do tomography method 
basename = basename(pickle_file).replace('FTAN','Ext-FTAN')
for disp in curves:
    # get teleseismic gv id
    sta1id = ".".join([disp.station1.network, disp.station1.name])
    sta2id = ".".join([disp.station2.network, disp.station2.name])
    telegvid1 = "-".join([sta1id, sta2id])
    telegvid2 = "-".join([sta2id, sta1id])
    
    # import teleseismic gv
    filename1 = join("../../tsmethod/output/selected/", 
                     ".".join([telegvid1, "telegv"]))
    filename2 = join("../../tsmethod/output/selected/", 
                     ".".join([telegvid2, "telegv"]))
    if exists(filename1):
        filename = filename1
        telegvid =  telegvid1
    elif exists(filename2):
        filename = filename2
        telegvid =  telegvid2
    else:
        filename = filename1 
        telegvid = telegvid1 
        logger.info("No Vg for {} [skipping]".format(telegvid1))
    periods = np.arange(25, 55)
    disp.insert_teleseismic(filename, periods=periods)
    logger.info("Insert Teleseismic dispersion curve from {}".format(filename))

# export data with pickle format
with open(join("../output/FTAN/", basename), mode='wb') as f:
    pickle.dump(curves, f, protocol=2)

