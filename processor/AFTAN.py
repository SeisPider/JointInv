#!/usr/bin/env python
import pickle
# import developing lib
from JointInv.psconfig import get_global_param
from JointInv import pscrosscorr
PARAM = get_global_param("../data/Configs/")

# import ccf data
with open("../data/output/cross_correlation/xcorr_ALS-HTB_2017-2017_sacpz_201701.pickle", "rb") as f:
    xcorrs = pickle.load(f)

# Get xcorr of station pair
xcorr = xcorrs['ALS']['HTB']
xcorr.plot_FTAN(whiten=True, PARAM=PARAM)
#xicorr.FTAN_complete(whiten=True, add_SNRs=True, vmin=)
