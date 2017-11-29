#! /usr/bin/env python                
import sys
from os.path import basename
import numpy as np               
import matplotlib.pyplot as plt               
import obspy  
import pickle
from glob import glob
from pysismo.pscrosscorr import MonthYear, MonthCrossCorrelation 
import datetime as dt

# find saved files and stack them
def export_data(dirname="../output/cross_correlation/*.dill"):
    filelists = glob(dirname)
    for filelist in filelists:
        with open(filelist, "rb") as f:
            fundaname = basename(filelist)
            fundaname.replace(".dill", "")
            temp = dill.load(f)
            temp.export(outprefix=fundaname)


def combination(xcdict1, xcdict2):
    """
    combine two xc dict.
    """
    # import all station pairs
    pairs = xcdict1.pairs()
    for pair in pairs:
        sta1, sta2 = pair
        try:
            xc1 = xcdict1[sta1][sta2]
            xc2 = xcdict2[sta1][sta2]
        except KeyError:
            continue
        # append xc1 to monthxcs of xc1
        xc1time = xc1.startday + dt.timedelta(1)
        xc1month = MonthYear(xc1time.month, xc1time.year)
        monthxc = MonthCrossCorrelation(month=xc1month, ndata=len(xc1.dataarray))
        monthxc.dataarray = xc1.dataarray
        monthxc.nday = xc1.nday
        xc1.monthxcs.append(monthxc)
        
        # append xc2 to monthxcs of xc1
        xc2time = xc2.startday + dt.timedelta(1)
        xc2month = MonthYear(xc2time.month, xc2time.year)
        monthxc = MonthCrossCorrelation(month=xc2month, ndata=len(xc2.dataarray))
        monthxc.dataarray = xc2.dataarray
        monthxc.nday = xc2.nday
        xc1.monthxcs.append(monthxc)


        # add xc2 main ccf to xc1 and change time
        xc1.dataarray += xc2.dataarray
        xc1.startday = min(xc1.startday, xc2.startday)
        xc1.endday = max(xc1.endday, xc2.endday)

        # change time length
        xc1.nday += xc2.nday
        xcdict1[sta1][sta2] = xc1
    return xcdict1

filelists = glob("../output/cross_correlation/*.pickle")
with open(filelists[0], "rb") as f:
    # initialization and delete first file
    rawxcdicts = pickle.load(f)
    filelists.pop(0)

for filedirname in filelists:
    with open(filedirname, "rb") as f:
        # initialization and delete first file
        xcdict2 = pickle.load(f)
    rawxcdicts = combination(rawxcdicts, xcdict2)

with open("../output/cross_correlation/xcorr_2015-2017_POLEZEROresponse.pickle", 'wb') as f:
    pickle.dump(rawxcdicts, f, protocol=4)





    
