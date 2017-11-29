#! /usr/bin/env python                
import os, sys               
import numpy as np               
import matplotlib.pyplot as plt 
from pysismo.psconfig import TOMO_DIR, DEPTHMODELS_DIR
import pickle

class oned_model(object):
    """
    class holding information of 1d oned_model
    """
    def __init__(loc, filedirname):
        """
        Initialization of 1d model contains basic info.
        """
        self.lat = loc[1]
        self.lon = loc[0]
        self.filedirname = filedirname
NB_SAMPLES = 5000 
DEPTHS = np.arange(70)
quantiles = [2.5, 97.5]
print("Loading velocity maps")
#s = ('2-pass-tomography_1996-2012_xmlresponse_3-60s_'
#     'earthquake-band=3-60s_periods=6-10s.pickle')
#PICKLE_FILE_SHORT_PERIODS = os.path.join(TOMO_DIR, s)
s = ('Ext-2-pass-tomography_whitenedxc_minSNR=7_2015-2017_POLEZEROresponse_ALPHA20.pickle')
PICKLE_FILE_LONG_PERIODS = os.path.join(TOMO_DIR, s)

#with open(PICKLE_FILE_SHORT_PERIODS, 'rb') as f:
#    VMAPS_SHORT = pickle.load(f)
with open(PICKLE_FILE_LONG_PERIODS, 'rb') as f:
    VMAPS_LONG = pickle.load(f)
PERIODVMAPS = {T: (VMAPS_LONG[T]) for T in range(8, 55)}

#PERIODVMAPS = {T: (VMAPS_SHORT[T] if T <= 10 else VMAPS_LONG[T]) for T in range(6, 26)}
PERIODS = np.array(sorted(PERIODVMAPS.keys()))
NB_BURN = min(int(NB_SAMPLES / 10), 200)


# Import models and observations
s = ('Num139.pickle')
PICKLE_FILE_MODELS = os.path.join(DEPTHMODELS_DIR, s)

#with open(PICKLE_FILE_SHORT_PERIODS, 'rb') as f:
#    VMAPS_SHORT = pickle.load(f)
with open(PICKLE_FILE_MODELS, 'rb') as f:
    vgarrays = pickle.load(f)
    sigmavg = pickle.load(f)
    vsmodels = pickle.load(f)

# depict figure 
fig, ax = plt.subplots(nrows=1, ncols=1)
vsz_arrays = [vsmodel.get_vs_at(DEPTHS) for vsmodel in vsmodels[NB_BURN:]]
vs_1stquant, vs_2ndquant = np.percentile(vsz_arrays, quantiles, axis=0)
ax.fill_betweenx(y=DEPTHS, x1=vs_1stquant, x2=vs_2ndquant, color='grey', alpha=0.3)
# mean
vsmean = np.mean(vsz_arrays, axis=0)
ax.plot(vsmean, DEPTHS, '--', color='k')
# representative model = model of posterior distribution closest to ensemble mean
i = np.argmin([np.sum((vsz - vsmean)**2) for vsz in vsz_arrays])
representative_model = vsmodels[NB_BURN:][i]
representative_model.name = 'Representative model'
representative_model.plot_model(ax=ax, color='r')
# best-fitting model
key = lambda vsmodel: vsmodel.misfit_to_vg(PERIODS, np.array(vgarrays), sigmavg)
best_model = min(vsmodels, key=key)
best_model.name = 'Best-fitting model'
best_model.plot_model(ax=ax, color='g')
plt.show()
