"""
Module for inter-station dispersion curve measurement based on selected rough 
group velocity dispersion curve
"""
from scipy import interpolate
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
class interstagv(object):
    """
    Class for interstation gv dispersion curve measurement 
    """
    def __init__(self, velomap1, velomap2):
        """Initialization
        """
        self.velomap1 = velomap1
        self.velomap2 = velomap2
    
    def inter_gv_measurement(self, periods=None, grid=1):
        """Measure gv
        """
        vmap1, vmap2 = self.velomap1, self.velomap2

        # interpolation
        insta1 = np.array([x.instaper for x in vmap1.disprec])
        velo1 = np.array([x.velo for x in vmap1.disprec])
        veloerr1 = np.array([x.veloerr for x in vmap1.disprec])
        
        insta2 = np.array([x.instaper for x in vmap2.disprec])
        velo2 = np.array([x.velo for x in vmap2.disprec])
        veloerr2 = np.array([x.veloerr for x in vmap2.disprec])
        if not periods:
            permin = max(insta1.min(), insta2.min())
            permax = min(insta1.max(), insta2.max())
            periods = np.arange(permin, permax, grid)
        
        dist1, dist2 = copy(vmap1.dist), copy(vmap2.dist)  
        
        arrspline1 = dist1/spline_interpolate(insta1, velo1, periods)
        arrspline2 = dist2/spline_interpolate(insta2, velo2, periods)
        
        arrmaxspline1 = dist1/spline_interpolate(insta1, velo1+veloerr1/2, periods)
        arrmaxspline2 = dist2/spline_interpolate(insta2, velo2+veloerr2/2, periods)
        
        arrminspline1 = dist1/spline_interpolate(insta1, velo1-veloerr1/2, periods)
        arrminspline2 = dist2/spline_interpolate(insta2, velo2-veloerr2/2, periods)
        # calculate dispersion
        deltadist = np.abs(dist1 - dist2)
        
        velo3 = deltadist / np.abs(arrspline1 - arrspline2)
        return insta1, insta2, periods, velo1, velo2, velo3

def spline_interpolate(x, y, splinex, fittype="spline", der=0, s=None):
    """Spline interpolation
    """
    x1, y1 = (np.array(t) for t in zip(*sorted(zip(x, y))))
    
    if fittype == "spline": 
        if not s:
            s = len(splinex) - np.sqrt(2 * len(splinex))
        tck = interpolate.splrep(x1, y1, s=s)
        result = interpolate.splev(splinex, tck, der=der)
    if fittype == "poly":
        z = np.polyfit(x1, y1, 4)
        f = np.poly1d(z)
        result = f(splinex)
    if fittype == "cubic":
        f2 = interpolate.interp1d(x1, y1, kind="cubic")
        result = f2(splinex)
    return np.array(result)

def sort_measurements(x, y):
    """Sort Measurements
    """
    x1, y1 = (np.array(t) for t in zip(*sorted(zip(x, y))))
    return x1, y1

        


