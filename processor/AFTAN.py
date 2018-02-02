#!/usr/bin/env python

# import self-developing lib
from JointInv.psconfig import get_global_param
from JointInv.psstation import Station
from JointInv.pscrosscorr import CrossCorrelation

from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
from obspy.io.sac import SACTrace
from obspy import UTCDateTime
#from copy import copy

PARAM = get_global_param("../data/Configs/")


def sac2xcorr(sacfile):
    """Transfer the sactrace to crosscorrelation class

    Parameter
    =========
    sacfile : str or path-like obj.
        dir of the sac file
    """
    # import data
    sactr = SACTrace.read(sacfile)

    # initialize stations
    coord1, coord2 = (sactr.evlo, sactr.evla, 0), (sactr.stlo, sactr.stla, 0)
    sta1 = Station(sactr.kevnm, sactr.knetwk, sactr.kcmpnm, sacfile,
                   basedir="./", coord=coord1)
    sta2 = Station(sactr.kstnm, sactr.kuser1, sactr.kuser2, sacfile,
                   basedir="./", coord=coord2)
    xcorr = CrossCorrelation(sta1, sta2, xcorr_dt=sactr.delta, xcorr_tmax=sactr.e,
                             startday=UTCDateTime(sactr.kt0), nday=sactr.user0,
                             endday=UTCDateTime(sactr.kt1), dataarray=sactr.data)
    return xcorr
# interpolate to obatint extract phase at particular time
def phase_interpolator(phase_map, periods, time_scale, dt, time, piover4=-1):
    """Return the exact phase at particular period and velocity

    Parameter
    =========
    phase_map : numpy array
        Resampled phase map changed with the velocity(time) at each period
    periods : numpy array
        Periods for FTAN analysis
    time_scale : numpy array
        Velocities scale of phase_map
    time :numpy array 
        group velocity arrival 
    """
    # Parameter initilization
    pi2, phase, interp_pha = np.pi * 2, np.zeros(len(periods)), np.zeros(3)
    dpha = np.zeros(len(periods))
    for idx, period in enumerate(np.array(periods)):
        phase_array = phase_map[idx, :]
        minidx = np.abs(time_scale - time[idx]).argmin()

        # choose five points centered with minidxth point
        interp_pha = phase_array[minidx - 5:minidx + 6]
        interp_time = time_scale[minidx - 5:minidx + 6]

        # check for 2*pi phase jump
        omega = pi2 / period
        for i in range(1, len(interp_time)):
            k = np.round(
                (interp_pha[i] - interp_pha[i-1] - omega * dt) / (pi2))
            interp_pha[i] = interp_pha[i] - k * pi2
        # interpolate with cubic
        f = interpolate.interp1d(interp_time, interp_pha, kind="slinear")
        interd = f(time[idx])
        interd += np.pi * piover4 / 4
        phase[idx] = interd


        # estimate the gradient
        print(interp_pha, interp_time)
        grad = np.gradient(interp_pha)
        f = interpolate.interp1d(interp_time, grad)
        dpha[idx] = f(time[idx])
    return phase, dpha


if __name__ == "__main__":

    # ========================================
    # Test group velocity difference between
    # personal FTAN and that of micheal
    # ========================================
    # Test transfer from SAC file to crosscorrelation class
    xcorr = sac2xcorr("./COR_TA.M14A_TA.M17A.SAC")
    PARAM.ftan_alpha *= np.sqrt(xcorr.dist()/1000)
    perArr = (1.0 / np.linspace(1/45, 1/4, 100))[::-1]

    PARAM.rawftan_periods = perArr
    PARAM.cleanftan_periods = perArr

    #print(PARAM.ftan_alpha)
    _, _, _, _, cleanpha, cleanvg, time_phase = xcorr.FTAN_complete(
        whiten=True, PARAM=PARAM)


    periods = PARAM.rawftan_periods

    # import micheal's result
    a = np.loadtxt("./COR_TA.M14A_TA.M17A.SAC_2_DISP.1")
    refmichealper, refgv, refcv = a[:, 2], a[:, 3], a[:,4]
    mask = ~np.isnan(cleanvg.v)
    tck = interpolate.splrep(periods[mask], cleanvg.v[mask], s=0)
    interped = interpolate.splev(refmichealper, tck, der=0)
    residual = (interped - refgv) / refgv
    print(residual.mean(), residual.std())
    # Result:
    #     [0.16% - 0.68%, 0.16% + 0.68%]
    # with taper

    # =======================================
    # Measure phase velocity
    # =======================================
    inst_periods = np.array([x[1] for x in cleanvg.nom2inst_periods])[mask]
    norm_periods = np.array([x[0] for x in cleanvg.nom2inst_periods])[mask]
    efficit_velo = cleanvg.v[mask]


    # initialize some parameters
    Delta = cleanvg.dist()
    
    plt.plot(inst_periods, efficit_velo, "o",  label="Observed gv")
    plt.plot(refmichealper, refgv, 'o', label="Micheal's gv")
    
    sU = 1.0 / efficit_velo
    gArr = Delta * sU
    vC = np.zeros(len(sU))
    dt = xcorr._get_xcorr_dt()

    # Obtain phases at each period
    num = -1
    time_scale = np.array([x * xcorr.dt for x in range(time_phase.shape[1])])
    pha, dpha = phase_interpolator(time_phase, norm_periods, time_scale, xcorr.dt,
                                      gArr)
    omega = dpha / dt
    # Calculate phase velocity of the last period

    # Obtain predicted velocity Vpred
    refPer, refvC = np.loadtxt("./COR_TA.M14A_TA.M17A.SAC_PHP", usecols=(0, 1),
                               unpack=True)
    tck = interpolate.splrep(refPer, refvC, s=0)
    Vpred = interpolate.splev(inst_periods[num], tck, der=0)
    
    # Measure phase velocity
    phpred = omega[num] * (gArr[num] - Delta / Vpred)
    k = round((phpred - pha[num]) / (2.0 * np.pi))
    vC[num] = Delta / (gArr[num] - (pha[num] + 2 * np.pi * k) / omega[num])
    
    for m in range(len(sU)-2, -1,-1):
        Vpred = 1/(((sU[m]+sU[m+1])*(omega[m]-omega[m+1])/2 + omega[m+1]/vC[m+1])/omega[m])
        #Vpred = vC[m+1]
        phpred = omega[m]*(gArr[m] - Delta / Vpred)
        k = round((phpred - pha[m])/(2*np.pi))
        vC[m] = Delta/(gArr[m]-(pha[m]+2*k*np.pi)/omega[m])
        print(norm_periods[m], Vpred, k, vC[m], phpred, pha[m], omega[m])
    
    
    tck = interpolate.splrep(inst_periods, vC, s=0)
    Vobs = interpolate.splev(refmichealper, tck, der=0)
    plt.plot(refmichealper, refcv, 'o', label="Micheal")
    #plt.plot(refmichealper, Vobs, 'o', label="Observed")
    #plt.plot(refmichealper, Vobs, 'o', label="Micheal")
    plt.plot(inst_periods, vC, 'o', label="Observed")
    plt.legend()
    plt.show()
