"""
Module for dispersion training data IO (classification).
"""
import glob
import os
import csv
from os.path import join, dirname, basename

import pandas as pd
import numpy as np
from sklearn.datasets.base import Bunch
from sklearn.tree import DecisionTreeClassifier

from .twostationgv import spline_interpolate, sort_measurements, point
from .. import logger

import matplotlib.pyplot as plt
from copy import copy


class record(object):
    """
    Class for extract info from MFT96 files
    """

    def __init__(self, recordstr, mode="raw_gv"):
        """
        Import string and master info.

        Parameters
        ==========

        mode : string
            Flag of file type. `raw_gv` for "MFT96" format file and `clean_cv`
            for  `POM96` format file.
        """
        self.raw = recordstr + "\n"

        if mode == "raw_gv":
            self._extrinfomft96(recordstr)
        if mode == "clean_cv":
            self._extrinfopom96(recordstr)
        self.mode = mode 
        # set default label [not on]
        self.label = np.array([1])

    def __repr__(self):
        """
        Representation of class record
        """
        if self.mode == "raw_gv":
            return " Raw group velocity record @{} s".format(self.period)
        if self.mode == "clean_cv":
            return " Clean phase velocity record @{} s".format(self.period)
    
    def _extrinfopom96(self, recordstr):
        """
        Extract info. in POM96 file

        Parameters
        ==========

        recordstr : string
            one-line string contains info. of possible dispersion point
        """
        splitor = recordstr.split()
        self.flag = splitor[0]
        self.wavetype = splitor[1]
        self.velotype = splitor[2]
        self.mode = splitor[3]
        self.period = float(splitor[4])
        self.velo = float(splitor[5])
        self.veloerr = float(splitor[6])
        self.stackamp = float(splitor[7])
        self.pknum = float(splitor[8])

    def _extrinfomft96(self, recordstr):
        """
        Extract info in MFT96 file
        """
        splitor = recordstr.split()
        self.flag = splitor[0]
        self.wavetype = splitor[1]
        self.velotype = splitor[2]
        self.mode = splitor[3]
        self.period = float(splitor[4])
        self.velo = float(splitor[5])
        self.veloerr = float(splitor[6])
        self.dist = float(splitor[7])
        self.azimuth = float(splitor[8])
        self.specamp = splitor[9]
        self.evla = float(splitor[10])
        self.evlo = float(splitor[11])
        self.stla = float(splitor[12])
        self.stlo = float(splitor[13])
        self.zero = splitor[14]
        self.pknum = splitor[15]
        self.instaper = float(splitor[16])

        if len(splitor) == 24:
            self.COMMENT = splitor[17]
            self.stnm = splitor[18]
            self.chnm = splitor[19]
            self.year = splitor[20]
            self.month = splitor[21]
            self.day = splitor[22]
            self.hour = splitor[23]
        elif len(splitor) == 25:
            self.upper = splitor[17]
            self.COMMENT = splitor[18]
            self.stnm = splitor[19]
            self.chnm = splitor[20]
            self.year = splitor[21]
            self.month = splitor[22]
            self.day = splitor[23]
            self.hour = splitor[24]


class velomap(object):
    """
    Class holds velocity map and corresponding method
    """

    def __init__(self, dispinfo, refdisp, trained_model=None, periodmin=25,
                 periodmax=100, line_smooth_judge=True, treshold=1,
                 digest_type="poly", velotype="raw_gv"):
        """
        Import peaks and reference dispersion curve. Extract 
        dispersion curve from this map.


        Parameters
        ==========

        trained_model : `classfier`
            classifier of decision tree from sklearn
        periodmin : `float` or `int`
            minimum period of dispersion curve
        periodmax : `float` or `int`
            maximum period of dispersion curve
        line_smooth_judge : boolean
            determine whether use line smooth condition to constrain dispersion
            curve
        treshold : `int` or `float`
            treshold * std (standard deviation) to determine acceptable region
        digest_type : string
            interpolate method used in generate possible dispersion curve
            `poly`, `cubic` and `spline`
        velotype : string
            type of velocity map `raw_gv` and `clean_cv`, indicating raw group
            velocity and clean phase velocity, separatly
        """
        self.velotype = velotype

        if velotype == "raw_gv":
            self.id = basename(dispinfo).replace(".mft96.disp", "")
            self.records = self.obtain_velomap(dispinfo, mode=velotype)
            self.rdrefgvdisp(refdisp)
            self.periodmin, self.periodmax = periodmin, periodmax

            # information extraction
            self.dist = self.records[0].dist
            self.azimuth = self.records[0].azimuth
            self.evla = self.records[0].evla
            self.evlo = self.records[0].evlo
            self.stla = self.records[0].stla
            self.stlo = self.records[0].stlo
        
        if velotype == "clean_cv":
            self.id = basename(dispinfo).replace(".pom96.dsp", "")
            self.rdrefgvdisp(refdisp)
            self.records = self.obtain_velomap(dispinfo, mode=velotype)
            self.event, self.sta1, self.sta2 = (self.id).split("-")
        
        try:
            if self.records == None:
                logger.info("No records of {}[skipping]".format(self.id))
                return None
        except ValueError:
            logger.info("Good records of {}".format(self.id))

        if trained_model:
            self.labels, self.midrec = self.classification(trained_model,
                                                            periodmin, periodmax)
        # judge outlier function should be improved later
        if line_smooth_judge:
            self.disprec = self.line_smooth_judge(treshold=treshold,
                                                  digest_type=digest_type)
        # trace back to obtain in relative short period
        self._trace_back()

    def __repr__(self):
        """
        Give out representation info
        """
        return "Dispersion Map of {}".format(self.id)

    def classification(self, trained_model, periodmin, periodmax):
        """
        Do clasification in a period-velocity map
        """
        # judge for each record
        labels = np.zeros_like(self.records)
        disprec = []

        for idx, record in enumerate(self.records):
            # judge with period limitation
            try:
                period = record.instaper
            except:
                period = record.period

            if period > periodmax or period < periodmin:
                labels[idx] = np.array([1])     # outlier
                continue

            # judge with trained model
            x_test = np.matrix([[period, record.velo]])
            record.label = trained_model.predict(x_test)
            labels[idx] = copy(record.label)

            if not record.label:
                disprec.append(record)

        return labels, np.array(disprec)

    def line_smooth_judge(self, treshold=1.96, digest_type="poly",
                          verbose=True):
        """
        Delete outliers in points selected by machine
        """
        records = copy(self.midrec)

        try:
            insta = obtain_elements(records, "instaper")
        except:
            insta = obtain_elements(records, "period")

        velo = np.array([x.velo for x in records])
        insta, velo = sort_measurements(insta, velo)

        if digest_type == digest_type:
            fitvelo = spline_interpolate(
                      insta, velo, insta, fittype=digest_type)
            residual = velo - fitvelo
            sigma = (velo - fitvelo).std()

            # detect and delete outlier
            outlierper = insta[np.abs(residual) >= treshold * sigma]
            try:
                remindrec = [rec for rec in records 
                             if rec.instaper not in outlierper]
                remineper = obtain_elements(remindrec, "instaper")
            except:
                remindrec = [rec for rec in records 
                             if rec.period not in outlierper]
                remineper = obtain_elements(remindrec, "period")
            
            # Plot determination 
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
            if verbose:
                self._plot_determine_procedure(insta, velo, remindrec, fitvelo,
                                               sigma, treshold, ax=axes[0])
            # re-check all points to find if some points are not included in 
            # machine-determined part
            try:
                allperiod = obtain_elements(self.records, "instaper")
            except:
                allperiod = obtain_elements(self.records, "period")

            allvelo = obtain_elements(self.records, "velo") 
            fitallvelo = spline_interpolate(
                         insta, velo, allperiod, fittype=digest_type)
            residual = allvelo - fitallvelo
            inlinepoint = allperiod[np.abs(residual) <= treshold * sigma]
            compliment_point = list(set(inlinepoint) - set(remineper))
            
            # Give some application
            compliment_recs = [] 
            for idx, rec in enumerate(self.records):
                try:
                    period = rec.instaper
                except:
                    period = rec.period
               
                # judge if points in 95% confidence area
                periodin = period in compliment_point
                ingreyarea = np.abs(residual[idx]) <= treshold * sigma
                if periodin and ingreyarea:
                    compliment_recs.append(rec)
            
            if verbose:
                self._plot_determine_procedure(insta, velo, remindrec, fitvelo,
                                               sigma, treshold, ax=axes[1],
                                               compliment_recs=compliment_recs)

                plt.savefig("{}.det.{}.png".format(self.id, self.velotype))
                plt.close()
        return list(set(remindrec + compliment_recs))
    
    def _plot_determine_procedure(self, insta, velo, remindrec, fitvelo,
                                  sigma, treshold, ax=None,
                                  compliment_recs=None):
        """
        Depict decision procedure
        """
        # 1. select with machine
        records = copy(self.records)
        
        try:
            allinsta = obtain_elements(records, "instaper")
            insta2 = obtain_elements(remindrec, "instaper")
        except:
            allinsta = obtain_elements(records, "period")
            insta2 = obtain_elements(remindrec, "period")

        if not ax:
            ax = plt.subplot(1,1,1)

        allvelo = obtain_elements(records, "velo")
        velo2 = obtain_elements(remindrec, "velo")
        allveloerr = obtain_elements(records, "veloerr")
        velo2err = obtain_elements(remindrec, "veloerr")

        ax.plot(insta, velo, "o", label="Machine-determined Velo.")
        ax.plot(allinsta, allvelo, "+", label="All Possible Points")

        ax.plot(self.refperiods, self.refvelo, label=" Reference Velo.")
        # 2. select with interpolation and 1 sigma region
        ax.plot(insta, fitvelo, label="Fit Velo.")
        ax.fill_between(insta, fitvelo - treshold * sigma,
                         fitvelo + treshold * sigma, facecolor="lightgrey",
                         interpolate=True,
                         label="{} sigma range".format(treshold))
        ax.plot(insta2, velo2, "o", label="Reminded")


        # compliment points
        if compliment_recs:
            try:
                complper = obtain_elements(compliment_recs, "instaper")
            except:
                complper = obtain_elements(compliment_recs, "period")
            compvelo = obtain_elements(compliment_recs, "velo")
            ax.plot(complper, compvelo, "o", label="Complimented")


        # Figure adjustion
        if compliment_recs:
            permin = min(insta.min(), insta2.min(), complper.min())
            permax = max(insta.max(), insta2.max(), complper.max())
        else:
            permin = min(insta.min(), insta2.min())
            permax = max(insta.max(), insta2.max())

        ax.set_xlim(permin, permax)

        ax.set_xlabel("Period [s]")
        ax.set_ylabel("Velocity of {} [km/s]".format(self.velotype))
        ax.set_title("{}".format(self.id))
        ax.legend()
        if not ax:
            plt.savefig("{}.det.{}.png".format(self.id, self.velotype))
            plt.close()
   
    # TODO: some outliers should be handled 
    def _trace_back(self, percent=0.1):
        """Trace beck to relative short period
        """
        records = copy(self.records)
        remindrecords = copy(self.disprec)

        # obtain points
        remindpoints = obtain_points(remindrecords)
        # obtain all points
        allpoints = obtain_points(records)

        # found out points around reference dispersion curve
        periods = np.array([x.per for x in allpoints])
        velos = np.array([x.velo for x in allpoints])

        splinevelo = spline_interpolate(self.refperiods, self.refvelo, periods)
        diff = velos - splinevelo


        # here define all the nearest points
        maskarray = np.abs(diff) <= percent * splinevelo
        nearperiods = periods[maskarray]
        nearvelos = velos[maskarray]

        nearpoints = [point for point in allpoints
                               if (point.per in nearperiods) and (point.velo
                                                                  in nearvelos)]
        self.nearpoints = nearpoints
        
        # obtain reliable points and possible lines
        reliapoints = [pt for pt in remindpoints if pt in nearpoints]
        psspoints = [pt for pt in (remindpoints + nearpoints)
                     if pt not in reliapoints]

        # outlier in reliapoints digest

        def slope_outlier(pointslist, treshold=1.96):
            reliavelos = np.array([x.velo for x in pointslist])
            slope = np.gradient(reliavelos) 
            
            slopestd, slopemean, outlier = slope.std(), slope.mean(), []
            for idx , slop in enumerate(slope):
                if np.abs(slop-slopemean) >=  treshold * slopestd and idx != 0:
                    print("Outlier: {}".format(pointslist[idx-1]))
                    outlier.append(pointslist[idx-1].velo)
            inliner = [x for x in pointslist if x.velo not in outlier]
            return inliner 
        
        #reliapoints = slope_outlier(reliapoints)
        def obtain_nearest(reliable_points, possible_points):
            velos = np.array([x.velo for x in reliapoints])
            
            rawslope = np.gradient(velos)
            mean, std = rawslope.mean(), rawslope.std()

            ptmin, ptmax = reliable_points[0], reliable_points[-1]
            lenmin = np.array([point.length(ptmin)
                               for point in possible_points])
            lenmax = np.array([point.length(ptmax)
                               for point in possible_points])
            leftedge = possible_points[lenmin.argmin()]
            rightedge = possible_points[lenmax.argmin()]

            return leftedge, rightedge
       
        psspointsnum = len(psspoints) - 1
        while (len(psspoints) > 0 and len(psspoints) != psspointsnum):
            leftpt, rightpt = obtain_nearest(reliapoints, psspoints)
            reliapoints.insert(0, leftpt)
            reliapoints.append(rightpt)
            psspointsnum = len(psspoints)
            psspoints = [pt for pt in psspoints
                         if pt.per < leftpt.per or pt.per > rightpt.per]
        
        # line fit and outlier digestion
        
        def spline_outlier(pointslist):
            """Detect outlier with curve fit
            """
            periods = np.array([pt.per for pt in pointslist])
            velos = np.array([pt.velo for pt in pointslist])

            velospline = spline_interpolate(periods, velos, periods, fittype="spline"
                                           ,s = 1)

            residual = velos - velospline
            std = residual.std()
            
            return [x for idx, x in enumerate(pointslist)
                    if np.abs(x.velo-velospline[idx]) <= 1.96 * std]
        
        reliapoints = spline_outlier(reliapoints)
        reliaperiods = np.array([pt.per for pt in reliapoints if pt.veloerr < pt.velo])
        reliavelos = np.array([pt.velo for pt in reliapoints if pt.veloerr < pt.velo])
        return reliapoints
    

    def MFT962SURF96(self, outfile, cpspath):
        """
        Export mft96 formated data to SURF96 format
        """

        # create temporary mft96 formated dispersion curve
        with open("temp.disp", 'w') as f:
            for record in self.disprec:
                f.writelines(record.raw)

        # transfer MFT96 file to SURF96 file
        excfile = join(cpspath, "MFTSRF")
        cmndstr = "{} {} > {}".format(excfile, "temp.disp", outfile)
        os.system(cmndstr)
        os.remove("temp.disp")

    def isowithsacmat96(self, surf96filepath, cpspath, srcpath=None,
                        srcsacfile=None):
        """
        Isolate SAC traces with extracted group velocity dispersion curves
        """
        if not srcsacfile:
            print(join(srcpath, "*{}*".format(self.id)))
            srcsacfile = glob.glob(
                join(srcpath, "*{}*.SAC".format(self.id)))[0]

        # isolate fundamental Rayleigh wave with sacmat96
        excfile = join(cpspath, "sacmat96")
        cmndstr = "{0} -F {1} -D {2} -AUTO".format(excfile, srcsacfile,
                                                   surf96filepath)
        os.system(cmndstr)

    def obtain_velomap(self, dispinfo, mode):
        records = []
        with open(dispinfo, "r") as f:
            lines = f.readlines()
            # extract info.
            for line in lines:
                line = line.strip()
                records.append(record(line, mode))
        if not records:
            return None
        else:
            return np.array(records)

    def rdrefgvdisp(self, refdisp):
        """Read reference dispersion file

        refdisp : string or path-like object
            dirname of reference dispersion curve
        """
        self.refperiods = np.loadtxt(refdisp, skiprows=1, usecols=2)
        if self.velotype == "raw_gv":
            self.refvelo = np.loadtxt(refdisp, skiprows=1, usecols=5)
        if self.velotype == "clean_cv":
            self.refvelo = np.loadtxt(refdisp, skiprows=1, usecols=4)

    def pltgvmap(self, permin=25, permax=100, filename=None):
        """Plot velocity map and denote selected ones 
        
        Parameter
        =========

        permin : `int` or `float`
            minimum period in plotting
        permax : `int` or `float`
            maximum period in plotting
        filename : string or path-like obj.
            give out dirname to store figure
        """
        if self.mode == "raw_gv":
            rawinstper =  obtain_elements(self.records, "instaper")
            clainstper =  obtain_elements(self.disprec, "instaper")
        if self.mode == "clean_cv":
            rawinstper =  obtain_elements(self.records, "period")
            clainstper =  obtain_elements(self.disprec, "period")

        rawgv = obtain_elements(self.records, "velo")
        rawgverr = obtain_elements(self.records, "veloerr")

        clagv = obtain_elements(self.records, "velo")
        clagverr = obtain_elements(self.disprec, "veloerr")

        xmin = permin - 1.0
        xmax = permax + 1.0
        ymin = rawgv.min() - 1.0
        ymax = rawgv.max() + 1.0

        plt.errorbar(rawinstper, rawgv, yerr=rawgverr, fmt="o",
                     label="Extreme Values")
        plt.errorbar(clainstper, clagv, yerr=clagverr, fmt="o",
                     label="Selected Points")

        plt.plot(self.refperiods, self.refgv, label="Synthetic [ak135]")
        plt.xlabel("Instaneous Period [s]")
        plt.ylabel("Group Velocity [km/s]")
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.title("Selection Map of {}".format(self.id))
        plt.legend()

        if not filename:
            plt.show()
        else:
            plt.savefig(filename)

def obtain_points(records):
    """obtain multiple points in records

    Parameter
    =========

    records : list 
        list of records
    """
    try:
        pers = obtain_elements(records, "instaper")
    except:
        pers = obtain_elements(records, "period")
    velos = obtain_elements(records, "velo")
    veloerrs = obtain_elements(records, "veloerr")

    ind = np.lexsort((velos,pers))
    pers = pers[ind]
    velos = velos[ind]
    veloerrs = veloerrs[ind]
    points = [point(pers[idx], velos[idx], veloerrs[idx])
              for idx in range(len(pers))]
    return points

def obtain_elements(records, element):
    """Return numpy array of an element in records

    Parameter
    =========

    records ： list
        list of class `record`
    element : string
        flag indicates variables types
    """
    if element == "instaper":
        return np.array([x.instaper for x in records])
    if element == "period":
        return np.array([x.period for x in records])
    if element == "velo":
        return np.array([x.velo for x in records])
    if element == "veloerr":
        return np.array([x.veloerr for x in records])




def load_disp(return_X_y=False, datasetdir=None, mode="raw_gv"):
    """Load and return the dispersion trainning data (classfication)

    Parameters
    ==========

    return_X_y : Boolean
         `True`: only return numpy array of data and target
         `False`: Return all
    datasetdir : string
         dataset dirname
    mode : string
         determine classifier type and function. Possible classifier
         1. `raw_gv` distinguish raw group velocipy dispersion curves
         2. `clean_gv` distinguish clean group velocipy dispersion curves
         3. `clean_cv` distinguish clean phase velocipy dispersion curves
    """

    # if not given dataset directory, func. use default dataset
    if not datasetdir:
        module_path = dirname(__file__)
        datasetdir = module_path
    # define classifier mode
    # 1. `raw_gv` distinguish raw group velocipy dispersion curves
    # 2. `clean_gv` distinguish clean group velocipy dispersion curves
    # 3. `clean_cv` distinguish clean phase velocipy dispersion curves
    if mode == "raw_gv":
        dataname = "raw_gv.csv"
        descname = "raw_gv.rst"
        feature_names=['Instantaneous period (s)', 'Raw Group Velocity (km/s)',
                        'Velocity error (km/s)']
    if mode == "clean_gv":
        dataname = "clean_gv.csv"
        descname = "clean_gv.rst"
        feature_names=['Instantaneous period (s)', 'Clean Group Velocity (km/s)',
                        'Velocity error (km/s)']

    if mode == "clean_cv":
        dataname = "clean_cv.csv"
        descname = "clean_cv.rst"
        feature_names=['Instantaneous period (s)', 'Clean Phase Velocity (km/s)',
                        'Velocity error (km/s)']

    with open(join(datasetdir, "data", dataname)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    with open(join(datasetdir, "descr", descname)) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=feature_names)


def gen_disp_classifier(datasetdir=None, min_samples_split=20, weighted=True,
                        mode="raw_gv"):
    """Generate dispersion classifier

    Parameters
    ===========

    min_samples_split : int
        minimum samples in splitting, default 20
    weighted: Boolean
         determine whethet weight with velocity error
    mode : string
         determine classifier type and function. Possible classifier
         1. `raw_gv` distinguish raw group velocipy dispersion curves
         2. `clean_gv` distinguish clean group velocipy dispersion curves
         3. `clean_cv` distinguish clean phase velocipy dispersion curves
    """
    if not datasetdir:
        disp = load_disp(mode=mode)
    else:
        disp = load_disp(datasetdir=datasetdir, mode=mode)

    x = disp.data[:, [0, 1]]
    y = disp.target
    if weighted:
        
        # give out weight vector
        if mode =="raw_gv":
            errweight = 1.0 / disp.data[:, -1]
        if mode == "clean_cv":
            errweight = 1.0 / disp.data[:, -2]

        clf = DecisionTreeClassifier(min_samples_split=min_samples_split
                                     ).fit(x, y, sample_weight=errweight)
        return clf
    else:
        clf = DecisionTreeClassifier(min_samples_split=min_samples_split
                                     ).fit(x, y)
        return clf
