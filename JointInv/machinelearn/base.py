"""
Module for dispersion training data IO (classification).
"""
import glob
import os
import csv
from os.path import join, dirname

import numpy as np
from sklearn.datasets.base import Bunch

import matplotlib.pyplot as plt
from copy import copy

CPSPATH = "~/src/CPS/PROGRAMS.330/bin"
class record(object):
    """
    Class for extract info from MFT96 files
    """

    def __init__(self, recordstr):
        """
        Import string and master info.
        """
        self.raw = recordstr + "\n"
        self._extrinfo(recordstr)

        # set default label [not on]
        self.label = np.array([1])

    def __repr__(self):
        """
        Representation of class record
        """
        return "Spectrum Amp Peak @{} s".format(self.period)

    def _extrinfo(self, recordstr):
        """
        Extract info in MFT96 file
        """
        self.flag = recordstr.split()[0]
        self.wavetype = recordstr.split()[1]
        self.velotype = recordstr.split()[2]
        self.mode = recordstr.split()[3]
        self.period = float(recordstr.split()[4])
        self.velo = float(recordstr.split()[5])
        self.veloerr = float(recordstr.split()[6])
        self.dist = recordstr.split()[7]
        self.azimuth = recordstr.split()[8]
        self.specamp = recordstr.split()[9]
        self.evla = recordstr.split()[10]
        self.evlo = recordstr.split()[11]
        self.stla = recordstr.split()[12]
        self.stlo = recordstr.split()[13]
        self.zero = recordstr.split()[14]
        self.pknum = recordstr.split()[15]
        self.instaper = float(recordstr.split()[16])

        if len(recordstr.split()) == 24:
            self.COMMENT = recordstr.split()[17]
            self.stnm = recordstr.split()[18]
            self.chnm = recordstr.split()[19]
            self.year = recordstr.split()[20]
            self.month = recordstr.split()[21]
            self.day = recordstr.split()[22]
            self.hour = recordstr.split()[23]
        elif len(recordstr.split()) == 25:
            self.upper = recordstr.split()[17]
            self.COMMENT = recordstr.split()[18]
            self.stnm = recordstr.split()[19]
            self.chnm = recordstr.split()[20]
            self.year = recordstr.split()[21]
            self.month = recordstr.split()[22]
            self.day = recordstr.split()[23]
            self.hour = recordstr.split()[24]


class velomap(object):
    """
    Class holds velocity map and corresponding method 
    """

    def __init__(self, dispinfo, refdisp, trained_model, periodmin=25,
                 periodmax=100, cpspath=CPSPATH):
        """
        Import peaks and reference dispersion curve. After that, extract
        dispersion curve from this map.
        """
        self.id = os.path.basename(dispinfo).replace(".mft96.disp", "")
        self.records = self.obtain_velomap(dispinfo)
        self.rdrefgvdisp(refdisp)
        self.periodmin, self.periodmax = periodmin, periodmax
        self.labels, self.disprec = self.classification(trained_model, periodmin,
                                                        periodmax)

    def __repr__(self):
        """
        Give out representation info
        """
        return "Dispersion Map and Curve of sta {}".format(self.records[0].stnm)

    def classification(self, trained_model, periodmin, periodmax):
        """
        Do clasification in a period-velocity map
        """
        # judge for each record
        labels = np.zeros_like(self.records)
        disprec = []

        for idx, record in enumerate(self.records):
            # judge with period limitation
            if record.instaper > periodmax or record.instaper < periodmin:
                labels[idx] = np.array([1])     # outlier
                continue

            # judge with trained model
            x_test = np.matrix([[record.instaper, record.velo]])
            record.label = trained_model.predict(x_test)
            labels[idx] = copy(record.label)

            if not record.label:
                disprec.append(record)
        return labels, np.array(disprec)

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

    def isowithsacmat96(self, srcpath, surf96filepath, cpspath=CPSPATH):
        """
        Isolate SAC traces with extracted group velocity dispersion curves
        """
        print(join(srcpath, "*{}*".format(self.id)))
        srcsacfile = glob.glob(join(srcpath, "*{}*.SAC".format(self.id)))[0]

        # isolate fundamental Rayleigh wave with sacmat96
        excfile = join(cpspath, "sacmat96")
        cmndstr = "{0} -F {1} -D {2} -AUTO".format(excfile, srcsacfile,
                                                   surf96filepath)
        os.system(cmndstr)

    def obtain_velomap(self, dispinfo):
        records = []
        with open(dispinfo, "r") as f:
            lines = f.readlines()
            # extract info.
            for line in lines:
                line = line.strip()
                records.append(record(line))
        if not records:
            return None
        else:
            return np.array(records)

    def rdrefgvdisp(self, refdisp):
        """
        Read reference dispersion file 
        """
        self.refperiods = np.loadtxt(refdisp, skiprows=1, usecols=2)
        self.refgv = np.loadtxt(refdisp, skiprows=1, usecols=5)

    def pltgvmap(self, permin=25, permax=100, filename=None):
        """
        Plot period-velocity map
        """
        self.instper = np.array([float(x.instaper) for x in self.records])
        self.gv = np.array([float(x.velo) for x in self.records])
        self.gverr = np.array([float(x.veloerr) for x in self.records])

        xmin = permin - 1.0
        xmax = permax + 1.0
        ymin = self.gv.min() - 1.0
        ymax = self.gv.max() + 1.0

        plt.errorbar(self.instper, self.gv, yerr=self.gverr, fmt="o",
                     label="Observation")
        plt.plot(self.refperiods, self.refgv, label="Synthetic [ak135]")
        plt.xlabel("Instaneous Period [s]")
        plt.ylabel("Group Velocity [km/s]")
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.legend()

        if not filename:
            plt.show()
        else:
            plt.savefig(filename)


def load_disp(return_X_y=False, datasetdir=None):
    """Load and return the dispersion trainning data (classfication) 
    """
    
    # if not given dataset directory, func. use default dataset 
    if not datasetdir:
        module_path = dirname(__file__)
        datasetdir = module_path

    with open(join(datasetdir, "data", "disp.csv")) as csv_file:
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

    with open(join(datasetdir, "descr", 'disp.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['Instantaneous period (s)', 'Group velocity (km/s)',
                                'Velocity error (km/s)'])
