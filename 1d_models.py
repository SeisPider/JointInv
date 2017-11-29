#!/usr/bin/python -u
"""
This script inverts dispersion curves at selected locations
for a 1D shear velocity model.

The model is parametrized with:
- one sedimentary layer, with fixed Vs, Vp/Vs, rho/Vs and free thickness
- a user-defined number of basement layers, with fixed Vp/Vs, rho/Vs and
  free Vs and thickness
- a semi-infinite mantle with fixed Vs, Vp/Vs, rho/Vs
So, the model's parameters are (1) the thicknesses of the sediment and basement
layers and (2) the Vs of the basement layers. In order to define the space of
plausible parameters, you must provide bounds for these parameters, and you can
add additional constraints on (1) the minimum acceptable increment of Vs between
two adjacent layer (e.g., 0 to force an increasing Vs with depth) and (2) on the
acceptable range for the total crustal thickness (sediment + basement). For each
free parameter's bound and fixed property, you can give a default global value,
as well as location-specific values, in case you have at hand better constraints
from independant observations.

The Computer Programs in Seismology [Herrmann, R.B., 2013. Computer Programs in
Seismology: an evolving tool for instruction and research, Seismol. Res. Let., 84(6),
1081-1088] take care of the forward model, wherein a theoretical Rayleigh group
dispersion curve, vg_model(T), is calculated given a model. Observed dispersion
curves, vg_obs(T), are constructed from user-defined dispersion maps, by compiling
group velocities observed at a (user-defined) number of grid nodes near the selected
locations. The misfit to observed dispersion curves is then:

  S(m) = 1/2 sum{ [vg_model(T) - <vg_obs(T)>]^2 / sigma_obs(T)^2 } over T

with <vg_obs(T)> and sigma_obs(T) the mean and standard deviation of observed
group velocities at period T, respectively.

The first step of the algorithm is to seek m0 that minimizes S(m) using a
constrained optimization (COBYLA). As the forward relationship is highly
non-linear, the optimization usually does a poor job, so this m0 shound not
be trusted as is: the objective is to provide a roughly acceptable starting
point to the Monte Carlo exploration that comes next.

The second step applies the Markov Chain Monte Carlo (MCMC) method of
Mosegaard, K. & Tarantola, A., [1995. Monte Carlo sampling of solutions to inverse
problems, J. Geophys. Res., 100(B7), 12431-12447] in order to sample the
posterior distribution of the parameters:

  f_post(m) = k.f_prior(m).L(m),

where f_prior is the prior distribution and L the likelihood function. The
algorithm uses Gaussian uncertainties, so that L(m) = exp(-S(m)). The prior
distribution is simply uniform within the parameters' bounds. It is sampled
with a prior random walk that, at each iteration:
- selects at random a thickness and a Vs,
- perturbates the selected thickness and Vs according to a uniform random walk,
  with user-defined step size an max jump size
This prior walk is then modified according to the Metropolis rule described by
Mosegaard & Tarantola [1995], in order to sample the posterior distribution.
Actually, the algorithm also rejects any perturbation towards an implausible
model (e.g., total crustal thickness out of bounds), which amounts to setting
a zero likelihood to implausible models and a uniform likelihood to plausible
models.

The results are exported to three files per location:
1) <location name> (prior distribution)_<suffix>.png
2) <location name>_<suffix>.png
3) <location name>_<suffix>.pickle
where <suffix> is a user-defined suffix.

The first file constains histogram plots of samples drawn from the prior
distribution, by accepting all the moves proposed by the prior walk.
You should check that you have drawn enough samples for the histograms to
look uniform.

The second file gives a summary of the posterior distribution of the
parameters. In particular, you'll find histograms of the sampled parameters
(and also of the depth of the base of the layers), with mean, standard dev,
95% confidence interval. Also shown are the observed and (95% interval of)
the theoretical dispersion curves and the (95% interval of) Vs vs depth,
together with the initial model (the one obtained after the optimization
step), the best-fitting model and the "representative" model (model closest
to the mean of posterior Vs vs depth).

The third file contains two objects exported with module pickle: (1) the
list of observed dispersion curves, as a list of numpy arrays
[vgarray1, vgarray2 ...], and (2) the models sampled from the posterior
distribution by the MCMC algorithm, as a list of instances of VsModel
(module psdepthmodel) [vsmodel1, vsmodel2 ...]
"""
from pysismo import psdepthmodel, psmcsampling
import os, sys
import shutil
import pickle
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from pysismo.psconfig import TOMO_DIR, DEPTHMODELS_DIR

# ==================================================================
# locations and associated names, {name: (lon, lat)}, around which
# extract the dispersion curves to be inverted. For each location,
# we extract the group velocities at the *NB_NEIGHBORS* nearest grid
# nodes of the dispersion maps
# ==================================================================

#LOCATIONS = {'Parana basin': (-52, -22),
#             'Sao Francisco craton': (-45, -19),
#             'Tocantins province': (-49, -13)
#             }
LOCATIONS = {
"Num1":(75,36),
"Num2":(75,37),
"Num3":(75,38),
"Num4":(75,39),
"Num5":(75,40),
"Num6":(75,41),
"Num7":(75,42),
"Num8":(75,43),
"Num9":(75,44),
"Num10":(75,45),
"Num11":(75,46),
"Num12":(75,47),
"Num13":(75,48),
"Num14":(76,36),
"Num15":(76,37),
"Num16":(76,38),
"Num17":(76,39),
"Num18":(76,40),
"Num19":(76,41),
"Num20":(76,42),
"Num21":(76,43),
"Num22":(76,44),
"Num23":(76,45),
"Num24":(76,46),
"Num25":(76,47),
"Num26":(76,48),
"Num27":(77,36),
"Num28":(77,37),
"Num29":(77,38),
"Num30":(77,39),
"Num31":(77,40),
"Num32":(77,41),
"Num33":(77,42),
"Num34":(77,43),
"Num35":(77,44),
"Num36":(77,45),
"Num37":(77,46),
"Num38":(77,47),
"Num39":(77,48),
"Num40":(78,36),
"Num41":(78,37),
"Num42":(78,38),
"Num43":(78,39),
"Num44":(78,40),
"Num45":(78,41),
"Num46":(78,42),
"Num47":(78,43),
"Num48":(78,44),
"Num49":(78,45),
"Num50":(78,46),
"Num51":(78,47),
"Num52":(78,48),
"Num53":(79,36),
"Num54":(79,37),
"Num55":(79,38),
"Num56":(79,39),
"Num57":(79,40),
"Num58":(79,41),
"Num59":(79,42),
"Num60":(79,43),
"Num61":(79,44),
"Num62":(79,45),
"Num63":(79,46),
"Num64":(79,47),
"Num65":(79,48),
"Num66":(80,36),
"Num67":(80,37),
"Num68":(80,38),
"Num69":(80,39),
"Num70":(80,40),
"Num71":(80,41),
"Num72":(80,42),
"Num73":(80,43),
"Num74":(80,44),
"Num75":(80,45),
"Num76":(80,46),
"Num77":(80,47),
"Num78":(80,48),
"Num79":(81,36),
"Num80":(81,37),
"Num81":(81,38),
"Num82":(81,39),
"Num83":(81,40),
"Num84":(81,41),
"Num85":(81,42),
"Num86":(81,43),
"Num87":(81,44),
"Num88":(81,45),
"Num89":(81,46),
"Num90":(81,47),
"Num91":(81,48),
"Num92":(82,36),
"Num93":(82,37),
"Num94":(82,38),
"Num95":(82,39),
"Num96":(82,40),
"Num97":(82,41),
"Num98":(82,42),
"Num99":(82,43),
"Num100":(82,44),
"Num101":(82,45),
"Num102":(82,46),
"Num103":(82,47),
"Num104":(82,48),
"Num105":(83,36),
"Num106":(83,37),
"Num107":(83,38),
"Num108":(83,39),
"Num109":(83,40),
"Num110":(83,41),
"Num111":(83,42),
"Num112":(83,43),
"Num113":(83,44),
"Num114":(83,45),
"Num115":(83,46),
"Num116":(83,47),
"Num117":(83,48),
"Num118":(84,36),
"Num119":(84,37),
"Num120":(84,38),
"Num121":(84,39),
"Num122":(84,40),
"Num123":(84,41),
"Num124":(84,42),
"Num125":(84,43),
"Num126":(84,44),
"Num127":(84,45),
"Num128":(84,46),
"Num129":(84,47),
"Num130":(84,48),
"Num131":(85,36),
"Num132":(85,37),
"Num133":(85,38),
"Num134":(85,39),
"Num135":(85,40),
"Num136":(85,41),
"Num137":(85,42),
"Num138":(85,43),
"Num139":(85,44),
"Num140":(85,45),
"Num141":(85,46),
"Num142":(85,47),
"Num143":(85,48),
}
NB_NEIGHBORS = 1

print("Select location(s) on which estimate depth models [all]:")
print('0 - All')
#print('\n'.join('{} - {}'.format(i + 1, k) for i, k in enumerate(sorted(LOCATIONS))))
#res = raw_input('\n')
res = sys.argv[1]
if res:
    LOCATIONS = dict(sorted(LOCATIONS.items())[int(i) - 1] for i in res.split())

# ======================================================================
# parametrization of the model: number of crustal layers, ratio Vp/Vs,
# rho/Vs (of sediments, crust and mantle) and Vs of sediments and mantle
# ======================================================================

DEPTHS = np.arange(70)  # depths over which the model will be plotted

# sediments: default
VS_SEDIMENTS = 1.07
RATIO_VP_VS_SEDIMENTS = 2.336  # Vp = 2.5 km/s,    crust 1.0
RATIO_RHO_VS_SEDIMENTS = 1.972  # rho = 2.11 g/cm3   crust 1.0
# sediments: location-specific (where available)
LOCAL_VS_SEDIMENTS = {}
LOCAL_RATIO_VP_VS_SEDIMENTS = {}
LOCAL_RATIO_RHO_VS_SEDIMENTS = {}
# crust: defaut
NB_CRUST_LAYERS = 3
RATIO_VP_VS_CRUST = 1.726  # crust 1.0
RATIO_RHO_VS_CRUST = 0.7616
# crust: location-specific (where available)
LOCAL_NB_CRUST_LAYERS = {}
LOCAL_RATIO_VP_VS_CRUST = {}
LOCAL_RATIO_RHO_VS_CRUST = {}
# mantle: default
VS_MANTLE = 4.54              # Chulick et al., 2013 (avg South America)
RATIO_VP_VS_MANTLE = 1.8017   # Vp = 8 km/s, Chulick et al. 2013 (avg South America)
RATIO_RHO_VS_MANTLE = 0.74229  # rho = 3.35 g/cm3
# mante: location-specific (where available)
LOCAL_VS_MANTLE = {}
LOCAL_RATIO_VP_VS_MANTLE = {}
LOCAL_RATIO_RHO_VS_MANTLE = {}

# ==================================================================
# space of plausible models: bounds on layers' Vs, layers' thickness
# and Moho depth, minimum Vs increment between two layers
# ==================================================================

# default bounds
DZ_SEDIMENTS_BOUNDS = (0.0, 10.0)
VS_CRUST_BOUNDS = (3.2, 5.5)
DZ_CRUST_BOUNDS = (10.0, 30.0)
MOHO_DEPTH_BOUNDS = (35.0, 70.0)
# location-specific bounds (if available)
# sediments thickness: 2 km around (rounded) value of Laske & Masters
LOCAL_DZ_SEDIMENTS_BOUNDS = {'Parana basin': (3.0, 7.0),      # Laske & Masters: 4.5 4.5 5 5.2 km
                             'Sao Francisco craton': (0.0, 2.0),  # (neighbours) 0.1 0.1 0.25 0.4 km
                             'Tocantins province': (0.0, 2.0)}                 # 0.01 0.01 0.1 0.1 km
LOCAL_VS_CRUST_BOUNDS = {}
LOCAL_DZ_CRUST_BOUNDS = {}
# Moho depth: 5 km around (rounded) value of Assumpcao et al.
LOCAL_MOHO_DEPTH_BOUNDS = {'Parana basin': (38.0, 48.0),  # Assumpcao et al.: 43.4 km
                           'Sao Francisco craton': (35.0, 45.0),            # 39.7 km
                           'Tocantins province': (33.0, 43.0)               # 38 km
                           }   # from Assumpcao et al.
# min Vs increment (set to 0 to force Vs increase with depth)
VS_MIN_INCREMENT = 0.0

# =============================================================
# parameters of the Monte Carlo exploration: sampling step, max
# allowed jump, nb of samples and size of the burn in phase
# =============================================================

# crustal Vs
VS_SAMPLINGSTEP = 0.02
VS_MAXJUMP = 0.1
# thickness of sediment layer
DZ_SEDIMENTS_SAMPLINGSTEP = 0.2
DZ_SEDIMENTS_MAXJUMP = 2.0
# thickness of crustal layers
DZ_CRUSTLAYER_SAMPLINGSTEP =0.2
DZ_CRUSTLAYER_MAXJUMP = 2.0
# nb of samples
NB_SAMPLES = 5000
#res = raw_input('Number of samples of MC exploration? [{}]\n'.format(NB_SAMPLES))
#NB_SAMPLES = int(res) if res.strip() else NB_SAMPLES
# nb of burnt in samples
NB_BURN = min(int(NB_SAMPLES / 10), 200)
#res = raw_input('Number of burnt in samples? [{}]\n'.format(NB_BURN))
#NB_BURN = int(res) if res.strip() else NB_BURN


# user-defined suffix to append to file names
#usersuffix = raw_input("\nEnter suffix to append: [none]\n").strip()
usersuffix = None

# =====================================================================
# assigning to each period a dispersion map from which group velocities
# will be etxracted, in a dict {period: velocity map}
# =====================================================================

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

# =========================
# loop on selected location
# =========================

for locname, (lon, lat) in sorted(LOCATIONS.items()):
    print("Working at location '{}': lon={}, lat={}".format(locname, lon, lat))

    # getting location-specific parameters/bounds if available, else default ones
    # parameters
    vs_sediments = LOCAL_VS_SEDIMENTS.get(locname, VS_SEDIMENTS)
    ratio_vp_vs_sediments = LOCAL_RATIO_VP_VS_SEDIMENTS.get(locname,
                                                            RATIO_VP_VS_SEDIMENTS)
    ratio_rho_vs_sediments = LOCAL_RATIO_RHO_VS_SEDIMENTS.get(locname,
                                                              RATIO_RHO_VS_SEDIMENTS)
    nb_crust_layers = LOCAL_NB_CRUST_LAYERS.get(locname, NB_CRUST_LAYERS)
    ratio_vp_vs_crust = LOCAL_RATIO_VP_VS_CRUST.get(locname, RATIO_VP_VS_CRUST)
    ratio_rho_vs_crust = LOCAL_RATIO_RHO_VS_CRUST.get(locname, RATIO_RHO_VS_CRUST)
    vs_mantle = LOCAL_VS_MANTLE.get(locname, VS_MANTLE)
    ratio_vp_vs_mantle = LOCAL_RATIO_VP_VS_MANTLE.get(locname, RATIO_VP_VS_MANTLE)
    ratio_rho_vs_mantle = LOCAL_RATIO_RHO_VS_MANTLE.get(locname, RATIO_RHO_VS_MANTLE)
    # arrays of ratio Vp/Vs and rho/Vs
    ratio_vp_vs = np.r_[ratio_vp_vs_sediments,
                        nb_crust_layers * [ratio_vp_vs_crust],
                        ratio_vp_vs_mantle]
    ratio_rho_vs = np.r_[ratio_rho_vs_sediments,
                         nb_crust_layers * [ratio_rho_vs_crust],
                         ratio_rho_vs_mantle]
    # bounds
    dz_sediments_bounds = LOCAL_DZ_SEDIMENTS_BOUNDS.get(locname, DZ_SEDIMENTS_BOUNDS)
    vs_crust_bounds = LOCAL_VS_CRUST_BOUNDS.get(locname, VS_CRUST_BOUNDS)
    dz_crust_bounds = LOCAL_DZ_CRUST_BOUNDS.get(locname, DZ_CRUST_BOUNDS)
    moho_depth_bounds = LOCAL_MOHO_DEPTH_BOUNDS.get(locname, MOHO_DEPTH_BOUNDS)

    # =================================================================
    # getting the dispersion curves at the *NB_NEIGHBORS* nearest nodes
    # =================================================================

    print("  getting observed dispersion curves at nearest nodes")
    vgarrays = [np.zeros_like(PERIODS, dtype='float') for _ in range(NB_NEIGHBORS)]
    meanvg = np.zeros_like(PERIODS, dtype='float')
    sigmavg = np.zeros_like(PERIODS, dtype='float')
    previous_grid = None
    previous_inodes = None
    for iT, T in enumerate(PERIODS):
        # getting the velocities of period T at the *NB_NEIGHBORS* nodes
        # closest to current location
        vmap = PERIODVMAPS[T]
        if vmap.grid == previous_grid:
            inodes = previous_inodes
        else:
            xnodes, ynodes = vmap.grid.xy_nodes()
            inodes = np.argsort((xnodes - lon)**2 + (ynodes - lat)**2)[:NB_NEIGHBORS]

        previous_grid = vmap.grid
        previous_inodes = inodes

        # velocities at period T + mean and std dev
        vels = np.array((vmap.v0 / (1 + vmap.mopt))).flatten()[inodes]
        for vg, vel in zip(vgarrays, vels):
            vg[iT] = vel
        meanvg[iT] = vels.mean()

        # standard deviation of velocities...
        if vels.size > 1:
            # ...estimated across the set of velocities if
            # several velocities available (NB_NEIGHBORS > 1)
            sigmavg[iT] = vels.std()
        else:
            # ...estimated from the covariance matrix of the
            # best-fitting parameters of the tomographic inversion
            # if only one velocity (NB_NEIGHBORS = 1)
            sigmamopt = np.sqrt(vmap.covmopt[inodes[0], inodes[0]])
            # m = (v0 - v) / v, so sigma_m = v0 * sigma_v / v^2
            sigmavg[iT] = vels[0]**2 * sigmamopt / vmap.v0

    # ==============================================================
    # quick estimate of best-fitting parameters (thicknesses and Vs)
    # ==============================================================

    def misfit(m):
        """
        Objective function to minimize: misfit between
        modelled and observed group velocities
        """
        # vector of parameters = [vs of crustal layers]
        #                        + [dz of sediment and crustal layers]
        vs_crust = m[:nb_crust_layers]

        # building the model
        vsmodel = psdepthmodel.VsModel(vs=np.r_[vs_sediments, vs_crust, vs_mantle],
                                       dz=m[nb_crust_layers:],
                                       ratio_vp_vs=ratio_vp_vs,
                                       ratio_rho_vs=ratio_rho_vs)
        return vsmodel.misfit_to_vg(periods=PERIODS,
                                    vg=meanvg,
                                    sigmavg=sigmavg)

    # estimating best-fitting parameters (thickness and Vs of layers)
    print("  estimating best-fitting depth model")

    # initial parameters
    dz0_sediments = 4.0
    vs0_crust = 3.7
    dz0_crust = (np.mean(moho_depth_bounds) - dz0_sediments) / nb_crust_layers
    vs0 = [vs0_crust] * nb_crust_layers
    dz0 = [dz0_sediments] + [dz0_crust] * nb_crust_layers
    m0 = np.array(vs0 + dz0)

    # constraint on Vs bounds
    constraints = []
    for i in range(nb_crust_layers):
        # Vs > min Vs
        constr = {'type': 'ineq', 'fun': lambda m, i=i: m[i] - vs_crust_bounds[0]}
        constraints.append(constr)
        # Vs < max Vs
        constr = {'type': 'ineq', 'fun': lambda m, i=i: vs_crust_bounds[1] - m[i]}
        constraints.append(constr)

    # constraints on sediment layer's thickness's bounds
    # thickness > min thickness
    constr = {'type': 'ineq',
              'fun': lambda m: m[nb_crust_layers] - dz_sediments_bounds[0]}
    constraints.append(constr)
    # thickness < min thickness
    constr = {'type': 'ineq',
              'fun': lambda m: dz_sediments_bounds[1] - m[nb_crust_layers]}
    constraints.append(constr)

    # constraints on crustal layer's thickness's bounds
    for i in range(nb_crust_layers + 1, len(m0)):
        # thickness > min thickness
        constr = {'type': 'ineq', 'fun': lambda m, i=i: m[i] - dz_crust_bounds[0]}
        constraints.append(constr)
        # thickness < min thickness
        constr = {'type': 'ineq', 'fun': lambda m, i=i: dz_crust_bounds[1] - m[i]}
        constraints.append(constr)

    # constraint on Moho depth
    constraints += [
        {'type': 'ineq',
         'fun': lambda m: np.sum(m[nb_crust_layers:]) - moho_depth_bounds[0]},
        {'type': 'ineq',
         'fun': lambda m: moho_depth_bounds[1] - np.sum(m[nb_crust_layers:])}
    ]

    # constraint on Vs increments:
    # Vs_nextlayer - Vs_currentlayer > VS_MIN_INCREMENT
    if not VS_MIN_INCREMENT is None:
        # constraint on increment between sediments and first crustal layer
        constr = {'type': 'ineq',
                  'fun': lambda m: m[0] - vs_sediments - VS_MIN_INCREMENT}
        constraints.append(constr)
        # constraints on increments between crustal layers
        for i in range(nb_crust_layers - 1):
            constr = {'type': 'ineq',
                      'fun': lambda m, i=i: m[i+1] - m[i] - VS_MIN_INCREMENT}
            constraints.append(constr)
        # constraint on increment between last crustal layer and mantle
        constr = {'type': 'ineq',
                  'fun': lambda m: vs_mantle - m[nb_crust_layers - 1] - VS_MIN_INCREMENT}
        constraints.append(constr)

    # constrained optimization
    mopt = minimize(misfit, x0=m0, method='COBYLA', constraints=constraints)['x']
    vscrustopt, dzsedopt, dzcrustopt = (mopt[:nb_crust_layers],
                                        mopt[nb_crust_layers],
                                        mopt[nb_crust_layers + 1:])

    # using the best-fitting model as initial model of the MC exploration
    vsmodelinit = psdepthmodel.VsModel(vs=np.r_[vs_sediments, vscrustopt, vs_mantle],
                                       dz=np.r_[dzsedopt, dzcrustopt],
                                       ratio_vp_vs=ratio_vp_vs,
                                       ratio_rho_vs=ratio_rho_vs,
                                       name='Initial model')

    # =======================
    # Monte-Carlo exploration
    # =======================

    # initializing parameters
    vscrustlayers = []
    dzcrustlayers = []
    for i in range(nb_crust_layers):
        vscrustlayer = psmcsampling.Parameter(
            name='Vs of crustal layer #{}'.format(i + 1),
            minval=vs_crust_bounds[0],
            maxval=vs_crust_bounds[1],
            step=VS_SAMPLINGSTEP,
            startval=vscrustopt[i],
            maxjumpsize=VS_MAXJUMP,
            nmaxsample=NB_SAMPLES)
        vscrustlayers.append(vscrustlayer)

        dzcrustlayer = psmcsampling.Parameter(
            name='Thickness of crustal layer #{}'.format(i + 1),
            minval=dz_crust_bounds[0],
            maxval=dz_crust_bounds[1],
            step=DZ_CRUSTLAYER_SAMPLINGSTEP,
            startval=dzcrustopt[i],
            maxjumpsize=DZ_CRUSTLAYER_MAXJUMP,
            nmaxsample=NB_SAMPLES)
        dzcrustlayers.append(dzcrustlayer)

    dzsedlayer = psmcsampling.Parameter(name='Thicknes of sediment layer',
                                        minval=dz_sediments_bounds[0],
                                        maxval=dz_sediments_bounds[1],
                                        step=DZ_SEDIMENTS_SAMPLINGSTEP,
                                        startval=dzsedopt,
                                        maxjumpsize=DZ_SEDIMENTS_MAXJUMP,
                                        nmaxsample=NB_SAMPLES)

    dzlayers = [dzsedlayer] + dzcrustlayers
    parameters = vscrustlayers + dzlayers

    # initial misfit and likelihood of parameters
    vsmodel_current = vsmodelinit
    misfit_current = vsmodel_current.misfit_to_vg(
        periods=PERIODS, vg=meanvg, sigmavg=sigmavg, storevg=True)  # storing vg model
    likelihood_current = np.exp(- misfit_current)

    # Monte Carlo sampling with Metropolis rule turned off (1st loop) and on (2nd loop)
    for switchon_Metropolis in [False, True]:
        s = "  Monte Carlo sampling of the {} distribution of the parameters"
        print(s.format('prior' if not switchon_Metropolis else 'posterior'))

        # (re-)initializing parameters
        for m in parameters:
            m.reinit()
        nrefused = 0
        vsmodels = []

        for isample in range(1, NB_SAMPLES):
            if switchon_Metropolis and (isample + 1) / 10 == (isample + 1) / float(10):
                s = '    Collected {} / {} samples ({:.1f} % of the moves accepted)'
                relaccepted = float(isample - nrefused) / isample
                print(s.format(isample + 1, NB_SAMPLES, 100.0 * relaccepted))

            # adding sample to posterior distribution
            for m in parameters:
                m.addsample()
            vsmodels.append(vsmodel_current)

            # proposing next (random walk) move, which, if accepted, would
            # sample uniformly the parameters space: the strategy consists
            # in perturbating one Vs and one thickness selected at random
            # (and freezing all other parameters)
            freevslayer = np.random.choice(vscrustlayers)
            freedzlayer = np.random.choice(dzlayers)
            for vscrustlayer in vscrustlayers:
                vscrustlayer.frozen = False if vscrustlayer == freevslayer else True
            for dzlayer in dzlayers:
                dzlayer.frozen = False if dzlayer == freedzlayer else True
            for m in parameters:
                _ = m.propose_next()

            # we always accept the move if Metropolis rule is switched off
            if not switchon_Metropolis:
                for m in parameters:
                    m.accept_move()
                continue

            # we always refuse move towards an implausible model
            if not VS_MIN_INCREMENT is None:
                # checking Vs increments
                Vsincr = [next(m1) - next(m0)
                          for m0, m1 in zip(vscrustlayers[:-1], vscrustlayers[1:])]
                if any(dVs < VS_MIN_INCREMENT for dVs in Vsincr):
                    nrefused += 1
                    continue
            # checking Moho depth
            total_thickness = sum(next(m) for m in dzlayers)
            if not moho_depth_bounds[0] <= total_thickness <= moho_depth_bounds[1]:
                nrefused += 1
                continue

            # Vs model corresponding to the proposed parameters value
            vsmodel_next = psdepthmodel.VsModel(
                vs=np.r_[vs_sediments, [next(m) for m in vscrustlayers], vs_mantle],
                dz=[next(m) for m in dzlayers],
                ratio_vp_vs=ratio_vp_vs,
                ratio_rho_vs=ratio_rho_vs,
                name='Sample of posterior distribution')

            # the move is accepted with probability P = L_next / L_current,
            # with L = likelihood
            misfit_next = vsmodel_next.misfit_to_vg(periods=PERIODS,
                                                    vg=meanvg,
                                                    sigmavg=sigmavg,
                                                    storevg=True)  # storing vg model
            accept_move = psmcsampling.accept_move(misfit_current,
                                                   likelihood_current,
                                                   misfit_next)
            if not accept_move:
                nrefused += 1
                continue

            # move is accepted
            misfit_current = misfit_next
            likelihood_current = np.exp(- misfit_current)
            for m in parameters:
                m.accept_move()
            vsmodel_current = vsmodel_next

        if not switchon_Metropolis:
            # ==========================================
            # exporting histograms of prior distribution
            # ==========================================

            # out prefix = e.g., "1d models/Parana basin (prior distribution)"
            outprefix = os.path.join(DEPTHMODELS_DIR, locname + ' (prior distribution)')
            if usersuffix:
                outprefix += '_{}'.format(usersuffix)

            outfile = '{}.png'.format(outprefix)
            print("  exporting prior distributions to file: " + outfile)
            fig = plt.figure(figsize=(5 * (nb_crust_layers + 1), 10), tight_layout=True)

            # Vs distributions
            for icol, m in enumerate(vscrustlayers, start=2):
                ax = fig.add_subplot(2, nb_crust_layers + 1, icol)
                m.hist(ax=ax, nburnt=NB_BURN)

            # thickness distributions
            for icol, m in enumerate(dzlayers, start=nb_crust_layers + 2):
                ax = fig.add_subplot(2, nb_crust_layers + 1, icol)
                m.hist(ax=ax, nburnt=NB_BURN)

            if os.path.exists(outfile):
                # backup
                shutil.copy(outfile, outfile + '~')
            fig.savefig(outfile, dpi=300)

            continue

    # ==========================================================
    # dumping (1) observed vg arrays, (2) array of std dev of vg
    # and (3) sampled models from posterior distribution
    # ==========================================================

    # out prefix = e.g., "1d models/Parana basin"
    outprefix = os.path.join(DEPTHMODELS_DIR, locname)
    if usersuffix:
        outprefix += '_{}'.format(usersuffix)

    outfile = '{}.pickle'.format(outprefix)
    print('    dumping observed vg and sampled models to file: ' + outfile)
    if os.path.exists(outfile):
        # backup
        shutil.copy(outfile, outfile + '~')
    f = open(outfile, 'wb')
    # dumping observed group velocities
    pickle.dump(vgarrays, f, protocol=2)
    # dumping std deviation of vg
    pickle.dump(sigmavg, f, protocol=2)
    # dumping sampled models from posterior distribution
    pickle.dump(vsmodels, f, protocol=2)
    f.close()

    # ======================================================================
    # plotting results: 95% confidence intervals, posterior distributions...
    # =====================================================================

    outfile = '{}.png'.format(outprefix)
    print("    plotting results and saving to file: " + outfile)

    ncols = nb_crust_layers + 2
    nrows = 3

    quantiles = [2.5, 97.5]
    fig = plt.figure(figsize=(5 * ncols, 4 * nrows), tight_layout=True)

    # 1st column, 2nd line: Vs versus z: inital model, representative model,
    # 95% interval and mean of posterior distribution
    ax = fig.add_subplot(nrows, ncols, ncols + 1)
    vsmodelinit.plot_model(ax=ax, color='b')  # initial model
    # 95% confidence interval
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
    key = lambda vsmodel: vsmodel.misfit_to_vg(PERIODS, meanvg, sigmavg)
    best_model = min(vsmodels, key=key)
    best_model.name = 'Best-fitting model'
    best_model.plot_model(ax=ax, color='g')

    # 1st colum, 1st line: group velocities vs period: observed velocities,
    # initial model, representative model, 95% interval and mean
    # of posterior distribution
    ax = fig.add_subplot(nrows, ncols, 1)
    # initial model
    vsmodelinit.plot_vg(PERIODS,
                        obsvgarrays=vgarrays,
                        sigmavg=sigmavg if NB_NEIGHBORS == 1 else None,
                        ax=ax,
                        color='b')
    representative_model.plot_vg(PERIODS, ax=ax, color='r')  # representative model
    best_model.plot_vg(PERIODS, ax=ax, color='g')  # best-fitting model
    # 95% confidence interval
    vgmodels = [vsmodel.stored_vg for vsmodel in vsmodels[NB_BURN:]]
    vg_1stquant, vg_2ndquant = np.percentile(vgmodels, quantiles, axis=0)
    ax.fill_between(x=PERIODS, y1=vg_2ndquant, y2=vg_1stquant, color='grey', alpha=0.3)
    # mean
    vgmean = np.mean(vgmodels, axis=0)
    ax.plot(PERIODS, vgmean, '--', color='k')
    ax.set_title(locname)

    # 1st line, next columns: posterior distribution of crustal layers' Vs
    for i, m in enumerate(vscrustlayers, start=3):
        ax = fig.add_subplot(nrows, ncols, i)
        m.hist(ax, nburnt=NB_BURN)

    # 2nd line, next columns: posterior distribution of
    # (sediment and crust) layers' thickness
    for i, m in enumerate(dzlayers, start=ncols + 2):
        ax = fig.add_subplot(nrows, ncols, i)
        m.hist(ax, nburnt=NB_BURN)

    # 3rd line, 1st column: misfit vs iteration nb
    misfits = [vsmodel.misfit_to_vg(PERIODS, vgarrays, sigmavg) for vsmodel in vsmodels]
    ax = fig.add_subplot(nrows, ncols, 2 * ncols + 1)
    ax.plot(misfits)
    ax.set_xlabel('Iteration nb')
    ax.set_ylabel('Misfit')
    ax.grid(True)

    # 3rd line, next columns: posterior distribution of depth of
    # base of crustal layers (including Moho depth)
    for icrustlayer in range(1, nb_crust_layers + 1):
        ax = fig.add_subplot(nrows, ncols, 2 * ncols + 1 + icrustlayer)
        zbase = dzsedlayer + sum(dzcrustlayers[:icrustlayer])
        if icrustlayer < nb_crust_layers:
            xlabel = 'Depth of base of crustal layer #{}'.format(icrustlayer)
        else:
            xlabel = 'Moho depth'
        zbase.hist(ax, nburnt=NB_BURN, xlabel=xlabel)

    if os.path.exists(outfile):
        # backup
        shutil.copy(outfile, outfile + '~')
    fig.savefig(outfile, dpi=300)
    fig.show()
