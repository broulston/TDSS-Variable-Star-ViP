%load_ext autoreload
%autoreload 2

import matplotlib
matplotlib.use('TkAGG')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable

from urllib.parse import urlencode
from urllib.request import urlretrieve

import numpy as np
import numpy.core.defchararray as np_f
import pandas as pd
import scipy as sci
from scipy.stats import f
from scipy.stats import kde
from subprocess import *
import os
import glob
from pathlib import Path
import re

from astropy.table import Table
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from astropy import coordinates as coords

import importlib
import tqdm

import time

import warnings

import ResearchTools.LCtools as LCtools
import VarStar_Vi_plot_functions as vi

spec_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/SDSS_spec/02-26-2020/SDSSspec/"
CSS_LC_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/CSS_LCs/csvs/"
ZTF_LC_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/ZTF/DATA/06-24-2020/"

ZTF_filters = ['g', 'r']
ZTF_LC_file_names = [f'TDSS_SES+PREV_DR16DR12griLT20_GAIADR2_Drake2014PerVar_ZTF_{ZTF_filter}_epochGT10_GroupID.fits' for ZTF_filter in ZTF_filters]
ZTF_g_LCs = Table.read(ZTF_LC_dir + ZTF_LC_file_names[0])
ZTF_r_LCs = Table.read(ZTF_LC_dir + ZTF_LC_file_names[1])
# ***********************************************
prop_out_dir, CSS_LC_plot_dir, ZTF_LC_plot_dir, Vi_plots_dir, datestr = vi.makeViDirs()
# ***********************************************
nbins = 50
TDSSprop = vi.TDSSprop(nbins)
# ***********************************************
latestFullVartoolsRun_filename = "completed_Vi_prop_2020-07-16.csv"
latestFullVartoolsRun = vi.latestFullVartoolsRun(prop_out_dir + latestFullVartoolsRun_filename)
# ***********************************************
# TDSS_cssid_orginal = TDSSprop.TDSS_cssid
# prop_header = "ra, dec, lc_id, Per_ls, logProb_ls, Amp_ls, Mt, a95, lc_skew, Chi2, brtcutoff,\
#                brt10per, fnt10per, fntcutoff, errmn, ferrmn, ngood, nrejects, nabove, nbelow, \
#                Tspan100, Tspan95, isAlias, VarStat, Con, m, b_lin, chi2_lin, a, b_quad, c, \
#                chi2_quad, EqW"
hasViRun, prop_id_last, properties = vi.checkViRun()  # if Vi has run, this will find where it let off and continue propid from there

prop_col_names_prefix = ['CSS_', 'ZTF_g_', 'ZTF_r_']
if hasViRun:
    pass
else:
    prop_id = 0
    prop_id_last = 0
    prop_col_names = ['lc_id', 'P', 'logProb', 'Amp', 'Mt', 'a95', 'lc_skew',
                      'Chi2', 'brtcutoff', 'brt10per', 'fnt10per', 'fntcutoff', 'errmn', 'ferrmn',
                      'ngood', 'nrejects', 'nabove', 'nbelow', 'Tspan100', 'Tspan95', 'isAlias', 'time_whittened',
                      'VarStat', 'Con', 'm', 'b_lin', 'chi2_lin', 'a', 'b_quad', 'c', 'chi2_quad']

    prop_col_names_full = [ii + jj for ii in prop_col_names_prefix for jj in prop_col_names]
    prop_col_names_full.insert(0, 'ViCompleted')
    prop_col_names_full.insert(0, 'dec')
    prop_col_names_full.insert(0, 'ra')
    prop_col_names_full.append('EqW')

    properties = np.zeros((len(TDSSprop.data), len(prop_col_names_full)))

    properties = Table(properties, names=prop_col_names_full)
    properties['ra'] = TDSSprop.data['ra']
    properties['dec'] = TDSSprop.data['dec']


last_periodic_filenames = np.loadtxt(f"{prop_out_dir}periodic_objects_filenames_07-16-2020.txt", dtype="S")
last_periodic_filenames = np.array([ii.decode() for ii in last_periodic_filenames])
periodic_filenames = []
# ***********************************************
importlib.reload(vi)
importlib.reload(LCtools)
plt.ioff()
# ***********************************************
# ***********************************************
# ***********************************************
# runVartools = True
# run_onlyPyHammerChanged = True
runLS = True
plotLCerr = True
plt_resid = False
plt_subLC = True
plot_rejected = False
checkalias = True
logProblimit = -10
Nepochs_required = 10
minP = 0.1
maxP = 100.0
nterms_LS = 1  # Currently astropy only allows calculation of the False Alarm probability (FAP) for neterms=1, so this can't be changed.

log10FAP = -5
nterms = 6
checkHarmonic = True
# ***********************************************
# ***********************************************
# ***********************************************
def propID_from_LCID(LCID, LC, PropTable):
    if LC == "CSS":
        return np.where(PropTable['CSSID'] == LCID)[0][0]
    elif LC == "ZTF_g":
        return np.where(PropTable['ZTF_g_GroupID'] == LCID)[0][0]
    elif LC == "ZTF_r":
        return np.where(PropTable['ZTF_r_GroupID'] == LCID)[0][0]

# ***********************************************
# ***********************************************
# ***********************************************
# prop_id = 341
prop_id = propID_from_LCID(1152051021730, "CSS", TDSSprop.data)
ROW = TDSSprop.data[prop_id]

ra_string = '{:0>9.5f}'.format(ROW['ra'])
dec_string = '{:0=+9.5f}'.format(ROW['dec'])

is_CSS = ROW['CSSLC']
is_ZTF_g = np.isfinite(ROW['ZTF_g_GroupID'])
is_ZTF_r = np.isfinite(ROW['ZTF_r_GroupID'])

# ***********************************************
# ***********************************************

ZTF_g_lc_data = ZTF_g_LCs[(ZTF_g_LCs['GroupID'] == ROW['ZTF_g_GroupID'])]['mjd', 'mag', 'magerr']
ZTF_gflc_data, LC_stat_properties = LCtools.process_LC(ZTF_g_lc_data.copy(), fltRange=5.0)
goodQualIndex = np.where(ZTF_gflc_data['QualFlag']==True)[0]
LC_period_properties, all_ZTFg_period_properties = LCtools.perdiodSearch(ZTF_gflc_data, minP=minP, maxP=maxP, log10FAP=log10FAP, checkHarmonic=checkHarmonic)

Nterms1, phase_fit1, y_fit1, phased_t1, resid1, reduced_ChiSq1, mtf1 = all_ZTFg_period_properties['AFD']
phased_lc_data = ZTF_gflc_data[goodQualIndex]['mjd', 'mag', 'magerr'].copy()
phased_lc_data.add_column(phased_t1, name="phase", index=1)
phased_lc_data.write("/Users/benjaminroulston/Desktop/RRd_LC_ZTFg.csv", format='ascii.csv')

plt.scatter(phased_lc_data['phase'], phased_lc_data['mag'], s=1, c='k')
ax = plt.gca()
ax.invert_yaxis()
plt.show()
plt.clf()
plt.close()

# ***********************************************
# ***********************************************

ZTF_r_lc_data = ZTF_r_LCs[(ZTF_r_LCs['GroupID'] == ROW['ZTF_r_GroupID'])]['mjd', 'mag', 'magerr']
ZTF_rflc_data, LC_stat_properties = LCtools.process_LC(ZTF_r_lc_data.copy(), fltRange=5.0)
goodQualIndex = np.where(ZTF_rflc_data['QualFlag']==True)[0]
LC_period_properties, all_ZTFr_period_properties = LCtools.perdiodSearch(ZTF_rflc_data, minP=minP, maxP=maxP, log10FAP=log10FAP, checkHarmonic=checkHarmonic)

Nterms1, phase_fit1, y_fit1, phased_t1, resid1, reduced_ChiSq1, mtf1 = all_ZTFr_period_properties['AFD']
phased_lc_data = ZTF_rflc_data[goodQualIndex]['mjd', 'mag', 'magerr'].copy()
phased_lc_data.add_column(phased_t1, name="phase", index=1)
phased_lc_data.write("/Users/benjaminroulston/Desktop/TDSS_example_Variables_2/RS-CVn_LC_ZTFr.csv", format='ascii.csv')

plt.scatter(phased_lc_data['phase'], phased_lc_data['mag'], s=1, c='k')
ax = plt.gca()
ax.invert_yaxis()
plt.show()
plt.clf()
plt.close()


# ***********************************************
# ***********************************************
CSS_LC_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/CSS_LCs/csvs/"

lc_file = CSS_LC_dir+str(ROW['CSSID'])+'.dat'
CSS_lc_data = Table.read(lc_file, format='ascii', names=['mjd', 'mag', 'magerr'])


CSS_flc_data, LC_stat_properties = LCtools.process_LC(CSS_lc_data.copy(), fltRange=5.0)
goodQualIndex = np.where(CSS_flc_data['QualFlag']==True)[0]
LC_period_properties, all_CSS_period_properties = LCtools.perdiodSearch(CSS_flc_data, minP=minP, maxP=maxP, log10FAP=log10FAP, checkHarmonic=checkHarmonic)

Nterms1, phase_fit1, y_fit1, phased_t1, resid1, reduced_ChiSq1, mtf1 = all_CSS_period_properties['AFD']
phased_lc_data = CSS_flc_data[goodQualIndex]['mjd', 'mag', 'magerr'].copy()
phased_lc_data.add_column(phased_t1, name="phase", index=1)

# plt.scatter(phased_lc_data['phase'], phased_lc_data['mag'], s=1, c='k')
plt.scatter(phased_lc_data['mjd'], phased_lc_data['mag'], s=1, c='k')
ax = plt.gca()
ax.invert_yaxis()
plt.show()
plt.clf()
plt.close()

phased_lc_data.write("/Users/benjaminroulston/Desktop/TDSS_example_Variables_2/CV_nonP_LC_ZTFr.csv", format='ascii.csv')










