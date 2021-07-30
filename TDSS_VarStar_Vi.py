# %load_ext autoreload
# %autoreload 2

import matplotlib
import matplotlib.style as mplstyle
matplotlib.use('TkAGG')
mplstyle.use('fast')

matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0
matplotlib.rcParams['agg.path.chunksize'] = 10000

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
all_filenames = []
for prop_id, ROW in enumerate(tqdm.tqdm(TDSSprop.data)):
        object_ra = ROW['ra']
        object_dec = ROW['dec']
        ra_string = '{:0>9.5f}'.format(object_ra)
        dec_string = '{:0=+9.5f}'.format(object_dec)

        this_Viplot_filename = f"{ra_string}{dec_string}_Vi.pdf"
        all_filenames.append(this_Viplot_filename)

all_filenames = np.array(all_filenames)
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
checkHarmonic = True
logProblimit = -10
Nepochs_required = 10
minP = 0.1
maxP = 100.0
# ***********************************************
# ***********************************************
# ***********************************************
prop = Table.read("/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/PROGRAM_SAMPLE/2020-06-24/FINAL_FILES/TDSS_SES+PREV_DR16DR12griLT20_GAIADR2_Drake2014PerVar_VSX_CSSID_ZTFIDs_LCpointer_PyHammer_VI_ALLPROP_07-27-2020.fits")

logProb_lim = -10
CSS_periodic_index = (prop['CSS_logProb'] <= logProb_lim).data
ZTFg_periodic_index = (prop['ZTF_g_logProb'] <= logProb_lim).data
ZTFr_periodic_index = (prop['ZTF_r_logProb'] <= logProb_lim).data

logProb_limit_array = np.stack((CSS_periodic_index, ZTFg_periodic_index, ZTFr_periodic_index), axis=1)
logFAP_10 = logProb_limit_array.sum(axis=1).copy()

logProb_lim = -5
CSS_periodic_index = (prop['CSS_logProb'] <= logProb_lim).data
ZTFg_periodic_index = (prop['ZTF_g_logProb'] <= logProb_lim).data
ZTFr_periodic_index = (prop['ZTF_r_logProb'] <= logProb_lim).data

logProb_limit_array = np.stack((CSS_periodic_index, ZTFg_periodic_index, ZTFr_periodic_index), axis=1)
logFAP_5 = logProb_limit_array.sum(axis=1).copy()

logFAP_10_unique, logFAP_10_unique_counts = np.unique(logFAP_10, return_counts=True)
logFAP_5_unique, logFAP_5_unique_counts = np.unique(logFAP_5, return_counts=True)


logFAP_10_1LC_index = np.where(logFAP_10 >= 1)[0]
logFAP_5_2LC_index = np.where(logFAP_5 >= 2)[0]

only_2LC_in_5 = []
for ii in logFAP_5_2LC_index:
    if ii in logFAP_10_1LC_index:
        pass
    else:
        only_2LC_in_5.append(ii)

only_2LC_in_5 = np.array(only_2LC_in_5)

# ***********************************************
# ***********************************************
# ***********************************************

CV_index = (np.char.strip(np.char.decode( prop['subclass'])) == "CV") | (np.char.strip(np.char.decode( prop['SUBCLASS_NOQSO'])) == "CV") | (np.char.strip(np.char.decode( prop['subClass_SDSSDR12'])) == "CV")
large_Halpha_EQWe_index = (prop['EqW'] < 0).data

move_ra = prop['ra'][large_Halpha_EQWe_index]
move_dec = prop['dec'][large_Halpha_EQWe_index]
moveDir = f"/Users/benjaminroulston/Desktop/large_Halpha_EQWe/"
get_Vi_panels(move_ra, move_dec, '2020-07-16', copy=True, moveDir=moveDir)


# ***********************************************
# ***********************************************
# ***********************************************
is_C = np.array([(('dC' in ii) and ('+dC') not in ii) for ii in prop['PyHammerSpecType']])

move_ra = prop['ra'][is_C]
move_dec = prop['dec'][is_C]
moveDir = f"/Users/benjaminroulston/Desktop/TDSS_VarStar_Cstar_ViP/"
vi.get_Vi_panels(move_ra, move_dec, '2020-07-16', copy=True, moveDir=moveDir)


# ***********************************************
# ***********************************************
# ***********************************************
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for prop_id, ROW in enumerate(tqdm.tqdm(TDSSprop.data)):
        if prop_id < prop_id_last:
            properties[prop_id]['ViCompleted'] = 1
            continue
        if not ROW['CSS_or_ZTF']:
            properties[prop_id]['ViCompleted'] = 1
            continue

        any_good_LC = (ROW['CSS_Nepochs'] >= Nepochs_required) | (ROW['ZTF_g_Nepochs'] >= Nepochs_required) | (ROW['ZTF_r_Nepochs'] >= Nepochs_required)
        if not any_good_LC:
            properties[prop_id]['ViCompleted'] = 1
            continue

        # if prop_id not in only_2LC_in_5:
        #     properties[prop_id]['ViCompleted'] = 1
        #     continue

        # if not ROW['PyHammerDiff'] & run_onlyPyHammerChanged:
        #     properties[prop_id]['ViCompleted'] = 1
        #     continue
        #
        # prop_id = 341
        # ROW = TDSSprop.data[prop_id]

        # periodic_index = 6
        # prop_id = np.where(all_filenames == last_periodic_filenames[periodic_index])[0][0]
        # ROW = TDSSprop.data[prop_id]

        object_ra = ROW['ra']
        object_dec = ROW['dec']
        ra_string = '{:0>9.5f}'.format(object_ra)
        dec_string = '{:0=+9.5f}'.format(object_dec)

        this_Viplot_filename = f"{ra_string}{dec_string}_Vi.pdf"
        if this_Viplot_filename not in last_periodic_filenames:
            properties[prop_id]['ViCompleted'] = 1
            continue

        mjd = ROW['mjd']
        plate = ROW['plate']
        fiberid = ROW['fiber']
        mjd_string = '{:0>5}'.format(str(np.int(mjd)))
        plate_string = '{:0>4}'.format(str(np.int(plate)))
        fiberid_string = '{:0>4}'.format(str(np.int(fiberid)))
        long_filename = f"spec-{plate_string}-{mjd_string}-{fiberid_string}.fits"

        fig = plt.figure(figsize=(13, 9), constrained_layout=True)
        gs = GridSpec(2, 7, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1, 0.4, 1, 1])  # , hspace=0.3, wspace=0.5)
        ax1 = fig.add_subplot(gs[0, :2])  # LC
        ax2 = fig.add_subplot(gs[0, 2:4])  # Chi2 vs Amp plots OR Palerversa+2013 plot
        ax3 = fig.add_subplot(gs[0, 5:])  # CMD
        ax4 = fig.add_subplot(gs[1, :])  # spectra with lines
        fig.suptitle(f'RA= {ra_string} DEC= {dec_string}', fontsize=16)

        CSS_prop, ZTF_g_prop, ZTF_r_prop, best_LC = vi.LC_analysis(ROW, TDSSprop, CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, ax1, CSS_LC_plot_dir, ZTF_LC_plot_dir, Nepochs_required, minP=minP, maxP=maxP, log10FAP=logProblimit, checkHarmonic=checkHarmonic, plt_subLC=plt_subLC, plot_rejected=plot_rejected)
        all_LC_props = [CSS_prop, ZTF_g_prop, ZTF_r_prop]

        if best_LC == 'CSS':
            if (CSS_prop['logProb'] <= logProblimit):
                periodic_filenames.append(f"{ra_string}{dec_string}_Vi.pdf")
        elif best_LC == 'ZTF_g':
            if (ZTF_g_prop['logProb'] <= logProblimit):
                periodic_filenames.append(f"{ra_string}{dec_string}_Vi.pdf")
        elif best_LC == 'ZTF_r':
            if (ZTF_r_prop['logProb'] <= logProblimit):
                periodic_filenames.append(f"{ra_string}{dec_string}_Vi.pdf")

        for LC_ii, LCprop in enumerate(all_LC_props):
            if LCprop:
                for key, value in LCprop.items():
                    properties[prop_id][prop_col_names_prefix[LC_ii] + key] = value

        this_EqW = vi.plot_SDSSspec(ROW, TDSSprop, prop_id, spec_dir, ax4)
        properties[prop_id]['EqW'] = this_EqW

        vi.plot_middle(all_LC_props, best_LC, latestFullVartoolsRun, ax2, log10FAP=logProblimit)

        vi.plot_CMD(TDSSprop, prop_id, ax3)

        plt.savefig(f"{Vi_plots_dir}{ra_string}{dec_string}_Vi.pdf", dpi=600, bbox_inches='tight')
        # plt.show()
        plt.clf()
        plt.close()

        properties[prop_id]['ViCompleted'] = 1
        if (prop_id % 100) == 0:
            properties.write(f"{prop_out_dir}completed_Vi_prop_{datestr}.fits", format='fits', overwrite=True)
            # properties.write(prop_out_dir+"completed_Vi_prop_"+datestr+".csv", format='csv', overwrite=True)
            # np.savetxt(prop_out_dir+"completed_Vi_prop_"+datestr+".csv", properties, delimiter=",", header=prop_header, fmt="%f, %f, %i, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %i, %f, %f, %f, %f, %f, %f, %f, %f, %f,  %f")


properties.write(f"{prop_out_dir}completed_Vi_prop_{datestr}.fits", format='fits', overwrite=True)
properties.write(f"{prop_out_dir}completed_Vi_prop_{datestr}.csv", format='csv', overwrite=True)
# np.savetxt(prop_out_dir+"completed_Vi_prop_"+datestr+".csv", properties, delimiter=",", header=prop_header,     fmt="%f, %f, %i, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f %f, %f, %f, %f, %f, %f, %f, %f, %i, %f, %f, %f, %f, %f, %f, %f, %f , %f, %f")

where_no_Vi = np.logical_and(np.logical_and(properties['CSS_lc_id'] == 0, properties['ZTF_g_P'] == 0), properties['ZTF_r_P'] == 0)
properties.add_column(where_no_Vi, index=3, name="noVi")
# properties.replace_column(name="noVi", col=where_no_Vi)
properties.write(f"{prop_out_dir}completed_Vi_prop_{datestr}.fits", format='fits', overwrite=True)
properties.write(f"{prop_out_dir}completed_Vi_prop_{datestr}.csv", format='csv', overwrite=True)

periodic_filenames = np.array(periodic_filenames)
np.savetxt(f"{prop_out_dir}periodic_objects_filenames.txt", periodic_filenames, fmt="%s")
