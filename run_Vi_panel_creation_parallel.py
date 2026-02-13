import matplotlib
import matplotlib.style as mplstyle

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
from tqdm import trange, tqdm

import time

import warnings

import ResearchTools.LCtools as LCtools
import VarStar_Vi_plot_functions as vi

from multiprocessing import Pool

# matplotlib.use('TkAGG')
mplstyle.use('fast')

matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0
matplotlib.rcParams['agg.path.chunksize'] = 10000

params = {"xtick.top": True,
          "ytick.right": True,
          "xtick.direction": "inout",
          "ytick.direction": "inout"}
plt.rcParams.update(params)

spec_dir = "/Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/HARD_COPY_ORGINAL_DATA/SDSS_spec/02-26-2020/SDSSspec/"
CSS_LC_dir = "/Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/HARD_COPY_ORGINAL_DATA/CSS_LCs/csvs/"
ZTF_LC_dir = "/Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/HARD_COPY_ORGINAL_DATA/ZTF/DATA/07-27-2021/"

ZTF_filters = ['g', 'r']
ZTF_LC_file_names = [f'TDSS_VarStar_ZTFDR6_{ZTF_filter}_GroupID.fits' for ZTF_filter in ZTF_filters]
ZTF_g_LCs = Table.read(ZTF_LC_dir + ZTF_LC_file_names[0])
ZTF_r_LCs = Table.read(ZTF_LC_dir + ZTF_LC_file_names[1])

prop_out_dir, CSS_LC_plot_dir, ZTF_LC_plot_dir, Vi_plots_dir, datestr = vi.makeViDirs()

nbins = 50
# SourceFilename = "/Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/HARD_COPY_ORGINAL_DATA/PROGRAM_SAMPLE/2021-07-27/FINAL_FILES/TDSS_SES+PREV_DR16DR12griLT20_GaiaEDR3_Drake2014PerVar_CSSID_ZTFIDs_LCpointer_PyHammer_EqW_LCProps.fits"
# SourceFilename = '/Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/WORKING_DIRECTORY/Statistical_Testing/LC_stat_calculations/Control_StatsCalc/TDSS_VarStar_FINAL_Var_ALL_PROP_STATS_03-15-2022.fits'
# SourceFilename = '/Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/WORKING_DIRECTORY/Statistical_Testing/LC_stat_calculations/Control_StatsCalc/TDSS_VarStar_FINAL_Var_ALL_PROP_STATS_10-31-2022.fits'
SourceFilename = '/Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/WORKING_DIRECTORY/Statistical_Testing/LC_stat_calculations/Control_StatsCalc/TDSS_VarStar_FINAL_Var_ALL_PROP_STATS_11-07-2022.fits'
TDSSprop = vi.TDSSprop(nbins, SourceFilename=SourceFilename)

# latestFullVartoolsRun_filename = "completed_Vi_prop_2020-07-16.csv"
# latestFullVartoolsRun = vi.latestFullVartoolsRun(prop_out_dir + latestFullVartoolsRun_filename)

TDSS_Table = TDSSprop.data.copy()  # Table.read(SourceFilename)


xiyizi = [None, None, None]

for ii, best_LC in enumerate(['CSS', 'ZTF_g', 'ZTF_r']):
    all_is_periodic = TDSS_Table[best_LC + '_logProb'] <= -5
    xiyizi[ii] = vi.make_middle_heatmap(TDSS_Table, best_LC, all_is_periodic)


def make_Vi_propID(prop_id, TDSS_Table, TDSSprop, logFAP_limit=-5):
    ROW = TDSS_Table[prop_id]
    object_ra = ROW['ra_GaiaEDR3']
    object_dec = ROW['dec_GaiaEDR3']
    ra_string = '{:0>9.5f}'.format(object_ra)
    dec_string = '{:0=+9.5f}'.format(object_dec)

    this_Viplot_filename = f"{prop_id}_{ra_string}{dec_string}_Vi.pdf"

    # mjd = ROW['mjd']
    # plate = ROW['plate']
    # fiberid = ROW['fiber']
    # mjd_string = '{:0>5}'.format(str(np.int64(mjd)))
    # plate_string = '{:0>4}'.format(str(np.int64(plate)))
    # fiberid_string = '{:0>4}'.format(str(np.int64(fiberid)))
    # long_filename = f"spec-{plate_string}-{mjd_string}-{fiberid_string}.fits"

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1, 1, 1])  # , hspace=0.3, wspace=0.5)
    ax1 = fig.add_subplot(gs[0, :2])  # LC
    ax2 = fig.add_subplot(gs[0, 2:4])  # Chi2 vs Amp plots OR Palerversa+2013 plot
    ax3 = fig.add_subplot(gs[0, 4:])  # CMD
    ax4 = fig.add_subplot(gs[1, :])  # spectra with lines
    fig.suptitle(f'RA= {ra_string} DEC= {dec_string}', fontsize=16)

    best_LC = vi.find_best_LC(ROW, logFAP_limit)

    lc_plot = vi.plot_LC_panel(ROW, ax1, best_LC, CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, plot_rejected=False)
    vi.plot_middle(ROW, TDSS_Table, best_LC, xiyizi, ax2, log10FAP=logFAP_limit)
    vi.plot_CMD(TDSSprop, prop_id, ax3)
    this_EqW = vi.plot_SDSSspec(ROW, spec_dir, ax4)

    return fig, [ax1, ax2, ax3, ax4], this_Viplot_filename


def make_Vi_propID_parallel(prop_id):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        logFAP_limit = -5

        ROW = TDSS_Table[prop_id]
        object_ra = ROW['ra_GaiaEDR3']
        object_dec = ROW['dec_GaiaEDR3']
        ra_string = '{:0>9.5f}'.format(object_ra)
        dec_string = '{:0=+9.5f}'.format(object_dec)

        this_Viplot_filename = f"{prop_id}_{ra_string}{dec_string}_Vi.pdf"

        # mjd = ROW['mjd']
        # plate = ROW['plate']
        # fiberid = ROW['fiber']
        # mjd_string = '{:0>5}'.format(str(np.int64(mjd)))
        # plate_string = '{:0>4}'.format(str(np.int64(plate)))
        # fiberid_string = '{:0>4}'.format(str(np.int64(fiberid)))
        # long_filename = f"spec-{plate_string}-{mjd_string}-{fiberid_string}.fits"

        fig = plt.figure(figsize=(13, 9), constrained_layout=True)
        gs = GridSpec(2, 6, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1, 1, 1])  # , hspace=0.3, wspace=0.5)
        ax1 = fig.add_subplot(gs[0, :2])  # LC
        ax2 = fig.add_subplot(gs[0, 2:4])  # Chi2 vs Amp plots OR Palerversa+2013 plot
        ax3 = fig.add_subplot(gs[0, 4:])  # CMD
        ax4 = fig.add_subplot(gs[1, :])  # spectra with lines
        fig.suptitle(f'RA= {ra_string} DEC= {dec_string}', fontsize=16)

        best_LC = vi.find_best_LC(ROW, logFAP_limit)

        lc_plot = vi.plot_LC_panel(ROW, TDSSprop, ax1, best_LC, CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, plot_rejected=False)
        vi.plot_middle(ROW, TDSS_Table, best_LC, xiyizi, ax2, one_log10_FAP=logFAP_limit, all_log10FAP=logFAP_limit)
        vi.plot_CMD(TDSSprop, prop_id, ax3)
        this_EqW = vi.plot_SDSSspec(ROW, TDSSprop, prop_id, spec_dir, ax4)

        plt.savefig(f"{Vi_plots_dir}{this_Viplot_filename}")
        plt.clf()
        plt.close(fig)


start_index = 0
if __name__ == '__main__':
    with Pool(os.cpu_count() - 2) as pool:
        r = list(tqdm(pool.imap(make_Vi_propID_parallel, range(start_index, len(TDSS_Table))), total=len(TDSS_Table)-start_index))
