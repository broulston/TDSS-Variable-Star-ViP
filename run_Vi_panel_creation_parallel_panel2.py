import matplotlib
import matplotlib.style as mplstyle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
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


def make_Vi_propID_parallel(prop_id):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # TDSS_Table
        # TDSSprop,
        logFAP_limit = -5

        ROW = TDSS_Table[prop_id]
        object_ra = ROW['ra_GaiaEDR3']
        object_dec = ROW['dec_GaiaEDR3']
        ra_string = '{:0>9.5f}'.format(object_ra)
        dec_string = '{:0=+9.5f}'.format(object_dec)

        this_Viplot_filename = f"{prop_id}_{ra_string}{dec_string}_Vi.pdf"

        mjd = ROW['mjd']
        plate = ROW['plate']
        fiberid = ROW['fiber']
        mjd_string = '{:0>5}'.format(str(np.int64(mjd)))
        plate_string = '{:0>4}'.format(str(np.int64(plate)))
        fiberid_string = '{:0>4}'.format(str(np.int64(fiberid)))
        # long_filename = f"spec-{plate_string}-{mjd_string}-{fiberid_string}.fits"

        # isCSS = not np.ma.is_masked(ROW['CSS_Nepochs'])
        # isZTFg = not np.ma.is_masked(ROW['ZTF_g_Nepochs'])
        # isZTFr = not np.ma.is_masked(ROW['ZTF_r_Nepochs'])

        isCSS = False if np.ma.is_masked(ROW['CSS_Nepochs']) else (ROW['CSS_Nepochs'] > 10)
        isZTFg = False if np.ma.is_masked(ROW['ZTF_g_Nepochs']) else (ROW['ZTF_g_Nepochs'] > 10)
        isZTFr = False if np.ma.is_masked(ROW['ZTF_r_Nepochs']) else (ROW['ZTF_r_Nepochs'] > 10)

        LCs_name = np.array(['CSS', "ZTF_g", "ZTF_r"])
        isLCs = [isCSS, isZTFg, isZTFr]
        N_LCs = np.sum(isLCs)
        selected_LCnames = LCs_name[np.where(isLCs)[0]]


                
        fig = plt.figure(figsize=(12, 12), constrained_layout=True, dpi=600)
        gs0 = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[2.0/4, 1./4, 1/4], width_ratios=[3/4, 1/4])

        gs00 = gridspec.GridSpecFromSubplotSpec(N_LCs, 1, subplot_spec=gs0[0, 0], hspace=0.001)
        gs01 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[:, 1], hspace=0.0)
        gs10 = gridspec.GridSpecFromSubplotSpec(1, N_LCs, subplot_spec=gs0[1, 0], hspace=0.0, height_ratios=[1.0])
        gs20 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[2, 0], hspace=0.0, height_ratios=[1])

        if N_LCs==1:
            ax1 = fig.add_subplot(gs00[0, 0])
            
            ax4 = fig.add_subplot(gs10[0, 0])
        elif N_LCs==2:
            if 'CSS' in selected_LCnames:
                ax1 = fig.add_subplot(gs00[0, 0])
                ax2 = fig.add_subplot(gs00[1, 0])
            else:
                ax1 = fig.add_subplot(gs00[0, 0])
                ax2 = fig.add_subplot(gs00[1, 0], sharex=ax1)
                plt.setp(ax1.get_xticklabels(), visible=False)
            
            ax4 = fig.add_subplot(gs10[0, 0])
            ax5 = fig.add_subplot(gs10[0, 1])
        elif N_LCs==3:
            ax1 = fig.add_subplot(gs00[0, 0])
            ax2 = fig.add_subplot(gs00[1, 0])
            ax3 = fig.add_subplot(gs00[2, 0], sharex=ax2)
            plt.setp(ax2.get_xticklabels(), visible=False)
            
            ax4 = fig.add_subplot(gs10[0, 0])
            ax5 = fig.add_subplot(gs10[0, 1])
            ax6 = fig.add_subplot(gs10[0, 2])

        ax10 = fig.add_subplot(gs20[0, 0])

        ax11 = fig.add_subplot(gs01[0, 0])
        ax12 = fig.add_subplot(gs01[1, 0])
        ax13 = fig.add_subplot(gs01[2, 0])


        best_LC = vi.find_best_LC(ROW, logFAP_limit)

        if N_LCs == 1:
            lc_plot = vi.plot_LC_panel(ROW, TDSSprop, ax1, selected_LCnames[0], CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, plot_rejected=False, logFAPlimit=-np.inf)
            ax1.set_title("")
            ax1.set_ylabel(f"{selected_LCnames[0].replace('_',' ')} mag")
            ax1.legend(ax1.get_legend_handles_labels()[0], ax1.get_legend_handles_labels()[1], loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))

            lc_plot = vi.plot_LC_panel(ROW, TDSSprop, ax4, selected_LCnames[0], CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, plot_rejected=False, logFAPlimit=1.0)
            ax1.set_ylabel(f"{selected_LCnames[0].replace('_',' ')} mag")
            ax4.get_legend().remove()
        elif N_LCs == 2:
            lc_plot = vi.plot_LC_panel(ROW, TDSSprop, ax1, selected_LCnames[0], CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, plot_rejected=False, logFAPlimit=-np.inf)
            ax1.set_title("")
            ax1.set_xlabel("")
            ax1.set_ylabel(f"{selected_LCnames[0].replace('_',' ')} mag")
            ax1.legend(ax1.get_legend_handles_labels()[0], ax1.get_legend_handles_labels()[1], loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))

            lc_plot = vi.plot_LC_panel(ROW, TDSSprop, ax4, selected_LCnames[0], CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, plot_rejected=False, logFAPlimit=1.0)
            ax4.set_ylabel(f"{selected_LCnames[0].replace('_',' ')} mag")
            ax4.get_legend().remove()

            lc_plot = vi.plot_LC_panel(ROW, TDSSprop, ax2, selected_LCnames[1], CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, plot_rejected=False, logFAPlimit=-np.inf)
            ax2.set_title("")
            ax2.set_ylabel(f"{selected_LCnames[1].replace('_',' ')} mag")
            ax2.get_legend().remove()

            lc_plot = vi.plot_LC_panel(ROW, TDSSprop, ax5, selected_LCnames[1], CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, plot_rejected=False, logFAPlimit=1.0)
            ax5.set_ylabel(f"{selected_LCnames[1].replace('_',' ')} mag")
            ax5.get_legend().remove()
        elif N_LCs == 3:
            lc_plot = vi.plot_LC_panel(ROW, TDSSprop, ax1, selected_LCnames[0], CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, plot_rejected=False, logFAPlimit=-np.inf)
            ax1.set_title("")
            ax1.set_xlabel("")
            ax1.set_ylabel(f"{selected_LCnames[0].replace('_',' ')} mag")
            ax1.legend(ax1.get_legend_handles_labels()[0], ax1.get_legend_handles_labels()[1], loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))

            lc_plot = vi.plot_LC_panel(ROW, TDSSprop, ax4, selected_LCnames[0], CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, plot_rejected=False, logFAPlimit=1.0)
            ax4.set_ylabel(f"{selected_LCnames[0].replace('_',' ')} mag")
            ax4.get_legend().remove()

            lc_plot = vi.plot_LC_panel(ROW, TDSSprop, ax2, selected_LCnames[1], CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, plot_rejected=False, logFAPlimit=-np.inf)
            ax2.set_title("")
            ax2.set_xlabel("")
            ax2.set_ylabel(f"{selected_LCnames[1].replace('_',' ')} mag")
            ax2.get_legend().remove()

            lc_plot = vi.plot_LC_panel(ROW, TDSSprop, ax5, selected_LCnames[1], CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, plot_rejected=False, logFAPlimit=1.0)
            ax5.set_ylabel(f"{selected_LCnames[1].replace('_',' ')} mag")
            ax5.get_legend().remove()

            lc_plot = vi.plot_LC_panel(ROW, TDSSprop, ax3, selected_LCnames[2], CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, plot_rejected=False, logFAPlimit=-np.inf)
            ax3.set_title("")
            ax3.set_ylabel(f"{selected_LCnames[2].replace('_',' ')} mag")
            ax3.get_legend().remove()

            lc_plot = vi.plot_LC_panel(ROW, TDSSprop, ax6, selected_LCnames[2], CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, plot_rejected=False, logFAPlimit=1.0)
            ax6.set_ylabel(f"{selected_LCnames[2].replace('_',' ')} mag")
            ax6.get_legend().remove()

        vi.plot_middle(ROW, TDSS_Table, best_LC, xiyizi, ax11, one_log10_FAP=-np.inf, all_log10FAP=-10.0)
        ax11.set_title(best_LC + "\n" + ax11.get_title())
        vi.plot_middle(ROW, TDSS_Table, best_LC, xiyizi, ax12, one_log10_FAP=1.0, all_log10FAP=-10.0)
        ax12.set_title(best_LC + "\n" + ax12.get_title())
        vi.plot_CMD(TDSSprop, prop_id, ax13)
        ax13.set_title("" + ax12.get_title())
        this_EqW = vi.plot_SDSSspec(ROW, TDSSprop, prop_id, spec_dir, ax10)

        plt.savefig(f"{Vi_plots_dir}{this_Viplot_filename}", bbox_inches='tight')
        plt.clf()
        plt.close(fig)


start_index = 0
if __name__ == '__main__':
    print('in if')
    with Pool(os.cpu_count()-2) as pool:
        r = list(tqdm(pool.imap(make_Vi_propID_parallel, range(start_index, len(TDSS_Table))), total=len(TDSS_Table)-start_index))
