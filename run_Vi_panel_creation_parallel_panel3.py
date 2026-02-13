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
mplstyle.use("fast")

matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["path.simplify_threshold"] = 1.0
matplotlib.rcParams["agg.path.chunksize"] = 10000

spec_dir = "/Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/HARD_COPY_ORGINAL_DATA/SDSS_spec/02-26-2020/SDSSspec/"
CSS_LC_dir = "/Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/HARD_COPY_ORGINAL_DATA/CSS_LCs/csvs/"
ZTF_LC_dir = "/Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/HARD_COPY_ORGINAL_DATA/ZTF/DATA/07-27-2021/"

ZTF_filters = ["g", "r"]
ZTF_LC_file_names = [
    f"TDSS_VarStar_ZTFDR6_{ZTF_filter}_GroupID.fits" for ZTF_filter in ZTF_filters
]
ZTF_g_LCs = Table.read(ZTF_LC_dir + ZTF_LC_file_names[0])
ZTF_r_LCs = Table.read(ZTF_LC_dir + ZTF_LC_file_names[1])

prop_out_dir, CSS_LC_plot_dir, ZTF_LC_plot_dir, Vi_plots_dir, datestr = vi.makeViDirs()

nbins = 50
# SourceFilename = "/Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/HARD_COPY_ORGINAL_DATA/PROGRAM_SAMPLE/2021-07-27/FINAL_FILES/TDSS_SES+PREV_DR16DR12griLT20_GaiaEDR3_Drake2014PerVar_CSSID_ZTFIDs_LCpointer_PyHammer_EqW_LCProps.fits"
# SourceFilename = '/Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/WORKING_DIRECTORY/Statistical_Testing/LC_stat_calculations/Control_StatsCalc/TDSS_VarStar_FINAL_Var_ALL_PROP_STATS_03-15-2022.fits'
# SourceFilename = '/Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/WORKING_DIRECTORY/Statistical_Testing/LC_stat_calculations/Control_StatsCalc/TDSS_VarStar_FINAL_Var_ALL_PROP_STATS_10-31-2022.fits'
# SourceFilename = '/Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/WORKING_DIRECTORY/Statistical_Testing/LC_stat_calculations/Control_StatsCalc/TDSS_VarStar_FINAL_Var_ALL_PROP_STATS_11-10-2022.fits'
SourceFilename = "/Users/benjaminroulston/Dropbox/Research/Manuscripts/tdss-varstar-paper/src/data/TDSS_VarStar_FINAL_Var_ALL_PROP_STATS_dealiased.fits"

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=u.UnitsWarning)
    TDSSprop = vi.TDSSprop(nbins, SourceFilename=SourceFilename)

# latestFullVartoolsRun_filename = "completed_Vi_prop_2020-07-16.csv"
# latestFullVartoolsRun = vi.latestFullVartoolsRun(prop_out_dir + latestFullVartoolsRun_filename)

TDSS_Table = TDSSprop.data.copy()  # Table.read(SourceFilename)

xiyizi = [None, None, None]

for ii, best_LC in enumerate(["CSS", "ZTF_g", "ZTF_r"]):
    all_is_periodic = TDSS_Table[best_LC + "_logProb"] <= -5
    xiyizi[ii] = vi.make_middle_heatmap(TDSS_Table, best_LC, all_is_periodic)


def make_Vi_propID_parallel(prop_id):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # TDSS_Table
        # TDSSprop,
        logFAP_limit = -5

        ROW = TDSS_Table[prop_id]
        object_ra = ROW["ra_GaiaEDR3"]
        object_dec = ROW["dec_GaiaEDR3"]
        ra_string = "{:0>9.5f}".format(object_ra)
        dec_string = "{:0=+9.5f}".format(object_dec)

        c = coords.ICRS(object_ra * u.deg, object_dec * u.deg)
        rahmsstr = c.ra.to_string(u.hour, precision=2, pad=True)
        decdmsstr = c.dec.to_string(u.degree, alwayssign=True, precision=2, pad=True)

        this_Viplot_filename = f"{prop_id}_{ra_string}{dec_string}_Vi.pdf"

        mjd = ROW["mjd"]
        plate = ROW["plate"]
        fiberid = ROW["fiber"]
        mjd_string = "{:0>5}".format(str(np.int64(mjd)))
        plate_string = "{:0>4}".format(str(np.int64(plate)))
        fiberid_string = "{:0>4}".format(str(np.int64(fiberid)))
        # long_filename = f"spec-{plate_string}-{mjd_string}-{fiberid_string}.fits"

        # isCSS = not np.ma.is_masked(ROW['CSS_Nepochs'])
        # isZTFg = not np.ma.is_masked(ROW['ZTF_g_Nepochs'])
        # isZTFr = not np.ma.is_masked(ROW['ZTF_r_Nepochs'])

        # isCSS = False if np.ma.is_masked(ROW['CSS_Nepochs']) else (ROW['CSS_Nepochs'] > 10)
        isZTFg = (
            False
            if np.ma.is_masked(ROW["ZTF_g_Nepochs"])
            else (ROW["ZTF_g_Nepochs"] > 10)
        )
        isZTFr = (
            False
            if np.ma.is_masked(ROW["ZTF_r_Nepochs"])
            else (ROW["ZTF_r_Nepochs"] > 10)
        )

        LCs_name = np.array(["ZTF_g", "ZTF_r"])
        isLCs = [isZTFg, isZTFr]
        N_LCs = np.sum(isLCs)
        selected_LCnames = LCs_name[np.where(isLCs)[0]]

        params = {
            "xtick.top": True,
            "ytick.right": True,
            "xtick.direction": "inout",
            "ytick.direction": "inout",
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "xtick.major.size": 10.0,
            "xtick.minor.size": 5.0,
            "ytick.labelsize": 20,
            "ytick.major.size": 10.0,
            "ytick.minor.size": 5.0,
        }
        plt.rcParams.update(params)

        fig = plt.figure(figsize=(27, 20.86), constrained_layout=True, dpi=600)
        gs0 = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1])

        gs00 = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=gs0[0, 0], hspace=0.0
        )
        gs01 = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=gs0[1, 0], hspace=0.0
        )
        gs10 = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=gs0[2, 0], hspace=0.0
        )

        if isZTFg:
            ax1 = fig.add_subplot(gs00[0, 0])
            ax2 = fig.add_subplot(gs00[0, 1])
            ax3 = fig.add_subplot(gs00[0, 2])
            ax4 = fig.add_subplot(gs00[0, 3])
            r_line_split = ""
        else:
            fig.text(0.45, 5 / 6, "No ZTF g Data", fontsize=30)
            r_line_split = ""

        if isZTFr:
            ax5 = fig.add_subplot(gs01[0, 0])
            ax6 = fig.add_subplot(gs01[0, 1])
            ax7 = fig.add_subplot(gs01[0, 2])
            ax8 = fig.add_subplot(gs01[0, 3])
            g_line_split = ""
        else:
            fig.text(0.45, 3 / 6, "No ZTF r Data", fontsize=30)
            g_line_split = ""

        if N_LCs == 2:
            title_fontsize = 15
        else:
            title_fontsize = 15

        ax9 = fig.add_subplot(gs10[0, :3])
        ax10 = fig.add_subplot(gs10[0, 3])

        # best_LC = vi.find_best_LC(ROW, logFAP_limit)

        g_title = ""
        r_title = ""
        if ROW["isDrake"]:
            drake_title = f"$-$ Drake: P={ROW['Drake_Per']} $|$ Amp={str(np.round(ROW['Drake_Vamp'], 2))} $|$ VarType={TDSSprop.Drake_num_to_vartype[(ROW['Drake_Cl'] - 1).astype(int)]['Var_Type'].strip()}"
        else:
            drake_title = ""

        if isZTFg:
            lc_plot = vi.plot_LC_panel(
                ROW,
                TDSSprop,
                ax1,
                "ZTF_g",
                CSS_LC_dir,
                ZTF_g_LCs,
                ZTF_r_LCs,
                plot_rejected=False,
                logFAPlimit=-np.inf,
            )
            ax1.set_title("")
            ax1.set_ylabel(f"ZTF g mag")
            ax1.legend(
                ax1.get_legend_handles_labels()[0],
                ax1.get_legend_handles_labels()[1],
                loc="lower center",
                ncol=3,
                frameon=False,
                bbox_to_anchor=(0.5, 1.0),
            )
            # ax1.text(0.02, 1.01, "This is a test of $\textcolor{blue}{diffeent}$ $\textcolor{red}{colors}$ $\textcolor{black}{in}$ text.", fontsize=14, transform=ax1.transAxes)

            lc_plot = vi.plot_LC_panel(
                ROW,
                TDSSprop,
                ax2,
                "ZTF_g",
                CSS_LC_dir,
                ZTF_g_LCs,
                ZTF_r_LCs,
                plot_rejected=False,
                logFAPlimit=1.0,
            )
            ax2.set_ylabel(f"ZTF g mag")
            ax2.get_legend().remove()
            ax2.set_title("")

            vi.plot_middle(
                ROW,
                TDSS_Table,
                "ZTF_g",
                xiyizi,
                ax3,
                one_log10_FAP=-np.inf,
                all_log10FAP=-10.0,
            )
            # ax3.set_title("ZTF g" + "\n" + ax3.get_title())
            ax3.set_title("")

            vi.plot_middle(
                ROW,
                TDSS_Table,
                "ZTF_g",
                xiyizi,
                ax4,
                one_log10_FAP=1.0,
                all_log10FAP=-10.0,
            )
            # ax4.set_title("ZTF g" + "\n" + ax4.get_title())
            ax4.set_title("")

            g_title = f"ZTF g GroupID: {ROW['ZTF_GroupID']} $|$ Ngood={ROW['ZTF_g_ngood']} $|$ Nreject={ROW['ZTF_g_nrejects']} $|$ Tspan={np.int32(ROW['ZTF_g_Tspan100'])} $|$ \
            a95={np.round(ROW['ZTF_g_a95'], 2)} $|$ $\chi^2$={np.round(ROW['ZTF_g_Chi2'], 2)} $|$ log($\chi^2$)={np.round(np.log10(ROW['ZTF_g_Chi2']), 2)} $|$ VarStat={np.round(ROW['ZTF_g_VarStat'], 1)} $|$ StetsonJ={np.round(ROW['ZTF_g_stetson_j'], 3)} $|$ con={np.round(ROW['ZTF_g_Con'], 3)} $|$ \
            {g_line_split}m={np.round(ROW['ZTF_g_m'], 4)}, b={np.round(ROW['ZTF_g_b_lin'], 4)}, $\chi^2_{{lin}}$={np.round(ROW['ZTF_g_chi2_lin'], 2)} $|$ \
            {g_line_split}a={np.round(ROW['ZTF_g_a'], 4)}, b={np.round(ROW['ZTF_g_b_quad'], 4)}, c={np.round(ROW['ZTF_g_c'], 4)}, $\chi^2_{{quad}}$={np.round(ROW['ZTF_g_chi2_quad'], 2)} $|$ \
            {g_line_split}logProb={np.round(ROW['ZTF_g_logProb'], 1)} $|$ P={np.round(ROW['ZTF_g_P'], 6)}w{np.int32(ROW['ZTF_g_time_whittened'])} $|$ log(P)={np.round(np.log10(ROW['ZTF_g_P']), 2)} $|$ Amp={np.round(ROW['ZTF_g_Amp'], 2)} $|$ log(Amp)={np.round(np.log10(ROW['ZTF_g_Amp']), 2)} $|$ Skew={np.round(ROW['ZTF_g_lc_skew'], 2)}"

        if isZTFr:
            lc_plot = vi.plot_LC_panel(
                ROW,
                TDSSprop,
                ax5,
                "ZTF_r",
                CSS_LC_dir,
                ZTF_g_LCs,
                ZTF_r_LCs,
                plot_rejected=False,
                logFAPlimit=-np.inf,
            )
            ax5.set_title("")
            ax5.set_ylabel(f"ZTF r mag")
            ax5.get_legend().remove()

            lc_plot = vi.plot_LC_panel(
                ROW,
                TDSSprop,
                ax6,
                "ZTF_r",
                CSS_LC_dir,
                ZTF_g_LCs,
                ZTF_r_LCs,
                plot_rejected=False,
                logFAPlimit=1.0,
            )
            ax6.set_ylabel(f"ZTF r mag")
            ax6.get_legend().remove()
            ax6.set_title("")

            vi.plot_middle(
                ROW,
                TDSS_Table,
                "ZTF_r",
                xiyizi,
                ax7,
                one_log10_FAP=-np.inf,
                all_log10FAP=-10.0,
            )
            # ax7.set_title("ZTF r" + "\n" + ax7.get_title())
            ax7.set_title("")

            vi.plot_middle(
                ROW,
                TDSS_Table,
                "ZTF_r",
                xiyizi,
                ax8,
                one_log10_FAP=1.0,
                all_log10FAP=-10.0,
            )
            # ax8.set_title("ZTF r" + "\n" + ax7.get_title())
            ax8.set_title("")

            r_title = f"ZTF r GroupID: {ROW['ZTF_GroupID']} $|$ Ngood={ROW['ZTF_r_ngood']} $|$ Nreject={ROW['ZTF_r_nrejects']} $|$ Tspan={np.int32(ROW['ZTF_r_Tspan100'])} $|$ \
            a95={np.round(ROW['ZTF_r_a95'], 2)} $|$ $\chi^2$={np.round(ROW['ZTF_r_Chi2'], 2)} $|$ log($\chi^2$)={np.round(np.log10(ROW['ZTF_r_Chi2']), 2)} $|$ VarStat={np.round(ROW['ZTF_r_VarStat'], 1)} $|$ StetsonJ={np.round(ROW['ZTF_r_stetson_j'], 3)} $|$ con={np.round(ROW['ZTF_r_Con'], 3)} $|$ \
            {r_line_split}m={np.round(ROW['ZTF_r_m'], 4)}, b={np.round(ROW['ZTF_r_b_lin'], 4)}, $\chi^2_{{lin}}$={np.round(ROW['ZTF_r_chi2_lin'], 2)} $|$ \
            {r_line_split}a={np.round(ROW['ZTF_r_a'], 4)}, b={np.round(ROW['ZTF_r_b_quad'], 4)}, c={np.round(ROW['ZTF_r_c'], 4)}, $\chi^2_{{quad}}$={np.round(ROW['ZTF_r_chi2_quad'], 2)} $|$ \
            {r_line_split}logProb={np.round(ROW['ZTF_r_logProb'], 1)} $|$ P={np.round(ROW['ZTF_r_P'], 6)}w{np.int32(ROW['ZTF_r_time_whittened'])} $|$ log(P)={np.round(np.log10(ROW['ZTF_r_P']), 2)} $|$ Amp={np.round(ROW['ZTF_r_Amp'], 2)} $|$ log(Amp)={np.round(np.log10(ROW['ZTF_r_Amp']), 2)} $|$ Skew={np.round(ROW['ZTF_r_lc_skew'], 2)}"

        vi.plot_CMD(TDSSprop, prop_id, ax10)
        # ax13.set_title("" + ax12.get_title())
        this_EqW = vi.plot_SDSSspec(ROW, TDSSprop, prop_id, spec_dir, ax9)

        fig.suptitle(
            t=f"RA:{ra_string} DEC:{dec_string} ({rahmsstr}, {decdmsstr}) {drake_title}\n{g_title} \n {r_title}",
            x=0.5,
            y=1.035,
            fontsize=title_fontsize,
        )

        plt.savefig(f"{Vi_plots_dir}{this_Viplot_filename}", bbox_inches="tight")
        plt.clf()
        plt.close(fig)


start_index = 0
# make_Vi_propID_parallel(0)
# make_Vi_propID_parallel(1)
# make_Vi_propID_parallel(3689)

# is_periodic = (TDSS_Table['ZTF_g_logProb'] <= -5).data | (TDSS_Table['ZTF_r_logProb'] <= -5).data
# is_periodic_index = np.where(is_periodic)[0]

is_periodic = np.where((TDSS_Table["ZTF_g_is_P"] | TDSS_Table["ZTF_r_is_P"]))[0]
is_notperiodic = np.where(~(TDSS_Table["ZTF_g_is_P"] | TDSS_Table["ZTF_r_is_P"]))[0]

not_P_Table = TDSS_Table[is_notperiodic]

not_P_filenames = []
for ii, prop_id in enumerate(is_notperiodic):
    ROW = not_P_Table[ii]
    object_ra = ROW["ra_GaiaEDR3"]
    object_dec = ROW["dec_GaiaEDR3"]
    ra_string = "{:0>9.5f}".format(object_ra)
    dec_string = "{:0=+9.5f}".format(object_dec)

    c = coords.ICRS(object_ra * u.deg, object_dec * u.deg)
    rahmsstr = c.ra.to_string(u.hour, precision=2, pad=True)
    decdmsstr = c.dec.to_string(u.degree, alwayssign=True, precision=2, pad=True)

    not_P_filenames.append(f"{prop_id}_{ra_string}{dec_string}_Vi.pdf")

not_P_filenames = np.array(not_P_filenames)

upload_Vi_T = Table()
upload_Vi_T.add_column(not_P_filenames, name="filename")
upload_Vi_T.add_column(not_P_Table["ra_GaiaEDR3"], name="ra")
upload_Vi_T.add_column(not_P_Table["dec_GaiaEDR3"], name="dec")

upload_Vi_T.write(
    "/Users/benjaminroulston/Desktop/nonP_IDcoords.csv",
    format="ascii.csv",
    overwrite=True,
)

# for prop_id in tqdm(is_periodic_index):
#     make_Vi_propID_parallel(prop_id)

if __name__ == "__main__":
    with Pool(os.cpu_count() - 2) as pool:
        r = list(
            tqdm(
                pool.imap(make_Vi_propID_parallel, range(start_index, len(TDSS_Table))),
                total=len(TDSS_Table) - start_index,
            )
        )

    # runDate = '/Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/WORKING_DIRECTORY/Vi/2023-03-24'
    # Vi_plots_dir = f'{runDate}/Vi_plots/'

    non_periodic_dir = f"{Vi_plots_dir}Non-Periodic"
    periodic_dir = f"{Vi_plots_dir}Periodic"

    if not os.path.exists(non_periodic_dir):
        os.makedirs(non_periodic_dir)
    if not os.path.exists(periodic_dir):
        os.makedirs(periodic_dir)

    # TDSS_Table = Table.read("/Users/benjaminroulston/Dropbox/Research/Manuscripts/tdss-varstar-paper/src/data/TDSS_VarStar_FINAL_Var_ALL_PROP_STATS_dealiased.fits")

    # is_periodic = np.where((TDSS_Table["ZTF_g_is_P"] | TDSS_Table["ZTF_r_is_P"]))[0]
    # is_notperiodic = np.where(~(TDSS_Table["ZTF_g_is_P"] | TDSS_Table["ZTF_r_is_P"]))[0]

    all_filenames = []
    per_filenames = []
    nonper_filenames = []
    for ii, ROW in enumerate(TDSS_Table):
        object_ra = ROW["ra_GaiaEDR3"]
        object_dec = ROW["dec_GaiaEDR3"]
        ra_string = "{:0>9.5f}".format(object_ra)
        dec_string = "{:0=+9.5f}".format(object_dec)

        this_Viplot_filename = f"{Vi_plots_dir}{ii}_{ra_string}{dec_string}_Vi.pdf"
        all_filenames.append(this_Viplot_filename)

        if ROW["ZTF_g_is_P"] | ROW["ZTF_r_is_P"]:
            new_Viplot_filename = f"{periodic_dir}/{ii}_{ra_string}{dec_string}_Vi.pdf"
            cp_cmd = os.system(f"cp {this_Viplot_filename} {new_Viplot_filename}")
            per_filenames.append(new_Viplot_filename)
        else:
            new_Viplot_filename = (
                f"{non_periodic_dir}/{ii}_{ra_string}{dec_string}_Vi.pdf"
            )
            cp_cmd = os.system(f"cp {this_Viplot_filename} {new_Viplot_filename}")
            nonper_filenames.append(new_Viplot_filename)

    # all_filenames = np.array(all_filenames)
    # per_filenames = np.array(per_filenames)
    # nonper_filenames = np.array(nonper_filenames)
