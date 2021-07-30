import matplotlib
matplotlib.use('TkAgg')
import matplotlib.style as mplstyle
mplstyle.use('fast')

from matplotlib.ticker import NullFormatter  # useful for `logit` scale
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg

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
from astropy.coordinates import SkyCoord


import importlib
import tqdm

import time

import warnings

import ResearchTools.LCtools as LCtools
import VarStar_Vi_plot_functions as vi

from astropy.timeseries import LombScargle
from sklearn.cluster import MeanShift, estimate_bandwidth

# ------------------------------- ViP Initialization CODE -------------------------------

def freq2per(frequency, period_unit=u.d):
    return (frequency**-1).to(period_unit)


def per2freq(period, frequency_unit=u.microHertz):
    return (period**-1).to(frequency_unit)


spec_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/SDSS_spec/02-26-2020/SDSSspec/"
CSS_LC_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/CSS_LCs/csvs/"
ZTF_LC_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/ZTF/DATA/06-24-2020/"

ZTF_filters = ['g', 'r']
ZTF_LC_file_names = [f'TDSS_SES+PREV_DR16DR12griLT20_GAIADR2_Drake2014PerVar_ZTF_{ZTF_filter}_epochGT10_GroupID.fits' for ZTF_filter in ZTF_filters]
ZTF_g_LCs = Table.read(ZTF_LC_dir + ZTF_LC_file_names[0])
ZTF_r_LCs = Table.read(ZTF_LC_dir + ZTF_LC_file_names[1])

prop_out_dir, CSS_LC_plot_dir, ZTF_LC_plot_dir, Vi_plots_dir, datestr = vi.makeViDirs()
nbins = 50
TDSSprop = vi.TDSSprop(nbins)
latestFullVartoolsRun_filename = "completed_Vi_prop_2020-07-16.csv"
latestFullVartoolsRun = vi.latestFullVartoolsRun(prop_out_dir + latestFullVartoolsRun_filename)

hasViRun, prop_id_last, properties = vi.checkViRun()  # if Vi has run, this will find where it let off and continue propid from there

prop_col_names_prefix = ['CSS_', 'ZTF_g_', 'ZTF_r_']

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
nterms_LS = 1

prop = Table.read("/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/PROGRAM_SAMPLE/2020-06-24/FINAL_FILES/TDSS_SES+PREV_DR16DR12griLT20_GAIADR2_Drake2014PerVar_VSX_CSSID_ZTFIDs_LCpointer_PyHammer_VI_ALLPROP_07-27-2020.fits")


def get_objectLC(prop_id, TDSSprop, LCsurvey):
    # prop_id = 5 # 12
    ROW = TDSSprop[prop_id]

    object_ra = ROW['ra']
    object_dec = ROW['dec']
    ra_string = '{:0>9.5f}'.format(object_ra)
    dec_string = '{:0=+9.5f}'.format(object_dec)

    is_CSS = ROW['CSSLC']
    is_ZTF_g = np.isfinite(ROW['ZTF_g_GroupID'])
    is_ZTF_r = np.isfinite(ROW['ZTF_r_GroupID'])

    if (LCsurvey == 'CSS') and is_CSS:
        lc_file = CSS_LC_dir+str(ROW['CSSID'])+'.dat'
        CSS_lc_data = Table.read(lc_file, format='ascii', names=['mjd', 'mag', 'magerr'])
        CSS_lc_data.sort('mjd')
        return ra_string, dec_string, CSS_lc_data
    if (LCsurvey == 'ZTF_g') and is_ZTF_g:
        ZTF_g_lc_data = ZTF_g_LCs[(ZTF_g_LCs['GroupID'] == ROW['ZTF_g_GroupID'])]['mjd', 'mag', 'magerr']
        ZTF_g_lc_data.sort('mjd')
        return ra_string, dec_string, ZTF_g_lc_data
    if (LCsurvey == 'ZTF_r') and is_ZTF_r:
        ZTF_r_lc_data = ZTF_r_LCs[(ZTF_r_LCs['GroupID'] == ROW['ZTF_r_GroupID'])]['mjd', 'mag', 'magerr']
        ZTF_r_lc_data.sort('mjd')
        return ra_string, dec_string, ZTF_r_lc_data

    return None, None, None


def process_LC(ra_string, dec_string, lc_data, LCsurvey):
    flc_data, LC_stat_properties = LCtools.process_LC(lc_data.copy(), fltRange=5.0)
    select_properties, all_properties = LCtools.perdiodSearch(flc_data, minP=0.1, maxP=100.0)

    # {'P': best_period, 'omega_best': omega_best, 'is_Periodic': is_Periodic,
    #                   'logProb': best_period_FAP, 'Amp': Amp, 'isAlias': isAlias,
    #                   'time_whittened': time_whittened, 'ls': ls, 'frequency': frequency, 'power': power,
    #                   'minP': minP, 'maxP': maxP, 'AFD': AFD_data, 't0': t0}

    P = all_properties["P"]
    freq_grid = all_properties['frequency']
    power = all_properties['power']
    logFAP_limit = -10
    FAP_power_peak = all_properties['ls'].false_alarm_level(10**logFAP_limit)
    df = (1 * u.d)**-1

    title = "RA: {!s} DEC: {!s} {!s}".format(ra_string, dec_string, LCsurvey)

    return flc_data, P, freq_grid, power, FAP_power_peak, logFAP_limit, df, title, all_properties


# LCsurvey = 'CSS'
# ra_string, dec_string, lc_data = get_objectLC(5, TDSSprop, LCsurvey)
# flc_data, P, freq_grid, power, FAP_power_peak, logFAP_limit, df, title, all_properties = process_LC(ra_string, dec_string, lc_data, LCsurvey)
# fig = plot_LC_analysis_ALLaliases(flc_data, P, freq_grid, power, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, df=df, title=title)


# def plot_LC_analysis_ALLaliases(lc_data, P, frequency, power, FAP_power_peak=None, logFAP_limit=None, df=None, title=""):
#     fig = plt.figure(figsize=(18, 9), constrained_layout=False)
#     gs = GridSpec(6, 4, figure=fig)

#     ax1 = fig.add_subplot(gs[0, :2])
#     ax2 = fig.add_subplot(gs[1, :2])

#     ax3 = fig.add_subplot(gs[:2, 2])

#     ax4 = fig.add_subplot(gs[:2, 3])

#     ax5 = fig.add_subplot(gs[2:4, 0])
#     ax6 = fig.add_subplot(gs[2:4, 1])
#     ax7 = fig.add_subplot(gs[2:4, 2])
#     ax8 = fig.add_subplot(gs[2:4, 3])

#     ax9 = fig.add_subplot(gs[4:, 0])
#     ax10 = fig.add_subplot(gs[4:, 1])
#     ax11 = fig.add_subplot(gs[4:, 2])
#     ax12 = fig.add_subplot(gs[4:, 3])

#     LCtools.plot_powerspec(frequency, power, ax1=ax1, ax2=ax2, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, alias_df=df, title=title)

#     LCtools.plt_any_lc_ax(lc_data, P, is_Periodic=False, ax=ax3, title="", phasebin=True, bins=25)

#     LCtools.plt_any_lc_ax(lc_data, 1.0*P, is_Periodic=True, ax=ax4, title="", phasebin=True, bins=25)

#     LCtools.plt_any_lc_ax(lc_data, 0.5*P, is_Periodic=True, ax=ax5, title="(1/2)", phasebin=True, bins=25)
#     LCtools.plt_any_lc_ax(lc_data, 2.0* P, is_Periodic=True, ax=ax9, title="2", phasebin=True, bins=25)

#     LCtools.plt_any_lc_ax(lc_data, (1/3)*P, is_Periodic=True, ax=ax6, title="(1/3)", phasebin=True, bins=25)
#     LCtools.plt_any_lc_ax(lc_data, 3.0* P, is_Periodic=True, ax=ax10, title="3", phasebin=True, bins=25)

#     LCtools.plt_any_lc_ax(lc_data, 0.25*P, is_Periodic=True, ax=ax7, title="(1/4)", phasebin=True, bins=25)
#     LCtools.plt_any_lc_ax(lc_data, 4.0* P, is_Periodic=True, ax=ax11, title="4", phasebin=True, bins=25)

#     LCtools.plt_any_lc_ax(lc_data, 0.2*P, is_Periodic=True, ax=ax8, title="(1/5)", phasebin=True, bins=25)
#     LCtools.plt_any_lc_ax(lc_data, 5.0* P, is_Periodic=True, ax=ax12, title="5", phasebin=True, bins=25)

#     ax3.set_title("")
#     fig.tight_layout()
#     return fig


def plot_LC_analysis_ALLaliases(lc_data, P, frequency, power, FAP_power_peak=None, logFAP_limit=None, df=None, title=""):
    fig = plt.figure(figsize=(18, 9), constrained_layout=False)
    gs = GridSpec(2, 6, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0:2])

    ax2 = fig.add_subplot(gs[0, 2:4])

    ax3 = fig.add_subplot(gs[0, 4:6])

    ax4 = fig.add_subplot(gs[1, 1:3])

    ax5 = fig.add_subplot(gs[1, 3:5])

    LCtools.plt_any_lc_ax(lc_data, 1.0*P, is_Periodic=True, ax=ax2, title="", phasebin=True, bins=25)

    LCtools.plt_any_lc_ax(lc_data, 0.5*P, is_Periodic=True, ax=ax1, title="(1/2)", phasebin=True, bins=25)
    LCtools.plt_any_lc_ax(lc_data, 2.0* P, is_Periodic=True, ax=ax3, title="2", phasebin=True, bins=25)

    LCtools.plt_any_lc_ax(lc_data, (1/3)*P, is_Periodic=True, ax=ax4, title="(1/3)", phasebin=True, bins=25)
    LCtools.plt_any_lc_ax(lc_data, 3.0* P, is_Periodic=True, ax=ax5, title="3", phasebin=True, bins=25)

    fig.tight_layout()
    return fig


# ------------------------------- START OF YOUR MATPLOTLIB CODE -------------------------------


def draw_plot(n):
    fig = matplotlib.figure.Figure(figsize=(18, 9), dpi=100)
    t = np.arange(0, 3, .01)
    fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t / n))
    #fig.tight_layout()
    return fig

# def draw_plot(lc_data):
#     fig = plt.figure(figsize=(5, 4))
#     fig.add_subplot(111).scatter(lc_data['mjd'], lc_data['mag'])
#     return fig


# fig = draw_plot(flc_data)


# ------------------------------- END OF YOUR MATPLOTLIB CODE -------------------------------

# ------------------------------- Beginning of Matplotlib helper code -----------------------


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


# ------------------------------- Beginning of GUI CODE -------------------------------
# fig = draw_plot(n)
# fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
vartypes = np.array(['RRab', 'RRc', 'EA', 'EB/EW', 'Single Min', 'Delta Scuti', 'Unknown', 'Non-periodic'])
harmonics = np.array([1 / 3, 1 / 2, 1, 2, 3])
LCsurvey = 'CSS'

prop = Table.read("/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/PROGRAM_SAMPLE/2020-06-24/FINAL_FILES/TDSS_SES+PREV_DR16DR12griLT20_GAIADR2_Drake2014PerVar_VSX_CSSID_ZTFIDs_LCpointer_PyHammer_VI_ALLPROP_07-27-2020.fits")
periodic_prop = Table.read("/Users/benjaminroulston/Desktop/completed_Vi_prop_2020-08-12_speedtest_PeriodOnly.fits")

all_coords = SkyCoord(ra=prop['ra'] * u.degree, dec=prop['dec'] * u.degree, frame='icrs')
periodic_coords = SkyCoord(ra=periodic_prop['ra'] * u.degree, dec=periodic_prop['dec'] * u.degree, frame='icrs')

all_prop_index = []
for ii in range(len(periodic_prop)):
    this_prop_index = np.argmin(all_coords.separation(periodic_coords[ii]))
    all_prop_index.append(this_prop_index)
    for key in periodic_prop.columns.keys()[4:]:
        prop[this_prop_index][key] = periodic_prop[ii][key]

all_prop_index = np.array(all_prop_index)
prop.add_column(col=prop['gmag_SDSSDR12'] - prop['imag_SDSSDR12'], name='gmi')

lc_prefixs = ['CSS_', 'ZTF_g_', 'ZTF_r_']
prop_col_names = ['lc_id', 'P', 'logProb', 'Amp', 'isAlias', 'time_whittened']

this_lc_index = np.where(prop[all_prop_index][f"{LCsurvey}_P"] != 0)[0]


def run_plot(n, prop, LCsurvey):
    ra_string, dec_string, lc_data = get_objectLC(n, prop, LCsurvey)
    if ra_string is None:
        return None, None
    flc_data, P, freq_grid, power, FAP_power_peak, logFAP_limit, df, title, all_properties = process_LC(ra_string, dec_string, lc_data, LCsurvey)
    fig = plot_LC_analysis_ALLaliases(flc_data, P, freq_grid, power, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, df=df, title=title)
    return fig, all_properties


ii = 0  # ii runs over the length of the peropdic objects
n = all_prop_index[this_lc_index[ii]]  # n convets ii from the range of only periodic objects into the entire 23,595 prop Table
total_N = this_lc_index.size

period_alias_frac = np.ones(total_N)
variable_type = np.empty(total_N, dtype="<U15")

prop_col_names = ['lc_id', 'P', 'logProb', 'Amp', 'isAlias', 'time_whittened']
recalc_stats_keys = ["ra", "dec"]
recalc_stats_keys.extend([f"{LCsurvey}_{col}" for col in prop_col_names])

new_Lc_prop = prop[all_prop_index[this_lc_index]][recalc_stats_keys].copy()

new_Lc_prop.add_column(np.ones(len(new_Lc_prop)), name=f"{LCsurvey}_Pharm")
new_Lc_prop.add_column(np.empty(len(new_Lc_prop), dtype="<U15"), name=f"{LCsurvey}_VarType")

col_HarmonicSelector = sg.Col([[sg.Radio('1/3 P', "Harmonic", default=False, key="-1/3P-")],
                               [sg.Radio('1/2 P', "Harmonic", default=False, key="-1/2P-")],
                               [sg.Radio('    P', "Harmonic", default=True, key="-1P-")],
                               [sg.Radio('   2P', "Harmonic", default=False, key="-2P-")],
                               [sg.Radio('   3P', "Harmonic", default=False, key="-3P-")]])

col_VarTypeSelector = sg.Col([[sg.Radio('RRab', "Type", default=False, key="-RRab-")],
                              [sg.Radio('RRc', "Type", default=False, key="-RRc-")],
                              [sg.Radio('EA', "Type", default=False, key="-EA-")],
                              [sg.Radio('EB/EW', "Type", default=False, key="-EB/EW-")],
                              [sg.Radio('Single Min', "Type", default=False, key="-SingleMin-")],
                              [sg.Radio('Delta Scuti', "Type", default=False, key="-DeltaScuti-")],
                              [sg.Radio('Unknown', "Type", default=True, key="-Unknown-")],
                              [sg.Radio('Non-periodic', "Type", default=True, key="-Non-periodic-")]])

col_previous = sg.Col([[sg.Button("Previous Object")], [sg.Button("Previous LC Survey")]])
col_next = sg.Col([[sg.Button("Next Object")], [sg.Button("Next LC Survey")]])

sg.theme('Default')
title = "TDSS VarStar Period ViP"

figure_w, figure_h = 2048, 1280
plt_w, plt_h = 2000, 800
layout = [[sg.Text(title)],
          [sg.Canvas(key='-CANVAS-', size=(plt_w, plt_h))],
          [sg.Button('Exit')],
          [col_HarmonicSelector, col_VarTypeSelector, sg.Button("Previous"), sg.Button("Next")]]

# create the form and show it without the plot
window = sg.Window('TDSS Variable Star Visual Inspection - Period Aliasing', layout,
                   size=(figure_w, figure_h), resizable=True, finalize=True,
                   element_justification='center', font='Helvetica 18')

# add the plot to the window

ra_string, dec_string, lc_data = get_objectLC(n, prop, LCsurvey)
flc_data, P, freq_grid, power, FAP_power_peak, logFAP_limit, df, title, all_properties = process_LC(ra_string, dec_string, lc_data, LCsurvey)
fig = plot_LC_analysis_ALLaliases(flc_data, P, freq_grid, power, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, df=df, title=title)
fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
while True:
    print(ii, n)
    event, values = window.read()
    print(event, values)
    if (event == 'Exit') or (event == sg.WIN_CLOSED):
        period_alias_frac[ii] = harmonics[np.where(np.array([values["-1/3P-"], values["-1/2P-"], values["-1P-"], values["-2P-"], values["-3P-"]]))[0][0]]
        variable_type[ii] = vartypes[np.where(np.array([values["-RRab-"], values["-RRc-"], values["-EA-"], values["-EB/EW-"], values["-SingleMin-"], values["-DeltaScuti-"], values["-Unknown-"], values["-Non-periodic-"]]))[0][0]]
        prop_col_names = ['P', 'logProb', 'Amp', 'isAlias', 'time_whittened']
        recalc_stats_keys = [f"{LCsurvey}_{col}" for col in prop_col_names]
        new_Lc_prop[ii][recalc_stats_keys] = all_properties['P'], all_properties['logProb'], all_properties['Amp'], all_properties['isAlias'], all_properties['time_whittened']
        new_Lc_prop[ii][f"{LCsurvey}_Pharm"] = period_alias_frac[ii]
        new_Lc_prop[ii][f"{LCsurvey}_VarType"] = variable_type[ii]
        new_Lc_prop.write(f"temp_prop_final_{LCsurvey}.fits", format='fits', overwrite=True)
        window.close()
        break

    if fig_canvas_agg:
        # ** IMPORTANT ** Clean up previous drawing before drawing again
        delete_figure_agg(fig_canvas_agg)

    if event == 'Previous':
        period_alias_frac[ii] = harmonics[np.where(np.array([values["-1/3P-"], values["-1/2P-"], values["-1P-"], values["-2P-"], values["-3P-"]]))[0][0]]
        variable_type[ii] = vartypes[np.where(np.array([values["-RRab-"], values["-RRc-"], values["-EA-"], values["-EB/EW-"], values["-SingleMin-"], values["-DeltaScuti-"], values["-Unknown-"], values["-Non-periodic-"]]))[0][0]]
        prop_col_names = ['P', 'logProb', 'Amp', 'isAlias', 'time_whittened']
        recalc_stats_keys = [f"{LCsurvey}_{col}" for col in prop_col_names]
        new_Lc_prop[ii][recalc_stats_keys] = all_properties['P'], all_properties['logProb'], all_properties['Amp'], all_properties['isAlias'], all_properties['time_whittened']
        new_Lc_prop[ii][f"{LCsurvey}_Pharm"] = period_alias_frac[ii]
        new_Lc_prop[ii][f"{LCsurvey}_VarType"] = variable_type[ii]

        if ii != 0:
            ii -= 1
            n = all_prop_index[this_lc_index[ii]]
        fig, all_properties = run_plot(n, prop, LCsurvey)
        fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
    elif event == 'Next':
        period_alias_frac[ii] = harmonics[np.where(np.array([values["-1/3P-"], values["-1/2P-"], values["-1P-"], values["-2P-"], values["-3P-"]]))[0][0]]
        variable_type[ii] = vartypes[np.where(np.array([values["-RRab-"], values["-RRc-"], values["-EA-"], values["-EB/EW-"], values["-SingleMin-"], values["-DeltaScuti-"], values["-Unknown-"], values["-Non-periodic-"]]))[0][0]]
        prop_col_names = ['P', 'logProb', 'Amp', 'isAlias', 'time_whittened']
        recalc_stats_keys = [f"{LCsurvey}_{col}" for col in prop_col_names]
        new_Lc_prop[ii][recalc_stats_keys] = all_properties['P'], all_properties['logProb'], all_properties['Amp'], all_properties['isAlias'], all_properties['time_whittened']
        new_Lc_prop[ii][f"{LCsurvey}_Pharm"] = period_alias_frac[ii]
        new_Lc_prop[ii][f"{LCsurvey}_VarType"] = variable_type[ii]

        if ii != total_N:
            ii += 1
            n = all_prop_index[this_lc_index[ii]]
        elif ii == total_N:  # end of inspection
            window.close()
            break
        fig, all_properties = run_plot(n, prop, LCsurvey)
        fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

    print(ii, n)
    if (ii % 100) == 0:
        new_Lc_prop.write(f"temp_prop_{LCsurvey}.fits", format='fits', overwrite=True)
