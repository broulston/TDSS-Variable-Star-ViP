from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.style as mplstyle
import matplotlib
mplstyle.use('fast')
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0
matplotlib.rcParams['agg.path.chunksize'] = 10000

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable

import PySimpleGUI as sg
import numpy as np

import os

from astropy.table import Table
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from astropy import coordinates as coords
from astropy.coordinates import SkyCoord, ICRS

import ResearchTools.LCtools as LCtools
import blosc
import pickle


############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
output_dir = "Analysis_Results/"
datestr = '2021-09-08'

lc_dir0 = output_dir+datestr+"/ZTF/"
lc_dir_CSS = output_dir+datestr+"/CSS/"
lc_dir_ZTFg = output_dir+datestr+"/ZTF/g"
lc_dir_ZTFr = output_dir+datestr+"/ZTF/r"
lc_dir_ZTFi = output_dir+datestr+"/ZTF/i"

raw_lc_analysis_dir_ZTF = output_dir+datestr+"/RAW_LC_ANALYSIS/"+"ZTF/"
raw_LC_analysis_dir_CSS = output_dir+datestr+"/RAW_LC_ANALYSIS/"+"CSS/"
raw_LC_analysis_dir_ZTFg = output_dir+datestr+"/RAW_LC_ANALYSIS/"+"ZTF/g/"
raw_LC_analysis_dir_ZTFr = output_dir+datestr+"/RAW_LC_ANALYSIS/"+"ZTF/r/"
raw_LC_analysis_dir_ZTFi = output_dir+datestr+"/RAW_LC_ANALYSIS/"+"ZTF/i/"   


checkHarmonic = False
log10FAP = -5.0
logFAP_limit = log10FAP
polyfit_deg = 3

CSS_LC_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/CSS_LCs/csvs/"
ZTF_LC_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/ZTF/DATA/07-27-2021/"

ZTF_LC_data = Table.read("/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/ZTF/DATA/07-27-2021/TDSS_VarStar_ZTFDR6_gri_GroupID.fits")
TDSS_prop = Table.read("/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/PROGRAM_SAMPLE/2021-07-27/FINAL_FILES/TDSS_SES+PREV_DR16DR12griLT20_GaiaEDR3_Drake2014PerVar_CSSID_ZTFIDs_LCpointer_PyHammer_EqW.fits")

sample_size = len(TDSS_prop)

############################################################################
############################################################################
############################################################################
############################################################################
############################################################################


def plt_raw_lc(lc_data, ax, title="", show_err_lines=True, plot_rejected=True):
    goodQualIndex = np.where(lc_data['QualFlag']==True)[0]
    badQualIndex = np.where(lc_data['QualFlag']==False)[0]
    mjd = lc_data['mjd'][goodQualIndex].data
    mag = lc_data['mag'][goodQualIndex].data
    err = lc_data['magerr'][goodQualIndex].data

    mjd_bad = lc_data['mjd'][badQualIndex].data
    mag_bad = lc_data['mag'][badQualIndex].data
    err_bad = lc_data['magerr'][badQualIndex].data

    if title != "":
        ax.set_title(title)

    fmagmn = np.mean(mag)
    ferrmn = np.mean(err)
    fmag_stdev = np.std(mag)

    if show_err_lines:
        ax.axhline(fmagmn, color='r', ls='-', lw=1.5, alpha=0.5)
        
        ax.axhline(fmagmn + 3 * ferrmn, color='g', ls='-.', lw=1.5, alpha=0.5, label='3X Mag Err')
        ax.axhline(fmagmn - 3 * ferrmn, color='g', ls='-.', lw=1.5, alpha=0.5)
        ax.axhline(fmagmn + 3 * fmag_stdev, color='b', ls=':', lw=1.5, alpha=0.5, label='3X Mag StDev')
        ax.axhline(fmagmn - 3 * fmag_stdev, color='b', ls=':', lw=1.5, alpha=0.5)


        ax.errorbar(mjd, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=1.0)
        if plot_rejected:
            ax.errorbar(mjd_bad, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=1.0)

        ax.set_xlabel('MJD')
        ax.grid()
        ax.set_ylabel('mag')
        ax.invert_yaxis()
    return ax

def plot_single_windowfun(flc_data, ax, title='Window Function', P=None):
    goodQualIndex = np.where(flc_data['QualFlag'] == 1)[0]
    lc_mjd = flc_data['mjd'][goodQualIndex]
    lc_mag = flc_data['mag'][goodQualIndex]
    lc_err = flc_data['magerr'][goodQualIndex]

    t_days = lc_mjd  # * u.day
    y_mags = lc_mag  # * u.mag
    dy_mags = lc_err  # * u.mag

    if t_days.unit == None:
        t_days = t_days * u.d
    if y_mags.unit == None:
        y_mags = y_mags * u.mag
    if dy_mags.unit == None:
        dy_mags = dy_mags * u.mag

    y_mags = y_mags / y_mags.value

    minP =  0.1 * u.d
    maximum_frequency = (minP)**-1
    frequency = np.linspace(0, maximum_frequency.value, num=250001)[1:] / u.d

    ls = LombScargle(t_days, y_mags, dy_mags, fit_mean=False, center_data=False)
    power = ls.power(frequency=frequency)

    title = ""
    plot_single_powerspec(frequency, power, ax1=ax, title=title, window=True, P=P)
    #ymin, ymax = ax1.get_ylim()
    #ymin - (0.11*ymax)
    # ax1.annotate('',
    #          xy=(((use_P*u.d)**-1).to(1/u.d).value, ymin),
    #          xytext=(((use_P*u.d)**-1).to(1/u.d).value, ymin - (0.11*ymax)),
    #          xycoords='data', annotation_clip=False, arrowprops=dict(arrowstyle="->", color='r'))
    #ax.tick_params(axis='both', which='major', labelsize=15, width=2.0, length=10)
    #ax.tick_params(axis='both', which='minor', width=1.0, length=5)

    #ax.xaxis.label.set_size(15)
    #ax.yaxis.label.set_size(15)
    #ax.set_xlabel('')
    return ax


def plot_single_powerspec(frequency, power, P, ax1=None, FAP_power_peak=None, logFAP_limit=None, alias_df=None, title="", window=False):
    minimum_frequency = frequency.min()
    maximum_frequency = frequency.max()

    minP = maximum_frequency**-1
    maxP = minimum_frequency**-1

    if (ax1 is None):
        fig = plt.figure(figsize=(12, 4), constrained_layout=True, dpi=600)
        gs = GridSpec(1, 1, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])

    ax1.set_title(title)
    ax1.plot(frequency.to(1/u.d), power, c='k', lw=0.75)
    if FAP_power_peak is not None:
        ax1.axhline(y=FAP_power_peak, c='r', ls='dashed', alpha=0.5, lw=0.75)
        if logFAP_limit is not None:
            ax1.text(0.8 * maximum_frequency.to(1/u.d).value, FAP_power_peak + 0.0, f"log(FAP) = {logFAP_limit}", c='r')

    xmin = minimum_frequency.to(1/u.d).value
    xmax = maximum_frequency.to(1/u.d).value

    #ax1.axvline(x=((365.25 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    #ax1.axvline(x=((29.530587981 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    #ax1.axvline(x=((27.321661 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    #ax1.axvline(x=((9 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    #ax1.axvline(x=((8 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    #ax1.axvline(x=((7 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    #ax1.axvline(x=((6 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    #ax1.axvline(x=((5 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    #ax1.axvline(x=((4 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    #ax1.axvline(x=((3 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    #ax1.axvline(x=((2 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    
    ax1.axvline(x=((1 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 2 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 3 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 4 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 5 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 6 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 7 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 8 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 9 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.set_xlabel(r'Frequency [d$^{-1}$]')
    ax1.set_ylabel('Power')
    ax1.set_xlim((xmin, xmax))

    xmin = minimum_frequency.to(1/u.d).value
    xmax = maximum_frequency.to(1/u.d).value
    dx = xmax - xmin
    np.ceil(dx)

    xmajortick = 1 #np.floor(np.ceil(dx) / 23)
    xminortick = xmajortick / 10

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(xmajortick))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(xminortick))

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    
    #ax1.tick_params(axis='both', which='major', labelsize=15, width=2.0, length=10)
    #ax1.tick_params(axis='both', which='minor', width=1.0, length=5)

    #ax1.xaxis.label.set_size(15)
    #ax1.yaxis.label.set_size(15)
    
    #ax1.set_xscale('log')
    if window==False:
        ymin, ymax = ax1.get_ylim()
        ax1.annotate('',
             xy=(((P*u.d)**-1).to(1/u.d).value, ymin),
             xytext=(((P*u.d)**-1).to(1/u.d).value, ymin - (0.11*ymax)),
             xycoords='data', annotation_clip=False, arrowprops=dict(arrowstyle="->", color='r'))

    if alias_df is not None:
        n = [-4, -3, -2, -1, 1, 2, 3, 4]
        f0 = frequency[np.argmax(power)]
        ax1.axvline(x=f0.to(1/u.d).value, c='r', ls='dashed', alpha=0.75, lw=0.75)
        for ii in n:
            ax1.axvline(x=np.abs((f0 + (ii * alias_df)).to(1/u.d).value), c='b', ls='dashed', alpha=0.5, lw=0.75)
            if ii > 1:
                ax1.axvline(x=(f0 / ii).to(1/u.d).value, c='g', ls='dashed', alpha=0.5, lw=0.75)
                ax1.axvline(x=(f0 * ii).to(1/u.d).value, c='g', ls='dashed', alpha=0.5, lw=0.75)

    return ax1

def plt_any_lc(lc_data, P, is_Periodic=False, figsize=(8, 3), title="", phasebin=False, bins=25, phasebinonly=False, show_err_lines=True, plot_rejected=False):
    fig = plt.figure(figsize=figsize, constrained_layout=False, dpi=600)
    gs = GridSpec(4, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0:3, :])
    ax2 = fig.add_subplot(gs[3, :], sharex=ax1)

    gs.update(hspace=0.0) # set the spacing between axes. 
    plt.setp(ax1.get_xticklabels(), visible=False)



    goodQualIndex = np.where(lc_data['QualFlag']==True)[0]
    badQualIndex = np.where(lc_data['QualFlag']==False)[0]
    mjd = lc_data['mjd'][goodQualIndex].data
    mag = lc_data['mag'][goodQualIndex].data
    err = lc_data['magerr'][goodQualIndex].data

    data = [mjd, mag, err]
    AFD_data = LCtools.AFD(data, P)

    Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiS, mfit = AFD_data

    Amp = y_fit.max() - y_fit.min()
    t0 = (mjd - (phased_t * P)).min()

    title = title  # + "Amp = {!s} $|$ t0 = {!s}".format(np.round(Amp, 3), np.round(t0, 7))

    mjd_bad = lc_data['mjd'][badQualIndex].data
    phase_bad = ((mjd_bad - t0) / P) % 1
    mag_bad = lc_data['mag'][badQualIndex].data
    err_bad = lc_data['magerr'][badQualIndex].data

    binned_phase, binned_mag, binned_err = LCtools.bin_phaseLC(phased_t, mag, err, bins=bins)

    if title is not "":
        ax1.set_title(title)
    # is_Periodic = True
    if is_Periodic:
        if phasebinonly:
            pass
        else:
            ax1.errorbar(phased_t, mag, err, fmt='.k', ecolor='k', lw=1, ms=4, capsize=0, alpha=0.750, elinewidth=0.25)
            ax1.errorbar(phased_t + 1, mag, err, fmt='.k', ecolor='k', lw=1, ms=4, capsize=0, alpha=0.75, elinewidth=0.25)

            if plot_rejected:
                ax1.errorbar(phase_bad, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=0, alpha=0.5)
                ax1.errorbar(phase_bad + 1, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=0, alpha=0.5)

        ax1.plot(phase_fit, y_fit, 'r', markeredgecolor='r', lw=0.5, fillstyle='top', linestyle='solid', zorder=10)
        ax1.plot(phase_fit + 1, y_fit, 'r', markeredgecolor='r', lw=0.5, fillstyle='top', linestyle='solid', zorder=10)
        # plt_ax.plot(phase_fit+2, y_fit, 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')

        fmagmn = np.mean(mag)
        ferrmn = np.mean(err)
        fmag_stdev = np.std(mag)

        #ax1.axhline(fmagmn, color='r', ls='-', lw=0.5, label='Mean Mag')
        if show_err_lines:
            ax1.axhline(fmagmn + 3 * ferrmn, color='g', ls='-.', lw=0.5, alpha=0.5, label='3X Mag Err')
            ax1.axhline(fmagmn - 3 * ferrmn, color='g', ls='-.', lw=0.5, alpha=0.5)
            ax1.axhline(fmagmn + 3 * fmag_stdev, color='b', ls=':', lw=0.5, alpha=0.5, label='3X Mag StDev')
            ax1.axhline(fmagmn - 3 * fmag_stdev, color='b', ls=':', lw=0.5, alpha=0.5)
        else:
            pass

        ax1.set_ylabel('mag')
        # plt_ax.set_xlabel('Phase')
        ax1.invert_yaxis()

        ax1.grid()
        if phasebin:
            ax1.errorbar(binned_phase, binned_mag, binned_err, fmt='sr', ecolor='red', lw=1, ms=4, capsize=0, alpha=0.3, zorder=10)
            ax1.errorbar(binned_phase + 1, binned_mag, binned_err, fmt='sr', ecolor='red', lw=1, ms=4, capsize=0, alpha=0.3, zorder=10)

        ax2.set_xlim(0.0, 2.0)

        ax2.errorbar(phased_t, resid, fmt='.k', ecolor='k', lw=1, ms=4, capsize=1.5, alpha=1.0)
        ax2.errorbar(phased_t + 1, resid, fmt='.k', ecolor='k', lw=1, ms=4, capsize=1.5, alpha=1.0)

        ax2.grid()
        ax2.set_xlim(0.0, 2.0)

        ax2.set_xlabel(r'Phase')
        ax2.set_ylabel('Residual')  # \n N$_{terms} = 4$')
    else:
        ax1.errorbar(mjd, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=1.0)
        if plot_rejected:
            ax1.errorbar(mjd_bad, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=1.0)

        fmagmn = np.mean(mag)
        ferrmn = np.mean(err)
        fmag_stdev = np.std(mag)

        ax1.axhline(fmagmn, color='r', ls='-', lw=2, label='Mean Mag')
        ax1.axhline(fmagmn + 3 * ferrmn, color='g', ls='-.', lw=2, alpha=0.5, label='3X Mag Err')
        ax1.axhline(fmagmn - 3 * ferrmn, color='g', ls='-.', lw=2, alpha=0.5)
        ax1.axhline(fmagmn + 3 * fmag_stdev, color='b', ls=':', lw=2, alpha=0.5, label='3X Mag StDev')
        ax1.axhline(fmagmn - 3 * fmag_stdev, color='b', ls=':', lw=2, alpha=0.5)

        ax1.set_xlabel('MJD')
        ax1.grid()
        ax1.set_ylabel('mag')
        ax1.invert_yaxis()
    return ax1, ax2



def plt_any_lc_fig(lc_data, P, ax1, ax2, is_Periodic=False, title="", phasebin=False, bins=25, phasebinonly=False, show_err_lines=True, plot_rejected=False):
    #fig = plt.figure(figsize=figsize, constrained_layout=False, dpi=600)
    #gs = GridSpec(4, 1, figure=fig)
    #ax1 = fig.add_subplot(gs[0:3, :])
    #ax2 = fig.add_subplot(gs[3, :], sharex=ax1)

    #gs.update(hspace=0.0) # set the spacing between axes. 
    #plt.setp(ax1.get_xticklabels(), visible=False)



    goodQualIndex = np.where(lc_data['QualFlag']==True)[0]
    badQualIndex = np.where(lc_data['QualFlag']==False)[0]
    mjd = lc_data['mjd'][goodQualIndex].data
    mag = lc_data['mag'][goodQualIndex].data
    err = lc_data['magerr'][goodQualIndex].data

    data = [mjd, mag, err]
    AFD_data = LCtools.AFD(data, P)

    Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiS, mfit = AFD_data

    Amp = y_fit.max() - y_fit.min()
    t0 = (mjd - (phased_t * P)).min()

    title = title  # + "Amp = {!s} $|$ t0 = {!s}".format(np.round(Amp, 3), np.round(t0, 7))

    mjd_bad = lc_data['mjd'][badQualIndex].data
    phase_bad = ((mjd_bad - t0) / P) % 1
    mag_bad = lc_data['mag'][badQualIndex].data
    err_bad = lc_data['magerr'][badQualIndex].data

    binned_phase, binned_mag, binned_err = LCtools.bin_phaseLC(phased_t, mag, err, bins=bins)

    if title is not "":
        ax1.set_title(title)
    # is_Periodic = True
    if is_Periodic:
        if phasebinonly:
            pass
        else:
            ax1.errorbar(phased_t, mag, err, fmt='.k', ecolor='k', lw=1, ms=4, capsize=0, alpha=0.750, elinewidth=0.25)
            ax1.errorbar(phased_t + 1, mag, err, fmt='.k', ecolor='k', lw=1, ms=4, capsize=0, alpha=0.75, elinewidth=0.25)

            if plot_rejected:
                ax1.errorbar(phase_bad, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=0, alpha=0.5)
                ax1.errorbar(phase_bad + 1, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=0, alpha=0.5)

        ax1.plot(phase_fit, y_fit, 'r', markeredgecolor='r', lw=0.5, fillstyle='top', linestyle='solid', zorder=10)
        ax1.plot(phase_fit + 1, y_fit, 'r', markeredgecolor='r', lw=0.5, fillstyle='top', linestyle='solid', zorder=10)
        # plt_ax.plot(phase_fit+2, y_fit, 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')

        fmagmn = np.mean(mag)
        ferrmn = np.mean(err)
        fmag_stdev = np.std(mag)

        #ax1.axhline(fmagmn, color='r', ls='-', lw=0.5, label='Mean Mag')
        if show_err_lines:
            ax1.axhline(fmagmn + 3 * ferrmn, color='g', ls='-.', lw=0.5, alpha=0.5, label='3X Mag Err')
            ax1.axhline(fmagmn - 3 * ferrmn, color='g', ls='-.', lw=0.5, alpha=0.5)
            ax1.axhline(fmagmn + 3 * fmag_stdev, color='b', ls=':', lw=0.5, alpha=0.5, label='3X Mag StDev')
            ax1.axhline(fmagmn - 3 * fmag_stdev, color='b', ls=':', lw=0.5, alpha=0.5)
        else:
            pass

        ax1.set_ylabel('mag')
        # plt_ax.set_xlabel('Phase')
        ax1.invert_yaxis()

        ax1.grid()
        if phasebin:
            ax1.errorbar(binned_phase, binned_mag, binned_err, fmt='sr', ecolor='red', lw=1, ms=4, capsize=0, alpha=0.3, zorder=10)
            ax1.errorbar(binned_phase + 1, binned_mag, binned_err, fmt='sr', ecolor='red', lw=1, ms=4, capsize=0, alpha=0.3, zorder=10)

        ax2.set_xlim(0.0, 2.0)

        ax2.errorbar(phased_t, resid, fmt='.k', ecolor='k', lw=1, ms=4, capsize=1.5, alpha=1.0)
        ax2.errorbar(phased_t + 1, resid, fmt='.k', ecolor='k', lw=1, ms=4, capsize=1.5, alpha=1.0)

        ax2.grid()
        ax2.set_xlim(0.0, 2.0)

        ax2.set_xlabel(r'Phase')
        ax2.set_ylabel('Residual')  # \n N$_{terms} = 4$')
    else:
        ax1.errorbar(mjd, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=1.0)
        if plot_rejected:
            ax1.errorbar(mjd_bad, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=1.0)

        fmagmn = np.mean(mag)
        ferrmn = np.mean(err)
        fmag_stdev = np.std(mag)

        ax1.axhline(fmagmn, color='r', ls='-', lw=2, label='Mean Mag')
        ax1.axhline(fmagmn + 3 * ferrmn, color='g', ls='-.', lw=2, alpha=0.5, label='3X Mag Err')
        ax1.axhline(fmagmn - 3 * ferrmn, color='g', ls='-.', lw=2, alpha=0.5)
        ax1.axhline(fmagmn + 3 * fmag_stdev, color='b', ls=':', lw=2, alpha=0.5, label='3X Mag StDev')
        ax1.axhline(fmagmn - 3 * fmag_stdev, color='b', ls=':', lw=2, alpha=0.5)

        ax1.set_xlabel('MJD')
        ax1.grid()
        ax1.set_ylabel('mag')
        ax1.invert_yaxis()
    return ax1, ax2
    
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
_VARS = {'window': False,
         'fig_agg': False,
         'pltFig': False}


def low_order_poly(mag, a, b, c, d, e, f_, g):
    return a + b * mag + c * mag**2 + d * mag**3 + e * mag**4 + f_ * mag**5 + g * mag**5


def get_star(prop_id):
    star_all_props = [[], [], [], []]
    ROW = TDSS_prop[prop_id]
    is_CSS = ROW['CSSLC']
    is_ZTF = ROW['ZTFLC']
    
    object_ra = ROW['ra_GaiaEDR3']
    object_dec = ROW['dec_GaiaEDR3']
    ra_string = '{:0>9.5f}'.format(object_ra)
    dec_string = '{:0=+9.5f}'.format(object_dec)
    
    c = ICRS(object_ra*u.degree, object_dec*u.degree)
    rahmsstr = c.ra.to_string(u.hour, precision=2, pad=True)
    decdmsstr = c.dec.to_string(u.degree, alwayssign=True, precision=2, pad=True)

    #this_filename_base = f"{ra_string}{dec_string}_"
    if is_CSS:
        pickle_filename = raw_LC_analysis_dir_CSS + f"{prop_id}_CSS_{ROW['CSSID']}.dat"
        if os.path.isfile(pickle_filename):
            lc_file = CSS_LC_dir+str(ROW['CSSID'])+'.dat'
            CSS_lc_data = Table.read(lc_file, format='ascii', names=['mjd', 'mag', 'magerr'])
            popt = np.array([-2.61242938e+01,  1.93636204e+00,  4.45971381e-01, -6.49419310e-02, 2.99231126e-03,  2.40758201e-01, -2.40805035e-01])
            magerr_resid_mean = 0.008825118765717422
            shift_const = 1.5 * magerr_resid_mean
            pred_magerr = low_order_poly(CSS_lc_data['mag'], popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
            bad_err_index = np.where(CSS_lc_data['magerr'] - (pred_magerr - shift_const) < 0.0)[0]
            CSS_lc_data['magerr'][bad_err_index] = pred_magerr[bad_err_index]
            
            mean_mag = np.nanmean(CSS_lc_data['mag'])
            flc_data, LC_stat_properties = LCtools.process_LC(CSS_lc_data.copy(), fltRange=5.0, detrend=False)

            with open(pickle_filename, 'rb') as f:
                compressed_pickle = f.read()

            depressed_pickle = blosc.decompress(compressed_pickle)
            props = pickle.loads(depressed_pickle) 
            star_all_props[0] = [CSS_lc_data, flc_data, props]

    if is_ZTF:
        for ii, this_ZTF_filter in enumerate(['g', 'r', 'i']):
            pickle_filename = raw_lc_analysis_dir_ZTF + f"{this_ZTF_filter}/" + f"{prop_id}_ZTF{this_ZTF_filter}_{ROW['ZTF_GroupID']}.xz"
            if os.path.isfile(pickle_filename):
                lc_index = (ZTF_LC_data['ZTF_GroupID'] == ROW['ZTF_GroupID']) & (ZTF_LC_data['filtercode'] == 'z'+this_ZTF_filter)
                lc_data = ZTF_LC_data[lc_index]
                    
                mean_mag = np.nanmean(lc_data['mag'])
                flc_data, LC_stat_properties = LCtools.process_LC(lc_data.copy(), fltRange=5.0, detrend=False)

                with open(pickle_filename, 'rb') as f:
                    compressed_pickle = f.read()

                depressed_pickle = blosc.decompress(compressed_pickle)
                props = pickle.loads(depressed_pickle) 
                star_all_props[ii+1] = [lc_data, flc_data, props]

    return star_all_props




def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def drawFirstFIG(star_props):
    _VARS['pltFig'] = plt.figure(figsize=(8, 4), dpi=100) # constrained_layout=True,
    gs = GridSpec(3, 3, figure=_VARS['pltFig'], height_ratios=[1, 1, 1], width_ratios=[1, 1, 1], hspace=0.0)

    inner_grid1 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], hspace=0.0, width_ratios=[1], height_ratios=[0.75, 0.25])
    inner_grid2 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.0, width_ratios=[1], height_ratios=[0.75, 0.25])
    inner_grid3 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, 1], hspace=0.0, width_ratios=[1], height_ratios=[0.75, 0.25])

    is_CSS = False
    is_ZTFg = False
    is_ZTFr = False
    is_ZTFi = False
    if len(star_props[0]) != 0:
        is_CSS =True
    if len(star_props[1]) != 0:
        is_ZTFg =True
    if len(star_props[2]) != 0:
        is_ZTFr =True
    if len(star_props[3]) != 0:
        is_ZTFi =True

    ax1 = _VARS['pltFig'].add_subplot(gs[0, 0])
    ax3 = _VARS['pltFig'].add_subplot(gs[0, 2])
    ax4 = _VARS['pltFig'].add_subplot(gs[1, 0], sharex=ax3)
    
    ax6 = _VARS['pltFig'].add_subplot(gs[1, 2], sharex=ax3)
    ax7 = _VARS['pltFig'].add_subplot(gs[2, 0], sharex=ax1)
    ax9 = _VARS['pltFig'].add_subplot(gs[2, 2], sharex=ax3)

    ax2 = _VARS['pltFig'].add_subplot(inner_grid1[0])
    ax25 = _VARS['pltFig'].add_subplot(inner_grid1[1], sharex=ax2)

    ax5 = _VARS['pltFig'].add_subplot(inner_grid2[0])
    ax55 = _VARS['pltFig'].add_subplot(inner_grid2[1], sharex=ax2)

    ax8 = _VARS['pltFig'].add_subplot(inner_grid3[0])
    ax85 = _VARS['pltFig'].add_subplot(inner_grid3[1], sharex=ax2)


    #plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.setp(ax5.get_xticklabels(), visible=False)
    plt.setp(ax6.get_xticklabels(), visible=False)

    CSS_mjdmin, CSS_mjdmax = np.nan, np.nan
    if is_CSS:
        plt_raw_lc(star_props[0][1], ax1, title="", show_err_lines=True, plot_rejected=True)
        plt_any_lc_fig(star_props[0][1], star_props[0][2]['P'], ax2, ax25, is_Periodic=True, title="", phasebin=False, bins=25, phasebinonly=False, show_err_lines=True, plot_rejected=False)
        plot_single_powerspec(star_props[0][2]['frequency'], star_props[0][2]['power'], P=star_props[0][2]['P'], ax1=ax3, FAP_power_peak=star_props[0][2]['FAP_power_peak'], logFAP_limit=log10FAP, title="")

        CSS_mjdmin = np.nanmin(star_props[0][1]['mjd'])
        CSS_mjdmax = np.nanmax(star_props[0][1]['mjd'])

    ZTFg_mjdmin, ZTFg_mjdmax = np.nan, np.nan
    if is_ZTFg:
        plt_raw_lc(star_props[1][1], ax4, title="", show_err_lines=True, plot_rejected=True)

        ZTFg_mjdmin = np.nanmin(star_props[1][1]['mjd'])
        ZTFg_mjdmax = np.nanmax(star_props[1][1]['mjd'])

    ZTFr_mjdmin, ZTFr_mjdmax = np.nan, np.nan
    if is_ZTFr:
        plt_raw_lc(star_props[2][1], ax7, title="", show_err_lines=True, plot_rejected=True)

        ZTFr_mjdmin = np.nanmin(star_props[2][1]['mjd'])
        ZTFr_mjdmax = np.nanmax(star_props[2][1]['mjd'])

    mjd_min = np.nanmin([CSS_mjdmin, ZTFg_mjdmin, ZTFr_mjdmin]) - 500
    mjd_max = np.nanmax([CSS_mjdmax, ZTFg_mjdmax, ZTFr_mjdmax]) + 500

    ax1.set_xlim(mjd_min, mjd_max)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(50))

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax4.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax4.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax7.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax7.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    ax1.set_ylabel('CSS mag')
    ax4.set_ylabel('ZTF g mag')
    ax7.set_ylabel('ZTF r mag')

    plt.tight_layout()
    _VARS['fig_agg'] = draw_figure(_VARS['window']['FIGURE'].TKCanvas, _VARS['pltFig'])
    return [ax1 ,ax3 ,ax4 ,ax6 ,ax7 ,ax9 ,ax2 ,ax25,ax5 ,ax55,ax8 ,ax85]



def drawFIG(star_props, axes):
    ax1 ,ax3 ,ax4 ,ax6 ,ax7 ,ax9 ,ax2 ,ax25,ax5 ,ax55,ax8 ,ax85 = axes
    #_VARS['pltFig'] = plt.figure(figsize=(8, 4), dpi=100) # constrained_layout=True,
    #gs = GridSpec(3, 3, figure=_VARS['pltFig'], height_ratios=[1, 1, 1], width_ratios=[1, 1, 1], hspace=0.0)

    #inner_grid1 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], hspace=0.0, width_ratios=[1], height_ratios=[0.75, 0.25])
    #inner_grid2 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.0, width_ratios=[1], height_ratios=[0.75, 0.25])
    #inner_grid3 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, 1], hspace=0.0, width_ratios=[1], height_ratios=[0.75, 0.25])

    is_CSS = False
    is_ZTFg = False
    is_ZTFr = False
    is_ZTFi = False
    if len(star_props[0]) != 0:
        is_CSS =True
    if len(star_props[1]) != 0:
        is_ZTFg =True
    if len(star_props[2]) != 0:
        is_ZTFr =True
    if len(star_props[3]) != 0:
        is_ZTFi =True

    #ax1 = _VARS['pltFig'].add_subplot(gs[0, 0])
    #ax3 = _VARS['pltFig'].add_subplot(gs[0, 2])
    #ax4 = _VARS['pltFig'].add_subplot(gs[1, 0], sharex=ax3)
    #
    #ax6 = _VARS['pltFig'].add_subplot(gs[1, 2], sharex=ax3)
    #ax7 = _VARS['pltFig'].add_subplot(gs[2, 0], sharex=ax1)
    #ax9 = _VARS['pltFig'].add_subplot(gs[2, 2], sharex=ax3)

    #ax2 = _VARS['pltFig'].add_subplot(inner_grid1[0])
    #ax25 = _VARS['pltFig'].add_subplot(inner_grid1[1], sharex=ax2)

    #ax5 = _VARS['pltFig'].add_subplot(inner_grid2[0])
    #ax55 = _VARS['pltFig'].add_subplot(inner_grid2[1], sharex=ax2)

    #ax8 = _VARS['pltFig'].add_subplot(inner_grid3[0])
    #ax85 = _VARS['pltFig'].add_subplot(inner_grid3[1], sharex=ax2)


    #plt.setp(ax1.get_xticklabels(), visible=False)
    #plt.setp(ax2.get_xticklabels(), visible=False)
    #plt.setp(ax3.get_xticklabels(), visible=False)
    #plt.setp(ax4.get_xticklabels(), visible=False)
    #plt.setp(ax5.get_xticklabels(), visible=False)
    #plt.setp(ax6.get_xticklabels(), visible=False)
    
    CSS_mjdmin, CSS_mjdmax = np.nan, np.nan
    if is_CSS:
        plt_raw_lc(star_props[0][1], ax1, title="", show_err_lines=True, plot_rejected=True)
        plt_any_lc_fig(star_props[0][1], star_props[0][2]['P'], ax2, ax25, is_Periodic=True, title="", phasebin=False, bins=25, phasebinonly=False, show_err_lines=True, plot_rejected=False)
        plot_single_powerspec(star_props[0][2]['frequency'], star_props[0][2]['power'], P=star_props[0][2]['P'], ax1=ax3, FAP_power_peak=star_props[0][2]['FAP_power_peak'], logFAP_limit=log10FAP, title="")

        CSS_mjdmin = np.nanmin(star_props[0][1]['mjd'])
        CSS_mjdmax = np.nanmax(star_props[0][1]['mjd'])

    ZTFg_mjdmin, ZTFg_mjdmax = np.nan, np.nan
    if is_ZTFg:
        plt_raw_lc(star_props[1][1], ax4, title="", show_err_lines=True, plot_rejected=True)

        ZTFg_mjdmin = np.nanmin(star_props[1][1]['mjd'])
        ZTFg_mjdmax = np.nanmax(star_props[1][1]['mjd'])

    ZTFr_mjdmin, ZTFr_mjdmax = np.nan, np.nan
    if is_ZTFr:
        plt_raw_lc(star_props[2][1], ax7, title="", show_err_lines=True, plot_rejected=True)

        ZTFr_mjdmin = np.nanmin(star_props[2][1]['mjd'])
        ZTFr_mjdmax = np.nanmax(star_props[2][1]['mjd'])

    mjd_min = np.nanmin([CSS_mjdmin, ZTFg_mjdmin, ZTFr_mjdmin]) - 500
    mjd_max = np.nanmax([CSS_mjdmax, ZTFg_mjdmax, ZTFr_mjdmax]) + 500

    ax1.set_xlim(mjd_min, mjd_max)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(50))

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax4.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax4.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax7.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax7.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    ax1.set_ylabel('CSS mag')
    ax4.set_ylabel('ZTF g mag')
    ax7.set_ylabel('ZTF r mag')

    plt.tight_layout()
    _VARS['fig_agg'] = draw_figure(_VARS['window']['FIGURE'].TKCanvas, _VARS['pltFig'])
    return [ax1 ,ax3 ,ax4 ,ax6 ,ax7 ,ax9 ,ax2 ,ax25,ax5 ,ax55,ax8 ,ax85]



def updateFig(star_props, axes):
    _VARS['fig_agg'].get_tk_widget().forget()
    plt.clf()
    #plt.close('all')
    drawFIG(star_props, axes)
    _VARS['fig_agg'] = draw_figure(_VARS['window']['FIGURE'].TKCanvas, _VARS['pltFig'])


############################################################################
############################################################################
############################################################################
############################################################################
############################################################################


Pfrac_col = [[sg.Radio('P', 'PeriodFrac', default=True), sg.Text(), sg.Text()],
             [sg.Radio('(1/2)P', 'PeriodFrac', default=False), sg.Text(), sg.Radio('2P', 'PeriodFrac', default=False)],
             [sg.Radio('(1/3)P', 'PeriodFrac', default=False), sg.Text(), sg.Radio('3P', 'PeriodFrac', default=False)],
             [sg.Radio('(1/4)P', 'PeriodFrac', default=False), sg.Text(), sg.Radio('4P', 'PeriodFrac', default=False)],
             [sg.Radio('(1/5)P', 'PeriodFrac', default=False), sg.Text(), sg.Radio('5P', 'PeriodFrac', default=False)]]

varType_col = [[sg.Radio('Periodic', 'isPeriodic', default=False)],
               [sg.Radio('Non-Periodic', 'isPeriodic', default=True)],
               [sg.Checkbox('Long term trends', default=False)],
               [sg.Text()],
               [sg.Text()]]

periodicType_col = [[sg.Radio('None', 'PeriodType', default=True),   sg.Text(), sg.Radio('single-min', 'PeriodType', default=False)],
                    [sg.Radio('RRab', 'PeriodType', default=False),  sg.Text(), sg.Radio('other', 'PeriodType', default=False)],
                    [sg.Radio('RRc', 'PeriodType', default=False),   sg.Text(), sg.Text()],
                    [sg.Radio('EA', 'PeriodType', default=False),    sg.Text(), sg.Text()],
                    [sg.Radio('EB/EW', 'PeriodType', default=False), sg.Text(), sg.Text()]]

button_col = [[sg.Text(), sg.Text()],
              [sg.Text(), sg.Text()],
              [sg.Text(), sg.Text()],
              [sg.Button("Previous"), sg.Button("Next")],
              [sg.Button('Save'), sg.Button('Save/Exit')]]

comment_col = [[sg.Text("Comments")],
               [sg.Multiline(default_text='', size=(35, 5))]]

layout = [[sg.Canvas(key='FIGURE')],
          [sg.Column(Pfrac_col), sg.VerticalSeparator(), sg.Column(varType_col), sg.VerticalSeparator(), sg.Column(periodicType_col), sg.VerticalSeparator(), sg.Column(comment_col), sg.VerticalSeparator(), sg.Column(button_col)]]


_VARS['window'] = sg.Window('TDSS Variable Star Light Curve Inspection',
                            layout,
                            finalize=True,
                            resizable=True,
                            location=(100, 100),
                            element_justification="center",
                            font='Helvetica 18')

############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
prop_id = 0
prev_current_next_star_props = [None, get_star(0), get_star(1)]
star_props = prev_current_next_star_props[1] #get_star(0)
axes = drawFirstFIG(star_props)

while True:
    event, values = _VARS['window'].read()
    if event == sg.WIN_CLOSED or event == 'Save/Exit':
        plt.close('all')
        break
    elif event == 'Previous':
        prop_id = prop_id - 1
        if prop_id==-1:
            break

        star_props = prev_current_next_star_props[0].copy() #get_star(prop_id)
        axes = updateFig(star_props, axes)
        if prop_id == 0:
            prev_current_next_star_props = [None, star_props, prev_current_next_star_props[1].copy()]
        else:
            prev_current_next_star_props = [get_star(prop_id-1), star_props, prev_current_next_star_props[1].copy()]

    elif event == 'Next': 
        print("1", prop_id)
        prop_id = prop_id + 1
        print("2", prop_id)
        if prop_id==sample_size:
            break
        star_props = prev_current_next_star_props[2].copy() #get_star(prop_id)
        print("3", prop_id, len(star_props))
        axes = updateFig(star_props, axes)
        print("4", prop_id, len(star_props))
        if prop_id == sample_size-1:
            prev_current_next_star_props = [prev_current_next_star_props[1].copy(), star_props, None]
        else:
            prev_current_next_star_props = [prev_current_next_star_props[1].copy(), star_props, get_star(prop_id+1)]

_VARS['window'].close()