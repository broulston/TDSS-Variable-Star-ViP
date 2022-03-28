from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.style as mplstyle
import matplotlib
matplotlib.use('pdf')
mplstyle.use('fast')
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0
matplotlib.rcParams['agg.path.chunksize'] = 10000

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import PySimpleGUI as sg
import numpy as np

import os
import sys
import gc

from astropy.table import Table
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from astropy import coordinates as coords
from astropy.coordinates import SkyCoord, ICRS

import ResearchTools.LCtools as LCtools
import blosc
import pickle

import tqdm.autonotebook as tqdm
from multiprocessing import Pool


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

    del goodQualIndex
    del badQualIndex
    del mjd
    del mag
    del err
    del mjd_bad
    del mag_bad
    del err_bad
    del fmagmn
    del ferrmn
    del fmag_stdev
    
    return ax

def plt_any_lc_fig(lc_data, P, ax_lc, ax_resid, is_Periodic=False, title="", phasebin=False, bins=25, phasebinonly=False, show_err_lines=True, plot_rejected=False):
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

    try:
        data = [mjd, mag, err]
        AFD_data = LCtools.AFD(data, P)

        Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiS, mfit = AFD_data

        Amp = y_fit.max() - y_fit.min()
        t0 = (mjd - (phased_t * P)).max()
        noAFD = False
    except np.linalg.LinAlgError:
        AFD_data = None
        Nterms = None
        phase_fit = None
        y_fit = None
        resid = None
        reduced_ChiS = None
        mfit = None
        Amp = np.nan
        t0 = mjd[np.argmin(mag)]
        phased_t = ((mjd-t0) / P) % 1
        noAFD = True

    title = title  # + "Amp = {!s} $|$ t0 = {!s}".format(np.round(Amp, 3), np.round(t0, 7))

    mjd_bad = lc_data['mjd'][badQualIndex].data
    phase_bad = ((mjd_bad - t0) / P) % 1
    mag_bad = lc_data['mag'][badQualIndex].data
    err_bad = lc_data['magerr'][badQualIndex].data

    binned_phase, binned_mag, binned_err = LCtools.bin_phaseLC(phased_t, mag, err, bins=bins)

    if title != "":
        ax_lc.set_title(title)
    # is_Periodic = True
    if is_Periodic:
        if phasebinonly:
            pass
        else:
            ax_lc.errorbar(phased_t, mag, err, fmt='.k', ecolor='k', lw=1, ms=4, capsize=0, alpha=0.750, elinewidth=0.25)
            ax_lc.errorbar(phased_t + 1, mag, err, fmt='.k', ecolor='k', lw=1, ms=4, capsize=0, alpha=0.75, elinewidth=0.25)

            if plot_rejected:
                ax_lc.errorbar(phase_bad, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=0, alpha=0.5)
                ax_lc.errorbar(phase_bad + 1, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=0, alpha=0.5)

        if noAFD:
            pass
        else:
            ax_lc.plot(phase_fit, y_fit, 'r', markeredgecolor='r', lw=0.5, fillstyle='top', linestyle='solid', zorder=10)
            ax_lc.plot(phase_fit + 1, y_fit, 'r', markeredgecolor='r', lw=0.5, fillstyle='top', linestyle='solid', zorder=10)
            # plt_ax.plot(phase_fit+2, y_fit, 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')

        fmagmn = np.mean(mag)
        ferrmn = np.mean(err)
        fmag_stdev = np.std(mag)

        #ax_lc.axhline(fmagmn, color='r', ls='-', lw=0.5, label='Mean Mag')
        if show_err_lines:
            ax_lc.axhline(fmagmn + 3 * ferrmn, color='g', ls='-.', lw=0.5, alpha=0.5, label='3X Mag Err')
            ax_lc.axhline(fmagmn - 3 * ferrmn, color='g', ls='-.', lw=0.5, alpha=0.5)
            ax_lc.axhline(fmagmn + 3 * fmag_stdev, color='b', ls=':', lw=0.5, alpha=0.5, label='3X Mag StDev')
            ax_lc.axhline(fmagmn - 3 * fmag_stdev, color='b', ls=':', lw=0.5, alpha=0.5)
        else:
            pass

        ax_lc.set_ylabel('mag')
        # plt_ax.set_xlabel('Phase')
        ax_lc.invert_yaxis()

        ax_lc.grid()
        if noAFD:
            pass
        else:
            if phasebin:
                ax_lc.errorbar(binned_phase, binned_mag, binned_err, fmt='sr', ecolor='red', lw=1, ms=4, capsize=0, alpha=0.3, zorder=10)
                ax_lc.errorbar(binned_phase + 1, binned_mag, binned_err, fmt='sr', ecolor='red', lw=1, ms=4, capsize=0, alpha=0.3, zorder=10)

        ax_resid.set_xlim(0.0, 2.0)

        if noAFD:
            pass
        else:
            ax_resid.errorbar(phased_t, resid, fmt='.k', ecolor='k', lw=1, ms=4, capsize=1.5, alpha=1.0)
            ax_resid.errorbar(phased_t + 1, resid, fmt='.k', ecolor='k', lw=1, ms=4, capsize=1.5, alpha=1.0)

        ax_resid.grid()

        ax_resid.set_xlabel(r'Phase')
        ax_resid.set_ylabel('Residual')  # \n N$_{terms} = 4$')
    else:
        ax_resid.errorbar(mjd, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=1.0)
        if plot_rejected:
            ax_resid.errorbar(mjd_bad, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=1.0)

        fmagmn = np.mean(mag)
        ferrmn = np.mean(err)
        fmag_stdev = np.std(mag)

        ax_resid.axhline(fmagmn, color='r', ls='-', lw=2, label='Mean Mag')
        ax_resid.axhline(fmagmn + 3 * ferrmn, color='g', ls='-.', lw=2, alpha=0.5, label='3X Mag Err')
        ax_resid.axhline(fmagmn - 3 * ferrmn, color='g', ls='-.', lw=2, alpha=0.5)
        ax_resid.axhline(fmagmn + 3 * fmag_stdev, color='b', ls=':', lw=2, alpha=0.5, label='3X Mag StDev')
        ax_resid.axhline(fmagmn - 3 * fmag_stdev, color='b', ls=':', lw=2, alpha=0.5)

        ax_resid.set_xlabel('MJD')
        ax_resid.grid()
        ax_resid.set_ylabel('mag')
        ax_resid.invert_yaxis()


    del goodQualIndex
    del badQualIndex
    del mjd
    del mag
    del err
    del data
    del AFD_data
    del Nterms
    del phase_fit
    del y_fit
    del phased_t
    del resid
    del reduced_ChiS
    del mfit
    del Amp
    del t0
    del noAFD
    del title
    del mjd_bad
    del phase_bad
    del mag_bad
    del err_bad
    del binned_phase
    del binned_mag
    del binned_err
    del fmagmn
    del ferrmn
    del fmag_stdev

    return ax_lc, ax_resid
    
def low_order_poly(mag, a, b, c, d, e, f_, g):
    return a + b * mag + c * mag**2 + d * mag**3 + e * mag**4 + f_ * mag**5 + g * mag**5

def get_star(prop_id):
    star_all_props = [[], [], [], []]
    ROW = TDSS_prop[prop_id]
    is_CSS = ROW['CSSLC']
    is_ZTF = ROW['ZTFLC']
    
    object_ra = ROW['ra_GaiaEDR3']
    object_dec = ROW['dec_GaiaEDR3']
    #ra_string = '{:0>9.5f}'.format(object_ra)
    #dec_string = '{:0=+9.5f}'.format(object_dec)
    
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


def drawFIGS(star_props, extras):
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

    prop_id, ROW, rahmsstr, decdmsstr = extras

    if is_CSS:
        base_filename = lc_dir_CSS + f"{prop_id}_CSS_{ROW['CSSID']}"
        drawFIG(star_props[0], 'CSS', base_filename, extras)

    if is_ZTFg:
        base_filename = lc_dir_ZTFg + f"/{prop_id}_ZTFg_{ROW['ZTF_GroupID']}"
        drawFIG(star_props[1], 'ZTFg', base_filename, extras)

    if is_ZTFr:
        base_filename = lc_dir_ZTFr + f"/{prop_id}_ZTFr_{ROW['ZTF_GroupID']}"
        drawFIG(star_props[2], 'ZTFr', base_filename, extras)

    del is_CSS
    del is_ZTFg
    del is_ZTFr
    del is_ZTFi
    del prop_id
    del ROW
    del rahmsstr
    del decdmsstr
    #del base_filename


def drawFIG(star_prop, thisfilter, base_filename, extras):
    lc_data, flc_data, all_period_properties = star_prop
    prop_id, ROW, rahmsstr, decdmsstr = extras

    best_P = all_period_properties['P']
    P_fracs = LCtools.createHarmonicFrac(5)
    P_fracs_str = np.array(['1/5', '1/4', '1/3', '1/2', '', '2', '3', '4', '5'])
    test_Ps = P_fracs * best_P
    test_fs = test_Ps**-1
    test_P_FAPs = np.zeros(test_Ps.size) 

    test_powers =  all_period_properties['ls'].power(test_fs/u.d)
    logFAP_of_power = np.log10(all_period_properties['ls'].false_alarm_probability(test_powers)) #get new logFAP of new power

    for ii, use_this_P in enumerate(test_Ps):
        data = [flc_data['mjd'], flc_data['mag'], flc_data['magerr']]

        try:
            AFDdata = LCtools.AFD(data, use_this_P, alpha=0.99, Nmax=6)
            Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiSq, mtf = AFDdata
            Amp = y_fit.max() - y_fit.min()
            t0 = (flc_data['mjd'] - (phased_t * use_this_P)).max()
        except np.linalg.LinAlgError:
            print("Singular Matrix", rahmsstr, decdmsstr, thisfilter)
            AFD_data = None
            Nterms = None
            phase_fit = None
            y_fit = None
            resid = None
            reduced_ChiS = None
            mfit = None
            Amp = all_period_properties['a95']
            t0 = np.nan

        title = f"prop\_id={prop_id} $|$ {rahmsstr}{decdmsstr} $|$ {P_fracs_str[ii]}P = {use_this_P.round(6)} d $|$ log10(FAP) = {logFAP_of_power[ii].round(2)} $|$ Amp = {np.round(Amp, 3)} mag"

        fig = plt.figure(figsize=(10, 6), constrained_layout=True, dpi=600)
        fig.suptitle(title)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.1)

        inner_grid0 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], hspace=0.0, width_ratios=[1], height_ratios=[0.75, 0.25])
        inner_grid1 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], hspace=0.0, width_ratios=[1], height_ratios=[0.75, 0.25])
        inner_grid2 = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, :])

        ax3 = fig.add_subplot(inner_grid2[0]) #powerspec#fig.add_subplot(gs[0:2, 0])

        ax1 = fig.add_subplot(inner_grid0[:])

        ax2 = fig.add_subplot(inner_grid1[0])
        ax25 = fig.add_subplot(inner_grid1[1], sharex=ax2)
        plt.setp(ax2.get_xticklabels(), visible=False)


        plt_raw_lc(flc_data, ax1, title="", show_err_lines=True, plot_rejected=True)
        plt_any_lc_fig(flc_data, use_this_P, ax2, ax25, is_Periodic=True, title="", phasebin=True, bins=20, phasebinonly=False, show_err_lines=True, plot_rejected=False)

        ax1.set_ylabel(f"{thisfilter} mag")
        ax2.set_ylabel(f"{thisfilter} mag")

        ax3.plot(all_period_properties['frequency'], all_period_properties['power'], lw=0.5, c='k')
        ax3.axhline(all_period_properties['power'].mean()*5, c='r', ls='dashed', lw=0.5)
        ax3.axhline(y=all_period_properties['FAP_power_peak'], c='r', ls='dashed', alpha=0.5, lw=0.75)
        ax3.text(0.8 * all_period_properties['frequency'].max().to(1/u.d).value, all_period_properties['FAP_power_peak'] - 0.00, f"log(FAP) = {'-5'}", c='r')
        ax3.text(9.1, all_period_properties['power'].mean()*5, "5$<$A$>$", c='r')

        ymin, ymax = ax3.get_ylim()
        ax3.set_xlim(0, 10)
        xmin, xmax = ax3.get_xlim()
        if (use_this_P**-1 >= xmin) & (use_this_P**-1 <= xmax):
            ax3.annotate('',
                 xy=(((use_this_P*u.d)**-1).to(1/u.d).value, ymin),
                 xytext=(((use_this_P*u.d)**-1).to(1/u.d).value, ymin - (0.11*ymax)),
                 xycoords='data', annotation_clip=False, arrowprops=dict(arrowstyle="->", color='r'))

        ax3.set_xlabel('Frequency [d$^{-1}$]')
        ax3.set_ylabel('Power')

        ax3.axvline(x=((365.25 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
        ax3.axvline(x=((29.530587981 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
        ax3.axvline(x=((27.321661 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
        ax3.axvline(x=((1 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
        ax3.axvline(x=((1 / 2 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
        ax3.axvline(x=((1 / 3 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
        ax3.axvline(x=((1 / 4 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
        ax3.axvline(x=((1 / 5 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
        ax3.axvline(x=((1 / 6 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
        ax3.axvline(x=((1 / 7 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
        ax3.axvline(x=((1 / 8 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
        ax3.axvline(x=((1 / 9 * u.d)**-1).to(1/u.d).value, c='k', ls='dashed', alpha=0.5, lw=0.75)

        ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        dy = ymax - ymin
        ymajortick = np.max([0.02, np.round(ymax / 7, 2)])
        yminortick = ymajortick / 10

        ax3.yaxis.set_major_locator(ticker.MultipleLocator(ymajortick))
        ax3.yaxis.set_minor_locator(ticker.MultipleLocator(yminortick))

        #plt.tight_layout()
        plt.savefig(base_filename + f"_{P_fracs.round(2)[ii]}P.pdf", dpi=600)
        fig.clear()
        plt.close(fig)

    del lc_data
    del flc_data
    del all_period_properties
    del prop_id
    del ROW
    del rahmsstr
    del decdmsstr
    del best_P
    del P_fracs
    del P_fracs_str
    del test_Ps
    del test_fs
    del test_P_FAPs
    del test_powers
    del logFAP_of_power
    del ii
    del use_this_P
    del data
    del AFDdata
    del Nterms
    del phase_fit
    del y_fit
    del phased_t
    del resid
    del reduced_ChiSq
    del mtf
    del Amp
    del t0
    del title
    del fig
    del gs
    del inner_grid0
    del inner_grid1
    del inner_grid2
    del ax3
    del ax1
    del ax2
    del ax25
    del ymin
    del ymax
    del xmin
    del xmax
    del dy
    del ymajortick
    del yminortick

############################################################################
############################################################################
############################################################################
############################################################################
############################################################################

def TDSS_LC_ANALYSIS(prop_id):
    ROW = TDSS_prop[prop_id]

    object_ra = ROW['ra_GaiaEDR3']
    object_dec = ROW['dec_GaiaEDR3']
    
    c = ICRS(object_ra*u.degree, object_dec*u.degree)
    rahmsstr = c.ra.to_string(u.hour, precision=2, pad=True)
    decdmsstr = c.dec.to_string(u.degree, alwayssign=True, precision=2, pad=True)

    star_props = get_star(prop_id)

    extras = [prop_id, ROW, rahmsstr, decdmsstr]
    drawFIGS(star_props, extras)

    del ROW
    del object_ra
    del object_dec
    del c
    del rahmsstr
    del decdmsstr
    del star_props
    del extras
    gc.collect()


start_index = 23490
if __name__ == '__main__':
    with Pool(os.cpu_count()-2) as pool:
        r = list(tqdm.tqdm(pool.imap(TDSS_LC_ANALYSIS, range(start_index, len(TDSS_prop))), total=len(TDSS_prop)-start_index))

# prop_id = int(sys.argv[1])
# TDSS_LC_ANALYSIS(prop_id)
