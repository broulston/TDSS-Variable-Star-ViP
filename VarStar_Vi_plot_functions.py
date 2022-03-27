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
from scipy.stats import kde
from scipy.stats import f
from subprocess import *
import os
import glob
import re
import tqdm

from astropy.table import Table
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from astropy import coordinates as coords
from astropy.timeseries import LombScargle

import ResearchTools.LCtools as LCtools

np.seterr(divide='ignore', invalid='ignore')


def low_order_poly(mag, a, b, c, d, e, f_, g):
    return a + b * mag + c * mag**2 + d * mag**3 + e * mag**4 + f_ * mag**5 + g * mag**5


def LC_analysis(ROW, TDSSprop, CSS_LC_dir, ZTF_g_LCs, ZTF_r_LCs, ax, CSS_LC_plot_dir, ZTF_LC_plot_dir, Nepochs_required, minP=0.1, maxP=100.0, log10FAP=-10, checkHarmonic=False, plt_subLC=False, plot_rejected=False):
    is_CSS = ROW['CSSLC']
    is_ZTF_g = np.isfinite(ROW['ZTF_g_Nepochs'])
    is_ZTF_r = np.isfinite(ROW['ZTF_r_Nepochs'])

    ra_string = '{:0>9.5f}'.format(ROW['ra_GaiaEDR3'])
    dec_string = '{:0=+9.5f}'.format(ROW['dec_GaiaEDR3'])

    if is_CSS:
        try:
            lc_file = CSS_LC_dir+str(ROW['CSSID'])+'.dat'
            CSS_lc_data = Table.read(lc_file, format='ascii', names=['mjd', 'mag', 'magerr'])
            if len(CSS_lc_data)>=Nepochs_required:
                popt = np.array([-2.61242938e+01,  1.93636204e+00,  4.45971381e-01, -6.49419310e-02, 2.99231126e-03,  2.40758201e-01, -2.40805035e-01])
                magerr_resid_mean = 0.008825118765717422
                shift_const = 1.5 * magerr_resid_mean
                pred_magerr = low_order_poly(CSS_lc_data['mag'], popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
                bad_err_index = np.where(CSS_lc_data['magerr'] - (pred_magerr - shift_const) < 0.0)[0]
                CSS_lc_data['magerr'][bad_err_index] = pred_magerr[bad_err_index]

                CSS_flc_data, LC_stat_properties = LCtools.process_LC(CSS_lc_data.copy(), fltRange=5.0)
                LC_period_properties, all_CSS_period_properties = LCtools.perdiodSearch(CSS_flc_data, minP=minP, maxP=maxP, log10FAP=log10FAP, checkHarmonic=checkHarmonic)
                all_CSS_period_properties = {**LC_stat_properties, **all_CSS_period_properties}
                CSS_prop = {**LC_stat_properties, **LC_period_properties}
                CSS_prop['lc_id'] = ROW['CSSID']

                # CSS_flc_data = [CSS_flc_data['mjd'].data, CSS_flc_data['mag'].data, CSS_flc_data['magerr'].data]

                if plt_subLC:
                    # fig = plt.figure()
                    # title = "RA: {!s} DEC: {!s} $|$ P = {!s}d $|$ {!s}w \n log10(FAP) = {!s} $|$ {!s} mag = {!s} $|$ Amp = {!s} $|$ t0 = {!s}".format(ra_string, dec_string, np.round(, 5), LC_period_properties['time_whittened'], np.round(LC_period_properties['logProb'], 2), "CSS", np.round(ROW['CSSmag'],2), np.round(LC_period_properties['Amp'],3), np.round(all_CSS_period_properties['t0'],5))   
                    # LCtools.plt_lc(CSS_flc_data, all_CSS_period_properties, fig=fig, title=title, plt_resid=True, phasebin=True, bins=25)
                    # # LCtools.plt_lc(CSS_flc_data, all_CSS_period_properties, fig=fig, title=title, plt_resid=True, phasebin=True, bins=25, plot_rejected=plot_rejected)
                    # plt.savefig(CSS_LC_plot_dir+ra_string+dec_string+"_CSS.pdf", dpi=600)
                    # plt.clf()
                    # plt.close()

                    freq_grid = all_CSS_period_properties['frequency']
                    power = all_CSS_period_properties['power']
                    logFAP_limit = log10FAP
                    FAP_power_peak = all_CSS_period_properties['ls'].false_alarm_level(10**logFAP_limit)
                    df = (1 * u.d)**-1
                    # title = "RA: {!s} DEC: {!s} $|$ P = {!s}d $|$ $m$ = {!s}".format(ra_string, dec_string, np.round(test_P, 7), np.round(mean_mag, 2))
                    title = "RA: {!s} DEC: {!s} CSS".format(ra_string, dec_string)
                    LCtools.plot_LC_analysis(CSS_flc_data, LC_period_properties['P'], freq_grid, power, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, df=df, title=title)
                    plt.savefig(CSS_LC_plot_dir + ra_string + dec_string + "_CSS.pdf", dpi=600)
                    # plt.show()
                    plt.clf()
                    plt.close()
            else:
                all_CSS_period_properties = None
                CSS_prop = None
        except FileNotFoundError:
            all_CSS_period_properties = None
            CSS_prop = None
    else:
        all_CSS_period_properties = None
        CSS_prop = None

    if is_ZTF_g:
        ZTF_g_lc_data = ZTF_g_LCs[(ZTF_g_LCs['ZTF_GroupID'] == ROW['ZTF_GroupID'])]['mjd', 'mag', 'magerr']
        if len(ZTF_g_lc_data)>=Nepochs_required:
            ZTF_gflc_data, LC_stat_properties = LCtools.process_LC(ZTF_g_lc_data.copy(), fltRange=5.0)
            LC_period_properties, all_ZTFg_period_properties = LCtools.perdiodSearch(ZTF_gflc_data, minP=minP, maxP=maxP, log10FAP=log10FAP, checkHarmonic=checkHarmonic)
            all_ZTFg_period_properties = {**LC_stat_properties, **all_ZTFg_period_properties}
            ZTF_g_prop = {**LC_stat_properties, **LC_period_properties}
            ZTF_g_prop['lc_id'] =  ROW['ZTF_GroupID']

            # ZTF_gflc_data = [ZTF_gflc_data['mjd'].data, ZTF_gflc_data['mag'].data, ZTF_gflc_data['magerr'].data]

            if plt_subLC:
                # fig = plt.figure()
                # title = "RA: {!s} DEC: {!s} $|$ P = {!s}d $|$ {!s}w \n log10(FAP) = {!s} $|$ {!s} mag = {!s} $|$ Amp = {!s} $|$ t0 = {!s}".format(ra_string, dec_string, np.round(LC_period_properties['P'], 5), LC_period_properties['time_whittened'], np.round(LC_period_properties['logProb'], 2), "ZTF g", np.round(ROW['ZTF_g_mag'],2), np.round(LC_period_properties['Amp'],3), np.round(all_ZTFg_period_properties['t0'],5))   
                # LCtools.plt_lc(ZTF_gflc_data, all_ZTFg_period_properties, fig=fig, title=title, plt_resid=True, phasebin=True, bins=25, plot_rejected=plot_rejected)
                # plt.savefig(ZTF_LC_plot_dir+"g/"+ra_string+dec_string+"_ZTFg.pdf", dpi=600)
                # plt.clf()
                # plt.close()

                freq_grid = all_ZTFg_period_properties['frequency']
                power = all_ZTFg_period_properties['power']
                logFAP_limit = log10FAP
                FAP_power_peak = all_ZTFg_period_properties['ls'].false_alarm_level(10**logFAP_limit)
                df = (1 * u.d)**-1
                # title = "RA: {!s} DEC: {!s} $|$ P = {!s}d $|$ $m$ = {!s}".format(ra_string, dec_string, np.round(test_P, 7), np.round(mean_mag, 2))
                title = "RA: {!s} DEC: {!s} ZTF g ".format(ra_string, dec_string)
                LCtools.plot_LC_analysis(ZTF_gflc_data, LC_period_properties['P'], freq_grid, power, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, df=df, title=title)
                plt.savefig(ZTF_LC_plot_dir+"g/"+ra_string+dec_string+"_ZTFg.pdf", dpi=600)
                # plt.show()
                plt.clf()
                plt.close()
        else:
            all_ZTFg_period_properties = None
            ZTF_g_prop = None
    else:
        all_ZTFg_period_properties = None
        ZTF_g_prop = None

    if is_ZTF_r:
        ZTF_r_lc_data = ZTF_r_LCs[(ZTF_r_LCs['ZTF_GroupID'] == ROW['ZTF_GroupID'])]['mjd', 'mag', 'magerr']
        if len(ZTF_r_lc_data)>=Nepochs_required:
            ZTF_rflc_data, LC_stat_properties = LCtools.process_LC(ZTF_r_lc_data.copy(), fltRange=5.0)
            LC_period_properties, all_ZTFr_period_properties = LCtools.perdiodSearch(ZTF_rflc_data, minP=minP, maxP=maxP, log10FAP=log10FAP, checkHarmonic=checkHarmonic)
            all_ZTFr_period_properties = {**LC_stat_properties, **all_ZTFr_period_properties}
            ZTF_r_prop = {**LC_stat_properties, **LC_period_properties}
            ZTF_r_prop['lc_id'] =  ROW['ZTF_GroupID']

            # ZTF_rflc_data = [ZTF_rflc_data['mjd'].data, ZTF_rflc_data['mag'].data, ZTF_rflc_data['magerr'].data]

            if plt_subLC:
                # fig = plt.figure()
                # title = "RA: {!s} DEC: {!s} $|$ P = {!s}d $|$ {!s}w \n log10(FAP) = {!s} $|$ {!s} mag = {!s} $|$ Amp = {!s} $|$ t0 = {!s}".format(ra_string, dec_string, np.round(LC_period_properties['P'], 5), LC_period_properties['time_whittened'], np.round(LC_period_properties['logProb'], 2), "ZTF r", np.round(ROW['ZTF_r_mag'],2), np.round(LC_period_properties['Amp'],3), np.round(all_ZTFr_period_properties['t0'],5))   
                # LCtools.plt_lc(ZTF_rflc_data, all_ZTFr_period_properties, fig=fig, title=title, plt_resid=True, phasebin=True, bins=25, plot_rejected=plot_rejected)
                # plt.savefig(ZTF_LC_plot_dir+"r/"+ra_string+dec_string+"_ZTFr.pdf", dpi=600)
                # plt.clf()
                # plt.close()

                freq_grid = all_ZTFr_period_properties['frequency']
                power = all_ZTFr_period_properties['power']
                logFAP_limit = log10FAP
                FAP_power_peak = all_ZTFr_period_properties['ls'].false_alarm_level(10**logFAP_limit)
                df = (1 * u.d)**-1
                # title = "RA: {!s} DEC: {!s} $|$ P = {!s}d $|$ $m$ = {!s}".format(ra_string, dec_string, np.round(test_P, 7), np.round(mean_mag, 2))
                title = "RA: {!s} DEC: {!s} ZTF r".format(ra_string, dec_string)
                LCtools.plot_LC_analysis(ZTF_rflc_data, LC_period_properties['P'], freq_grid, power, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, df=df, title=title)
                plt.savefig(ZTF_LC_plot_dir+"r/"+ra_string+dec_string+"_ZTFr.pdf", dpi=600)
                # plt.show()
                plt.clf()
                plt.close()
        else:
            all_ZTFr_period_properties = None
            ZTF_r_prop = None
    else:
        all_ZTFr_period_properties = None
        ZTF_r_prop = None

    best_LC = 'ZTF_g'# = find_best_LC(all_CSS_period_properties, all_ZTFg_period_properties, all_ZTFr_period_properties)

    if ROW['isDrake']:
        title_line2 = "\n Drake: P={!s} $|$ Amp={!s} $|$ VarType={!s}".format(ROW['Drake_Per'], ROW['Drake_Vamp'], TDSSprop.Drake_num_to_vartype[(ROW['Drake_Cl']-1).astype(int)]['Var_Type'].strip().replace("*\*", " "))
    else:
        title_line2 = ""

    # p_factor = 1.0
    # all_CSS_period_properties['P'] = all_CSS_period_properties['P']   * p_factor
    # all_ZTFg_period_properties['P'] = all_ZTFg_period_properties['P'] * p_factor
    # all_ZTFr_period_properties['P'] = all_ZTFr_period_properties['P'] * p_factor
    if best_LC=='CSS':
        title_line1 = "CSS ID: {!s} $|$ P={!s} $|$ logProb={!s} \n Amp={!s} $|$ ngood={!s} $|$ nreject={!s} $|$ Con={!s} \n nabove={!s} ({!s}\%) $|$ nbelow={!s} ({!s}\%) $|$ VarStat={!s} \n Tspan100={!s} $|$ Tspan95={!s}\n m={!s} $|$ b={!s} $|$ $\chi^2$={!s} \n a={!s} $|$ b={!s} $|$ c={!s} $|$ $\chi^2$={!s}".format(ROW['CSSID'], str(np.round(all_CSS_period_properties['P'],5))+"w"+str(all_CSS_period_properties['time_whittened']), np.round(all_CSS_period_properties['logProb'],3), np.round(all_CSS_period_properties['Amp'],2), all_CSS_period_properties['ngood'], all_CSS_period_properties['nrejects'], np.round(all_CSS_period_properties['Con'],3), all_CSS_period_properties['nabove'], np.int(np.round((all_CSS_period_properties['nabove']/all_CSS_period_properties['ngood'])*100,2)), all_CSS_period_properties['nbelow'], np.int(np.round((all_CSS_period_properties['nbelow']/all_CSS_period_properties['ngood'])*100,2)), np.round(all_CSS_period_properties['VarStat'],2), np.round(all_CSS_period_properties['Tspan100'],2), np.round(all_CSS_period_properties['Tspan95'],2), np.round(all_CSS_period_properties['m'],4), np.round(all_CSS_period_properties['b_lin'],4), np.round(all_CSS_period_properties['chi2_lin'],2), np.round(all_CSS_period_properties['a'],4), np.round(all_CSS_period_properties['b_quad'],4), np.round(all_CSS_period_properties['c'],4),np.round(all_CSS_period_properties['chi2_quad'],2))
        title_str = title_line1+title_line2
        plt_lc(CSS_flc_data, all_CSS_period_properties,  ROW=ROW, title=title_str, ax=ax, plot_rejected=plot_rejected)
    elif best_LC=='ZTF_g':
        title_line1 = "ZTF g GroupID: {!s} $|$ P={!s} $|$ logProb={!s} \n Amp={!s} $|$ ngood={!s} $|$ nreject={!s} $|$ Con={!s} \n nabove={!s} ({!s}\%) $|$ nbelow={!s} ({!s}\%) $|$ VarStat={!s} \n Tspan100={!s} $|$ Tspan95={!s}\n m={!s} $|$ b={!s} $|$ $\chi^2$={!s} \n a={!s} $|$ b={!s} $|$ c={!s} $|$ $\chi^2$={!s}".format(ROW['ZTF_GroupID'], str(np.round(all_ZTFg_period_properties['P'],5))+"w"+str(all_ZTFg_period_properties['time_whittened']), np.round(all_ZTFg_period_properties['logProb'],3), np.round(all_ZTFg_period_properties['Amp'],2), all_ZTFg_period_properties['ngood'], all_ZTFg_period_properties['nrejects'], np.round(all_ZTFg_period_properties['Con'],3), all_ZTFg_period_properties['nabove'], np.int(np.round((all_ZTFg_period_properties['nabove']/all_ZTFg_period_properties['ngood'])*100,2)), all_ZTFg_period_properties['nbelow'], np.int(np.round((all_ZTFg_period_properties['nbelow']/all_ZTFg_period_properties['ngood'])*100,2)), np.round(all_ZTFg_period_properties['VarStat'],2), np.round(all_ZTFg_period_properties['Tspan100'],2), np.round(all_ZTFg_period_properties['Tspan95'],2), np.round(all_ZTFg_period_properties['m'],4), np.round(all_ZTFg_period_properties['b_lin'],4), np.round(all_ZTFg_period_properties['chi2_lin'],2), np.round(all_ZTFg_period_properties['a'],4), np.round(all_ZTFg_period_properties['b_quad'],4), np.round(all_ZTFg_period_properties['c'],4),np.round(all_ZTFg_period_properties['chi2_quad'],2))
        title_str = title_line1+title_line2
        plt_lc(ZTF_gflc_data, all_ZTFg_period_properties,  ROW=ROW, title=title_str, ax=ax, plot_rejected=plot_rejected)
    elif best_LC=='ZTF_r':
        title_line1 = "ZTF r GroupID: {!s} $|$ P={!s} $|$ logProb={!s} \n Amp={!s} $|$ ngood={!s} $|$ nreject={!s} $|$ Con={!s} \n nabove={!s} ({!s}\%) $|$ nbelow={!s} ({!s}\%) $|$ VarStat={!s} \n Tspan100={!s} $|$ Tspan95={!s}\n m={!s} $|$ b={!s} $|$ $\chi^2$={!s} \n a={!s} $|$ b={!s} $|$ c={!s} $|$ $\chi^2$={!s}".format(ROW['ZTF_GroupID'], str(np.round(all_ZTFr_period_properties['P'],5))+"w"+str(all_ZTFr_period_properties['time_whittened']), np.round(all_ZTFr_period_properties['logProb'],3), np.round(all_ZTFr_period_properties['Amp'],2), all_ZTFr_period_properties['ngood'], all_ZTFr_period_properties['nrejects'], np.round(all_ZTFr_period_properties['Con'],3), all_ZTFr_period_properties['nabove'], np.int(np.round((all_ZTFr_period_properties['nabove']/all_ZTFr_period_properties['ngood'])*100,2)), all_ZTFr_period_properties['nbelow'], np.int(np.round((all_ZTFr_period_properties['nbelow']/all_ZTFr_period_properties['ngood'])*100,2)), np.round(all_ZTFr_period_properties['VarStat'],2), np.round(all_ZTFr_period_properties['Tspan100'],2), np.round(all_ZTFr_period_properties['Tspan95'],2), np.round(all_ZTFr_period_properties['m'],4), np.round(all_ZTFr_period_properties['b_lin'],4), np.round(all_ZTFr_period_properties['chi2_lin'],2), np.round(all_ZTFr_period_properties['a'],4), np.round(all_ZTFr_period_properties['b_quad'],4), np.round(all_ZTFr_period_properties['c'],4),np.round(all_ZTFr_period_properties['chi2_quad'],2))
        title_str = title_line1+title_line2
        plt_lc(ZTF_rflc_data, all_ZTFr_period_properties, ROW=ROW, title=title_str, ax=ax, plot_rejected=plot_rejected)
    else:
        assert best_LC is None, f"ERROR: no best LC found: ({best_LC})"

    return CSS_prop, ZTF_g_prop, ZTF_r_prop, best_LC


def find_best_LC(all_CSS_period_properties, all_ZTFg_period_properties, all_ZTFr_period_properties):
    base_LCs = ['CSS', 'ZTF_g', 'ZTF_r']
    best_LC = []
    if all_CSS_period_properties:
        best_LC.append('CSS')
        CSS_ngood = all_CSS_period_properties['ngood']
        CSS_ferrmn = all_CSS_period_properties['ferrmn']
        CSS_is_Periodic = all_CSS_period_properties['is_Periodic']
        CSS_logProb = all_CSS_period_properties['logProb']
        CSS_ChiSQ = all_CSS_period_properties['Chi2']
    else:
        CSS_ngood = 0
        CSS_ferrmn = None
        CSS_is_Periodic = False
        CSS_logProb = None
        CSS_ChiSQ = 0
    if all_ZTFg_period_properties:
        best_LC.append('ZTF_g')
        ZTF_g_ngood = all_ZTFg_period_properties['ngood']
        ZTF_g_ferrmn = all_ZTFg_period_properties['ferrmn']
        ZTF_g_is_Periodic = all_ZTFg_period_properties['is_Periodic']
        ZTF_g_logProb = all_ZTFg_period_properties['logProb']
        ZTF_g_ChiSQ = all_ZTFg_period_properties['Chi2']
    else:
        ZTF_g_ngood = 0
        ZTF_g_ferrmn = None
        ZTF_g_is_Periodic = False
        ZTF_g_logProb = None
        ZTF_g_ChiSQ = 0
    if all_ZTFr_period_properties:
        best_LC.append('ZTF_r')
        ZTF_r_ngood = all_ZTFr_period_properties['ngood']
        ZTF_r_ferrmn = all_ZTFr_period_properties['ferrmn']
        ZTF_r_is_Periodic = all_ZTFr_period_properties['is_Periodic']
        ZTF_r_logProb = all_ZTFr_period_properties['logProb']
        ZTF_r_ChiSQ = all_ZTFr_period_properties['Chi2']
    else:
        ZTF_r_ngood = 0
        ZTF_r_ferrmn = None
        ZTF_r_is_Periodic = False
        ZTF_r_logProb = None
        ZTF_r_ChiSQ = 0

    if len(best_LC) == 1:
        return best_LC[0]
    else:
        ngood = [CSS_ngood, ZTF_g_ngood, ZTF_r_ngood]
        ferrmn = [CSS_ferrmn, ZTF_g_ferrmn, ZTF_r_ferrmn]
        is_periodic = [CSS_is_Periodic, ZTF_g_is_Periodic, ZTF_r_is_Periodic]
        logProb = [CSS_logProb, ZTF_g_logProb, ZTF_r_logProb]
        ChiSqs = [CSS_ChiSQ, ZTF_g_ChiSQ, ZTF_r_ChiSQ]

        ngood = [ii if ii is not None else 0 for ii in ngood]
        ferrmn = [ii if ii is not None else np.inf for ii in ferrmn]
        is_periodic = [ii if ii is not None else False for ii in is_periodic]
        logProb = [ii if ii is not None else np.inf for ii in logProb]

        # ngood[ngood is None] = 0
        # ferrmn[ferrmn is None] = np.inf
        # is_periodic[is_periodic is None] = False
        # logProb[logProb is None] = np.inf

        if np.any(is_periodic):
            if  np.where(is_periodic)[0].size==1:
                return base_LCs[np.where(is_periodic)[0][0]]
            else:
                #return base_LCs[np.argmin(ferrmn)]
                periodNames = [base_LCs[ii] for ii in np.where(is_periodic)[0]]
                periodChiSqs = [ChiSqs[ii] for ii in np.where(is_periodic)[0]]
                return periodNames[np.argmax(periodChiSqs)]
        else:
            #return base_LCs[np.argmax(ngood)]
            return base_LCs[np.argmax(ChiSqs)]


def plt_lc(lc_data, all_period_properties, ax, ROW, title="", plt_resid=False, phasebin=False, bins=25, RV=False, plot_rejected=False):
    goodQualIndex = np.where(lc_data['QualFlag']==True)[0]
    badQualIndex = np.where(lc_data['QualFlag']==False)[0]
    mjd = lc_data['mjd'][goodQualIndex].data
    mag = lc_data['mag'][goodQualIndex].data
    err = lc_data['magerr'][goodQualIndex].data

    AFD_data = all_period_properties['AFD']
    #AFD_data = LCtools.AFD([mjd, mag, err], all_period_properties['P'], alpha=0.99, Nmax=6)

    if len(AFD_data)==8:
        deasliased_period, Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiS, mfit = AFD_data
    else:
        Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiS, mfit = AFD_data

    t0 = all_period_properties['t0']
    P = all_period_properties['P']
    mjd_bad = lc_data['mjd'][badQualIndex].data
    phase_bad = ((mjd_bad - t0) / P) % 1
    mag_bad = lc_data['mag'][badQualIndex].data
    err_bad = lc_data['magerr'][badQualIndex].data

    binned_phase, binned_mag, binned_err = LCtools.bin_phaseLC(phased_t, mag, err, bins=bins)

    is_Periodic = all_period_properties['is_Periodic']
    if  is_Periodic:
        if plt_resid:
            frame1=fig.add_axes((.1,.3,.8,.6))
            #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
            plt_ax = frame1
        else:
            plt_ax = ax
        
        ax.set_title(title)
        #ax.title("Nterms: "+str(Nterms))
        if RV:
            ax.errorbar(phased_t, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5)
            ax.errorbar(phased_t+1, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5)
        else:
            ax.errorbar(phased_t, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
            ax.errorbar(phased_t+1, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
            if plot_rejected:
                plt_ax.errorbar(phase_bad, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=0.3)
                plt_ax.errorbar(phase_bad+1, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=0.3)
        #ax.errorbar(phased_t+2, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)

        # N_terms
        ax.plot(phase_fit, y_fit, 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')
        ax.plot(phase_fit+1, y_fit, 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')
        #ax.plot(phase_fit+2, y_fit, 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')

        fmagmn = np.mean(mag)
        ferrmn = np.mean(err)
        fmag_stdev = np.std(mag)

        ax.axhline(fmagmn, color='r', ls='-', lw=2, label='Mean Mag')
        ax.axhline(fmagmn+3*ferrmn, color='g', ls='-.', lw=2 ,alpha=0.5, label='3X Mag Err')
        ax.axhline(fmagmn-3*ferrmn, color='g', ls='-.', lw=2 ,alpha=0.5)
        ax.axhline(fmagmn+3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5, label='3X Mag StDev')
        ax.axhline(fmagmn-3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5)

        #ax.set_xlim(0.0, 2.0)
        #ax.set_ylim(18.2, 16.7)
        if RV:
            ax.set_ylabel('RV [km s$^{-1}$]')
        else:
            ax.set_ylabel('mag')
            #ax.set_xlabel('Phase')
            ax.invert_yaxis()

        ax.grid()
        if phasebin:
            ax.errorbar(binned_phase, binned_mag, binned_err, fmt='sr', ecolor='red', lw=1, ms=4, capsize=1.5, alpha=0.3)
            ax.errorbar(binned_phase+1, binned_mag, binned_err, fmt='sr', ecolor='red', lw=1, ms=4, capsize=1.5, alpha=0.3)

        if plt_resid:
            frame1.set_xlim(0.0, 2.0)
            frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
            frame2 = fig.add_axes((.1,.1,.8,.2))

            frame2.errorbar(phased_t, resid, fmt='.k', ecolor='k', lw=1, ms=4, capsize=1.5, alpha=0.3)
            frame2.errorbar(phased_t+1, resid, fmt='.k', ecolor='k', lw=1, ms=4, capsize=1.5, alpha=0.3)
            #plt_ax.errorbar(phased_t+2, resid, fmt='.k', ecolor='k', lw=1, ms=4, capsize=1.5, alpha=0.3)

            frame2.grid()
            #ax2 = plt.gca()
            frame2.set_xlim(0.0, 2.0)
            frame2.set_ylim(1.5*resid.min(), 1.5*resid.max())
            #frame2.yaxis.set_major_locator(plt.MaxNLocator(4))

            frame2.set_xlabel(r'Phase')
            frame2.set_ylabel('Residual')# \n N$_{terms} = 4$')
        else:
            ax.set_xlabel('Phase')

        if (P <= 0.5):
            SDSS_spec_ExpT = 1.0 / 24.0  # in days
            spec_mjd = ROW['mjd']
            spec_in_phase_begin = (((spec_mjd - 0.5 * SDSS_spec_ExpT) - t0) / P) % 1
            spec_in_phase_end = (((spec_mjd + 0.5 * SDSS_spec_ExpT) - t0) / P) % 1

            ax.axvspan(spec_in_phase_begin, spec_in_phase_end, color='r', alpha=0.2)
            ax.axvspan(spec_in_phase_begin + 1, spec_in_phase_end + 1, color='r', alpha=0.2)
        
        ax.legend(loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.25))
    else:
        # fig1 = plt.figure()
        ax.set_title(title)
        if RV:
            ax.errorbar(mjd, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5)
        else:
            ax.errorbar(mjd, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
            if plot_rejected:
                ax.errorbar(mjd_bad, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=0.3)

        if (ROW['mjd'] >= mjd.min()) and (ROW['mjd'] <= mjd.max()):
            ax.axvline(x=ROW['mjd'], lw=0.75, color='r')

        fmagmn = np.mean(mag)
        ferrmn = np.mean(err)
        fmag_stdev = np.std(mag)

        ax.axhline(fmagmn, color='r', ls='-', lw=2, label='Mean Mag')
        ax.axhline(fmagmn+3*ferrmn, color='g', ls='-.', lw=2, alpha=0.5, label='3X Mag Err')
        ax.axhline(fmagmn-3*ferrmn, color='g', ls='-.', lw=2, alpha=0.5)
        ax.axhline(fmagmn+3*fmag_stdev, color='b', ls=':', lw=2, alpha=0.5, label='3X Mag StDev')
        ax.axhline(fmagmn-3*fmag_stdev, color='b', ls=':', lw=2, alpha=0.5)

        ax.legend(loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.25))
        # ax.set_xlim(0.0, 2.0)
        # ax.set_ylim(18.2, 16.7)
        ax.set_xlabel('MJD')
        ax.grid()
        if RV:
            ax.set_ylabel('RV [km s$^{-1}$]')
        else:
            ax.set_ylabel('mag')
            ax.invert_yaxis()


def plot_SDSSspec(ROW, TDSSprop, prop_id, spec_dir, plt_ax):
    line_list_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/General/SpecLineLists/"
    # spectral_type_prop_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/General/SpecTypeProp/"

    spec_box_size = 10
    temp_box_size = 50

    xmin = 3800
    xmax = 10000
    # sig_range = 3.0
    major_tick_space = 1000
    minor_tick_space = 100

    ra_string = '{:0>9.5f}'.format(ROW['ra_GaiaEDR3'])
    dec_string = '{:0=+9.5f}'.format(ROW['dec_GaiaEDR3'])
    plate_string = '{:0>4}'.format(ROW['plate'])
    mjd_string = '{:0>5}'.format(ROW['mjd'])
    fiberid_string = '{:0>4}'.format(ROW['fiber'])
    short_spec_filename = f"spec-{plate_string}-{mjd_string}-{fiberid_string}.fits"
    try:
        file_data = fits.open(spec_dir + short_spec_filename)  # cols are wavelength,flux
        wavelength = 10.0**file_data[1].data['loglam']
        flux = file_data[1].data['flux']
        # err = np.sqrt(1 / file_data[1].data['ivar'])
    # except IOError:
    except:
        # throw_error[ii] = 1
        print("Spec plot err: ", ra_string, dec_string, short_spec_filename)

    flux = removeSdssStitchSpike(wavelength, flux)

    specTypeMatch = ROW['PyHammerSpecType'].strip()
    pyhammer_RV = str(np.round(ROW['PyHammerRV']))  # str(pyhammer_RV)

    if "+" in specTypeMatch:
        template_file_dir = "/Users/benjaminroulston/Dropbox/GitHub/PyHammer/resources/templates_SB2/"
        tempName = specTypeMatch + ".fits"
        spectype1, spectype2 = specTypeMatch.replace('+', " ").split()

        spectype1 = splitSpecType(spectype1)[0]
        spectype2 = splitSpecType(spectype2)[0]
    else:
        specTypeMatch_code, specTypeMatch_subType_code = splitSpecType(specTypeMatch)

        spec_code_alph = np.array(['O', 'B', 'A', 'F', 'G', 'K', 'M', 'L', 'dC', 'DA'])
        # spec_code_num = np.arange(10)

        this_spec_num_code = np.where(spec_code_alph == specTypeMatch_code)[0][0]

        template_file_dir = "/Users/benjaminroulston/Dropbox/GitHub/PyHammer/resources/templates/"
        if this_spec_num_code == 0:
            tempName = 'O' + str(specTypeMatch_subType_code) + '.fits'
        # Spectral type B
        elif this_spec_num_code == 1:
            tempName = 'B' + str(specTypeMatch_subType_code) + '.fits'
        # Spectral types A0, A1, A2 (where there are no metallicity changes)
        elif this_spec_num_code == 2 and float(specTypeMatch_subType_code) < 3:
            tempName = 'A' + str(specTypeMatch_subType_code) + '.fits'
        # Spectral type A3 through A9
        elif this_spec_num_code == 2 and float(specTypeMatch_subType_code) > 2:
            tempName = 'A' + str(specTypeMatch_subType_code) + '_-1.0_Dwarf.fits'
        # Spectral type F
        elif this_spec_num_code == 3:
            tempName = 'F' + str(specTypeMatch_subType_code) + '_-1.0_Dwarf.fits'
        # Spectral type G
        elif this_spec_num_code == 4:
            tempName = 'G' + str(specTypeMatch_subType_code) + '_+0.0_Dwarf.fits'
        # Spectral type K
        elif this_spec_num_code == 5:
            tempName = 'K' + str(specTypeMatch_subType_code) + '_+0.0_Dwarf.fits'
        # Spectral type M (0 through 8)
        elif this_spec_num_code == 6 and float(specTypeMatch_subType_code) < 9:
            tempName = 'M' + str(specTypeMatch_subType_code) + '_+0.0_Dwarf.fits'
        # Spectral type M9 (no metallicity)
        elif this_spec_num_code == 6 and float(specTypeMatch_subType_code) == 9:
            tempName = 'M' + str(specTypeMatch_subType_code) + '.fits'
        # Spectral type L
        elif this_spec_num_code == 7:
            tempName = 'L' + str(specTypeMatch_subType_code) + '.fits'
        elif this_spec_num_code == 8:
            tempName = 'dC' + str(specTypeMatch_subType_code) + '.fits'
        elif this_spec_num_code == 9:
            tempName = 'DA' + str(specTypeMatch_subType_code) + '.fits'
    # Open the template
    temp = fits.open(template_file_dir + tempName)
    temp_loglam = temp[1].data.field('LogLam')
    temp_lam = 10.0**temp_loglam
    temp_flux = temp[1].data.field('Flux')

    if "+" in specTypeMatch:
        line_lis_all1 = np.genfromtxt(line_list_dir + "spec_types/" + spectype1 + "star_lines.list", comments='#', dtype="S")
        line_lis_all2 = np.genfromtxt(line_list_dir + "spec_types/" + spectype2 + "star_lines.list", comments='#', dtype="S")

        lineList_wavelength1 = np.float64(line_lis_all1[:, 0])
        lineList_wavelength2 = np.float64(line_lis_all2[:, 0])
        lineList_wavelength = np.hstack((lineList_wavelength1, lineList_wavelength2))
        lineList_labels = np.empty(lineList_wavelength.size, dtype="U60")
        for ii in range(lineList_wavelength1.size):
            lineList_labels[ii] = line_lis_all1[ii, 1].decode(encoding="utf-8", errors="strict")
        for ii in range(lineList_wavelength2.size):
            jj = ii + lineList_wavelength1.size
            lineList_labels[jj] = line_lis_all2[ii, 1].decode(encoding="utf-8", errors="strict")
    else:
        # line_lis_all = np.genfromtxt("aaaLineList_2.list",comments='#',dtype="S")
        # line_lis_all = np.genfromtxt(line_list_dir+"H_lines.list",comments='#',dtype="S")
        # line_lis_all = np.genfromtxt(line_list_dir+"spec_types/"+matched_spec_type+"star_lines.list",comments='#',dtype="S")
        line_lis_all = np.genfromtxt(f"{line_list_dir}spec_types/{specTypeMatch_code}star_lines.list", comments='#', dtype="S")

        lineList_wavelength = np.float64(line_lis_all[:, 0])
        lineList_labels = np.empty(lineList_wavelength.size, dtype="U60")
        for ii in range(lineList_wavelength.size):
            lineList_labels[ii] = line_lis_all[ii, 1].decode(encoding="utf-8", errors="strict")

    trim_spectrum_left = 10  # number of pixels to trim from left side
    smooth_flux = smooth(flux[trim_spectrum_left:], spec_box_size)
    smooth_wavelength = smooth(wavelength[trim_spectrum_left:], spec_box_size)

    smooth_temp_flux = smooth(temp_flux, temp_box_size)
    smooth_temp_wavelength = smooth(temp_lam, temp_box_size)

    lam8000_index = np.argmin(np.abs(smooth_wavelength - 8000.0))
    current_spec_flux_at_8000 = smooth_flux[lam8000_index]
    temp_flux_scaled = temp_flux * current_spec_flux_at_8000
    smooth_temp_flux = smooth_temp_flux * current_spec_flux_at_8000

    plotted_region = np.where((smooth_wavelength >= xmin) & (smooth_wavelength <= xmax))[0]
    plotted_region_temp = np.where((smooth_temp_wavelength >= xmin) & (smooth_temp_wavelength <= xmax))[0]
    ymin = min(smooth_flux[plotted_region].min(), smooth_temp_flux[plotted_region_temp].min())
    ymax = max(smooth_flux[plotted_region].max(), smooth_temp_flux[plotted_region_temp].max())

    this_EqW = eqw(temp_lam, temp_flux_scaled, wavelength, flux)
    cz = np.round(file_data[2].data['Z_NOQSO'][0] * (const.c.value / 1000), 2)
    cz_err = np.round(file_data[2].data['Z_ERR_NOQSO'][0] * (const.c.value / 1000), 2)
    try:
        subclass = file_data[2].data['SUBCLASS_NOQSO'][0].split()[0]
    except:
        subclass = file_data[2].data['SUBCLASS_NOQSO'][0]
    if np.isnan(this_EqW):
        EqW_string = ""
        plot_title = str("RA: "+ra_string+", DEC: "+dec_string+" $|$ cz = "+str(cz)+"$\pm$"+str(cz_err)+" km s$^{-1}$ $|$ SDSS Subclass = "
                    +str(subclass)+"\n PyHammer = "+specTypeMatch+EqW_string+", RV = "+pyhammer_RV+" km s$^{-1}$"+"\n "
                    +"prop. $|$ Plate = "+plate_string+" MJD = "+mjd_string+" Fiberid = "+fiberid_string+" $|$ GaiaEDR3 Dist = "+str(np.int(np.round(ROW['rpgeo'],2)))
                    +" pc (SNR = "+str(np.round(ROW['parallax']/ROW['parallax_error'],2))+") $|$ GaiaEDR3 PMtot = "+str(np.round(TDSSprop.gaia_pmTOT[prop_id],2))
                    +" mas/yr (SNR = "+str(np.round(TDSSprop.gaia_pmTOT[prop_id]/TDSSprop.gaia_pmTOT_error[prop_id], 2))+")")
    elif this_EqW > -2.0:
        EqW_string = ""
        plot_title = str("RA: "+ra_string+", DEC: "+dec_string+" $|$ cz = "+str(cz)+"$\pm$"+str(cz_err)+" km s$^{-1}$ $|$ SDSS Subclass = "
                    +str(subclass)+"\n PyHammer = "+specTypeMatch+EqW_string+", RV = "+pyhammer_RV+" km s$^{-1}$"+"\n "
                    +"prop. $|$ Plate = "+plate_string+" MJD = "+mjd_string+" Fiberid = "+fiberid_string+" $|$ GaiaEDR3 Dist = "+str(np.int(np.round(ROW['rpgeo'],2)))
                    +" pc (SNR = "+str(np.round(ROW['parallax']/ROW['parallax_error'],2))+") $|$ GaiaEDR3 PMtot = "+str(np.round(TDSSprop.gaia_pmTOT[prop_id],2))
                    +" mas/yr (SNR = "+str(np.round(TDSSprop.gaia_pmTOT[prop_id]/TDSSprop.gaia_pmTOT_error[prop_id], 2))+")")
    else:
        EqW_string = "e"
        this_EqW_str = str(np.round(this_EqW,2))
        plot_title = str("RA: "+ra_string+", DEC: "+dec_string+" $|$ cz = "+str(cz)+"$\pm$"+str(cz_err)+" km s$^{-1}$ $|$ SDSS Subclass = "
                    +str(subclass)+"\n PyHammer = "+specTypeMatch+EqW_string+", RV = "+pyhammer_RV+" km s$^{-1}$, EQW = "+this_EqW_str+"\n "
                    +"prop. $|$ Plate = "+plate_string+" MJD = "+mjd_string+" Fiberid = "+fiberid_string+" $|$ GaiaEDR3 Dist = "+str(np.int(np.round(ROW['rpgeo'],2)))
                    +" pc (SNR = "+str(np.round(ROW['parallax']/ROW['parallax_error'],2))+") $|$ GaiaEDR3 PMtot = "+str(np.round(TDSSprop.gaia_pmTOT[prop_id],2))
                    +" mas/yr (SNR = "+str(np.round(TDSSprop.gaia_pmTOT[prop_id]/TDSSprop.gaia_pmTOT_error[prop_id], 2))+")")

    plt_ax.plot(smooth_wavelength, smooth_flux, color='black', linewidth=0.5)
    # plt_ax.plot(temp_lam, temp_flux_scaled, color='red', alpha=0.3, linewidth=0.5)
    plt_ax.plot(smooth_temp_wavelength, smooth_temp_flux, color='red', alpha=0.3, linewidth=0.5)

    plt_ax.set_xlabel("Wavelength [\AA]")  # , fontdict=font)
    plt_ax.set_ylabel("Flux [10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]")  # , fontdict=font)
    plt_ax.set_title(plot_title)
    plt_ax.set_xlim([xmin, xmax])
    plt_ax.set_ylim([ymin, ymax])
    # plot.axvspan(5550, 5604, facecolor=ma.colorAlpha_to_rgb('grey', 0.5)[0])  # , alpha=0.3)
    plt_ax.xaxis.set_major_locator(ticker.MultipleLocator(major_tick_space))
    plt_ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_tick_space))
    for ll in range(lineList_wavelength.size):
        plt_ax.axvline(x=lineList_wavelength[ll], ls='dashed', c='k', alpha=0.1)
        # x_bounds = plt_ax.get_xlim()
        # vlineLabel_value = lineList_wavelength[ll] + 20.0
        # plt_ax.annotate(s=lineList_labels[ll], xy =(((vlineLabel_value-x_bounds[0])/(x_bounds[1]-x_bounds[0])),0.01),
                         #xycoords='axes fraction', verticalalignment='right', horizontalalignment='right bottom' , rotation = 90)
        plt_ax.text(lineList_wavelength[ll]+20.0,plt_ax.get_ylim()[0]+0.50,lineList_labels[ll],rotation=90, color='k', alpha=0.2)
    return this_EqW


def plot_SDSS_photo(ra, dec, image_dir, plt_ax):
    ra_string = '{:0>9.5f}'.format(ra)
    dec_string = '{:0=+9.5f}'.format(dec)

    coord = coords.SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

    impix = 1024
    imsize = 1*u.arcmin
    #SDSS BOSS spec fiber size is 2 arcsec
    fiber_size = 2.0
    scale = impix/imsize.value
    fiber_marker_scale = np.sqrt(scale * fiber_size)
    cutoutbaseurl = 'http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'
    query_string = urlencode(dict(ra=coord.ra.deg,
                                  dec=coord.dec.deg,
                                  width=impix, height=impix,
                                  scale=imsize.to(u.arcsec).value/impix))
    url = cutoutbaseurl + '?' + query_string

    # this downloads the image to your disk
    image_filename = image_dir+ra_string+dec_string+".jpg"
    urlretrieve(url, image_filename)

    img=mpimg.imread(image_filename)
    imgplot = plt_ax.imshow(img)
    plt_ax.xaxis.set_visible(False)
    plt_ax.yaxis.set_visible(False)
    plt_ax.set_xticks([])
    plt_ax.set_yticks([])
    #WCSAxes(plt_ax, wcs=)
    plt_ax.scatter(impix/2.0, impix/2.0, s=fiber_marker_scale, edgecolors='white', marker="+", facecolors='none')


def plot_CMD(TDSSprop, prop_id, plt_ax):
    xi = TDSSprop.xi
    yi = TDSSprop.yi
    zi = TDSSprop.zi


    object_SDSS_gmr = TDSSprop.SDSS_gmr[prop_id]
    object_SDSS_Mr = TDSSprop.SDSS_M_r[prop_id]
    object_SDSS_gmi = TDSSprop.SDSS_gmi[prop_id]
    object_SDSS_Mi = TDSSprop.SDSS_M_i[prop_id]
    object_SDSS_Mi_lo_err = TDSSprop.SDSS_M_i_lo_err[prop_id]
    object_SDSS_Mi_hi_err = TDSSprop.SDSS_M_i_hi_err[prop_id]
    lowerlim_Mi = TDSSprop.lowerLimSDSS_M_i #object_SDSS_Mi
    object_SDSS_Mi_lo_err = np.abs(object_SDSS_Mi - lowerlim_Mi[prop_id])
    object_absM_errs = [[object_SDSS_Mi_lo_err], [object_SDSS_Mi_hi_err]]
    object_color_errs = TDSSprop.SDSS_gmi_err[prop_id]

    object_color = object_SDSS_gmi
    object_color_errs = object_color_errs
    object_absM = object_SDSS_Mi
    object_absM_errs = object_absM_errs
    upperLimDist = TDSSprop.upperLimDist[prop_id]
    lowerLim_M = TDSSprop.lowerLimSDSS_M_i[prop_id]

    if upperLimDist is np.ma.masked:
        upperLimDist = np.nan
    else:
        pass

    if lowerLim_M is np.ma.masked:
        lowerLim_M = np.nan
    else:
        pass

    sdss_zams_prop_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/General/SpecTypeProp/"
    sdss_zams_prop =  np.genfromtxt(sdss_zams_prop_dir+"tab5withMvSDSScolors.dat")

    #M_r = sdss_zams_prop[:,3]  
    M_i = sdss_zams_prop[:,4]  
    g_r = sdss_zams_prop[:,14]
    g_i = g_r + sdss_zams_prop[:,15]
    #M_r_zabms = -0.75 + M_r
    M_i_zabms = -0.75 + M_i

    l = TDSSprop.gaia_l[prop_id]
    b = TDSSprop.gaia_b[prop_id]
    Z = TDSSprop.gaia_Z[prop_id]
    U = TDSSprop.gaia_U[prop_id]
    V = TDSSprop.gaia_V[prop_id]
    W = TDSSprop.gaia_W[prop_id]

    # Gaia_CMD_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/WORKING_DIRECTORY/Vi/Vi_program/sup_data/"
    # Gaia_CMD_data = fits.open(Gaia_CMD_dir+"Gaia_table1b_for_CMD.fits")

    # bp_rp_err_nan =  np.isnan(Gaia_CMD_data[1].data.field('e_bp_min_rp_val'))
    # usable_bp_rp_index = np.where(bp_rp_err_nan == False)[0] 

    # Gaia_G = Gaia_CMD_data[1].data.field('Gmag')
    # Gaia_bp_rp = Gaia_CMD_data[1].data.field('bp_rp')
    # sdss_dr7_wd = fits.open(sdss_zams_prop_dir+"SDSS_DR7_WD_with_Gaia.fits")
    # wd_Gaia_dist = sdss_dr7_wd[1].data.field('rest')
    # wd_g = sdss_dr7_wd[1].data.field('gmag')
    # wd_r = sdss_dr7_wd[1].data.field('rmag')
    # wd_M_r = wd_r + 5.0 -5.0*np.log10(wd_Gaia_dist)
    # wd_gmr = wd_g -  wd_r
    #cmd_data = [gaia_bp_rp[~np.isnan(gaia_bp_rp)],gaia_Mg[~np.isnan(gaia_bp_rp)]]
    #k = kde.gaussian_kde(cmd_data)
    #nbins=20
    #xi, yi = np.mgrid[cmd_data[0].min():cmd_data[0].max():nbins*1j, cmd_data[1].min():cmd_data[1].max():nbins*1j]
    #zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
    plt_ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.viridis)
    plt_ax.set_xlabel("$g - i$")
    plt_ax.set_ylabel("M$_{i}$")

    plt_ax.set_xlim([-1.0,4.5])
    plt_ax.set_ylim([-1.0,15.0])
    plt_ax.invert_yaxis()

    plt_ax.plot(g_i,M_i, color="orange", lw=2.0)
    plt_ax.plot(g_i,M_i_zabms, color="darkred", lw=2.0)

    plt_ax.errorbar(object_color, object_absM, xerr=object_color_errs, yerr=object_absM_errs,uplims=True, lolims=False, color='red', marker="+", markersize= 5, zorder=10)

    upperLimDist
    try:
        title_str = "M$_i$ = {!s} \n g-i = {!s} \n UpperLim Dist = {!s} pc \n LowerLim M$_i$ = {!s} \n l = {!s}$^\circ$ b = {!s}$^\circ$ Z = {!s}pc \n U = {!s} V = {!s} W = {!s} [km/s]".format(np.round(object_absM,2), np.round(object_color,2),np.int(np.round(upperLimDist,2)), np.round(lowerLim_M,2), np.round(l,2), np.round(b,2), np.round(Z,2), np.round(U,2), np.round(V,2), np.round(W,2))
    except ValueError:
        upperLimDist = str(upperLimDist)
        lowerLim_M = str(lowerLim_M)
        title_str = "M$_i$ = {!s} \n g-i = {!s} \n UpperLim Dist = {!s} pc \n LowerLim M$_i$ = {!s} \n l = {!s}$^\circ$ b = {!s}$^\circ$ Z = {!s}pc \n U = {!s} V = {!s} W = {!s} [km/s]".format(np.round(object_absM,2), np.round(object_color,2),upperLimDist, lowerLim_M, np.round(l,2), np.round(b,2), np.round(Z,2), np.round(U,2), np.round(V,2), np.round(W,2))

    plt_ax.set_title(title_str, fontsize=12)


def plot_middle(all_LC_props, best_LC, latestFullVartoolsRun, plt_ax, log10FAP=-10.0):
    xi = latestFullVartoolsRun.xi_2
    yi = latestFullVartoolsRun.yi_2
    zi = latestFullVartoolsRun.zi_2

    LC_prop = all_LC_props[np.where(np.array(['CSS', 'ZTF_g', 'ZTF_r']) == best_LC)[0][0]]

    all_Per_ls = latestFullVartoolsRun.all_Per_ls
    all_logProb_ls = latestFullVartoolsRun.all_logProb_ls
    all_Amp_ls = latestFullVartoolsRun.all_Amp_ls
    all_skewness = latestFullVartoolsRun.all_skewness

    logP, logA, skew, logProb = dealias(np.log10(all_Per_ls), np.log10(all_Amp_ls),
                                        all_skewness, all_logProb_ls)

    is_periodic = LC_prop['logProb'] <= log10FAP
    if is_periodic:
        cm = plt.cm.get_cmap('viridis')
        # log_allPer = np.log10(all_Per_ls[where_periodic])
        # log_allAmp = np.log10(all_Amp_ls[where_periodic])
        # where_notPlot = ((log_allPer >= np.log10(0.5)-sample_around_logP_region) & (log_allPer <= np.log10(0.5)+sample_around_logP_region)) | ((log_allPer >= np.log10(1.0)-sample_around_logP_region) & (log_allPer <= np.log10(1.0)+sample_around_logP_region))
        # this_all_skewness = all_skewness[where_periodic]
        # sc = plt_ax.scatter(log_allPer[~where_notPlot], log_allAmp[~where_notPlot], s=2.5, c=this_all_skewness[~where_notPlot], cmap=cm, vmin=-1, vmax=1)
        sc = plt_ax.scatter(logP, logA, s=2.5, c=skew, cmap=cm, vmin=-1, vmax=1)
        divider1 = make_axes_locatable(plt_ax)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(sc, cax=cax1)
        cbar1.ax.get_yaxis().labelpad = 0
        cbar1.ax.set_ylabel('Skewness', rotation=270)

        single_point_color = cbar1.mappable.to_rgba(LC_prop['lc_skew'])
        plt_ax.scatter(np.log10(LC_prop['P']), np.log10(LC_prop['Amp']), s=200.0, marker="X", color=single_point_color, edgecolors='red')

        plt_ax.set_xlabel("log$_{10}$(P / d)")
        plt_ax.set_ylabel("log$_{10}$(A / mag)")
        plt_ax.set_xlim([-1.5, 0.77])
        plt_ax.set_ylim([-1.05, 0.3])
        title_str = "log$_{10}$(P / day) = "+str(np.round(np.log10(LC_prop['P']),2))+"\n log$_{10}$(Amp / mag) = "+str(np.round(np.log10(LC_prop['Amp']),2))+"\n Skewness = "+str(np.round(LC_prop['lc_skew'],2))
        plt_ax.set_title(title_str, fontsize=12)
    else:
        plt_ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.viridis)
        plt_ax.set_xlabel("log$_{10}$($\chi^2$)")
        plt_ax.set_ylabel("a95")
        plt_ax.set_xlim([-0.5, 2.0])
        plt_ax.set_ylim([0.0, 2.0])
        plt_ax.scatter(np.log10(LC_prop['Chi2']), LC_prop['a95'], s=100.0, marker="X", color='red')
        title_str = "$\chi^2$ = " + str(np.round(LC_prop['Chi2'], 2)) + "\n log$_{10}$($\chi^2$) = " + str(np.round(np.log10(LC_prop['Chi2']), 2)) + "\n a95 = " + str(np.round(LC_prop['a95'], 2))
        plt_ax.set_title(title_str, fontsize=12)


def dealias(logP, logA, skew, logProb, sample_around_logP_region=0.05, log10FAP=-10):
    where_notPlot = ((logP >= np.log10(0.5) - sample_around_logP_region) & (logP <= np.log10(0.5) + sample_around_logP_region)) | ((logP >= np.log10(1.0) - sample_around_logP_region) & (logP <= np.log10(1.0) + sample_around_logP_region))
    logP = logP[~where_notPlot]
    logA = logA[~where_notPlot]
    skew = skew[~where_notPlot]
    logProb = logProb[~where_notPlot]
    # specType = specType[~where_notPlot]

    where_periodic = np.where(logProb <= log10FAP)[0]
    logP = logP[where_periodic]
    logA = logA[where_periodic]
    skew = skew[where_periodic]
    logProb = logProb[where_periodic]
    # specType = specType[where_periodic]

    # specType = periodic_prop['PyHammerSpecType']
    # specType = np.array([ii.strip().decode() for ii in specType.data.data])

    # maintype = np.array([ii[0] for ii in specType])
    # maintypeNum = np.array([np.where(letterSpt == ii)[0][0] for ii in maintype])

    cut_index = (skew < -0.4) & (logP < -0.5)
    # cut_spec_type = specType[cut_index]
    # cut_maintype = np.array([ii[0] for ii in cut_spec_type])
    # cut_maintypeNum = np.array([np.where(letterSpt == ii)[0][0] for ii in cut_maintype])
    logP[cut_index] = logP[cut_index] + np.log10(2)  # P * 2

    cut_index = (skew > 0.4) & (logP > -1.0) & (logP < 0.0)
    # cut_spec_type = specType[cut_index]
    # cut_maintype = np.array([ii[0] for ii in cut_spec_type])
    # cut_maintypeNum = np.array([np.where(letterSpt == ii)[0][0] for ii in cut_maintype])
    logP[cut_index] = logP[cut_index] + np.log10(2)  # P * 2

    cut_index = (skew < -0.2) & (logP < -0.75)
    # cut_spec_type = specType[cut_index]
    # cut_maintype = np.array([ii[0] for ii in cut_spec_type])
    # cut_maintypeNum = np.array([np.where(letterSpt == ii)[0][0] for ii in cut_maintype])
    logP[cut_index] = logP[cut_index] - np.log10(2)  # P * 0.5

    return logP, logA, skew, logProb


def eqw(temp_lam, temp_flux_scaled, wavelength, flux):
    try:
        region1 = np.where( (temp_lam >= 6507) & (temp_lam <= 6543))[0]
        region2 = np.where( (temp_lam >= 6583) & (temp_lam <= 6631))[0]
        line_region  = np.where( (temp_lam >= 6543) & (temp_lam <= 6583))[0]

        region1_wave_avg = np.nanmean(temp_lam[region1])
        region2_wave_avg = np.nanmean(temp_lam[region2])

        region1_flux_avg = np.nanmean(temp_flux_scaled[region1])
        region2_flux_avg = np.nanmean(temp_flux_scaled[region2])

        p = np.polyfit([region1_wave_avg, region2_wave_avg], [region1_flux_avg, region2_flux_avg], deg=1)

        interp_wave_range = np.linspace(start=6507, stop=6631)
        interp_flux =  p[0]*interp_wave_range + p[1]


        object_halpha_range = np.where( (wavelength >= 6507) & (wavelength <= 6631))[0]
        object_region1 = np.where( (wavelength[object_halpha_range] >= 6507) & (wavelength[object_halpha_range] <= 6543))[0]
        object_region2 = np.where( (wavelength[object_halpha_range] >= 6583) & (wavelength[object_halpha_range] <= 6631))[0]
        object_line_region = np.where( (wavelength[object_halpha_range] >= 6543) & (wavelength[object_halpha_range] <= 6583))[0]

        temp_interped_flux = np.interp(object_halpha_range, interp_wave_range, interp_flux)

        EQW_perpix = (temp_interped_flux - flux[object_halpha_range]) / temp_interped_flux
        EQW = EQW_perpix.sum()

        noise = np.sqrt((EQW_perpix[object_region1]**2).sum() + (EQW_perpix[object_region2]**2).sum()) 
        signal = EQW_perpix[object_line_region].sum()

        SNR = signal / noise

        if np.abs(SNR) > 3.0:
            return EQW
        else:
            return np.nan
    except:
        return np.nan


def removeSdssStitchSpike(wavelength, flux):
    """
    All SDSS spectrum have a spike in the spectra between 5569 and 5588 angstroms where
    the two detectors meet. This method will remove the spike at that point by linearly
    interpolating across that gap.
    """
    # Make a copy so as to not alter the original, passed in flux
    flux = flux.copy()
    # Search for the indices of the bounding wavelengths on the spike. Use the
    # fact that the wavelength is an array in ascending order to search quickly
    # via the searchsorted method.
    lower = np.searchsorted(wavelength, 5569)
    upper = np.searchsorted(wavelength, 5588)
    # Define the flux in the stitch region to be linearly interpolated values between
    # the lower and upper bounds of the region.
    flux[lower:upper] = np.interp(wavelength[lower:upper],
                                  [wavelength[lower],wavelength[upper]],
                                  [flux[lower],flux[upper]])
    return flux


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


def splitSpecType(s):
    # head = s.rstrip('0123456789')
    # tail = s[len(head):]
    if 'dC' in s:
        head = 'dC'
        tail = s[-1]
    else:
        head, tail, _ = re.split('(\d.*)', s)
    return head, tail


def makeViDirs(Vi_dir="/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/WORKING_DIRECTORY/Vi/", ZTF_filters=['g', 'r']):
    datestr = check_output(["/bin/date","+%F"])
    datestr = datestr.decode().replace('\n', '')
    if not os.path.exists(Vi_dir+datestr):
        os.mkdir(Vi_dir+datestr)

    lc_plt_dir = Vi_dir+datestr+"/LC_plots/"
    Vi_plots_dir = Vi_dir+datestr+"/Vi_plots/"

    if not os.path.exists(lc_plt_dir):
        os.mkdir(lc_plt_dir)
    if not os.path.exists(Vi_plots_dir):
        os.mkdir(Vi_plots_dir)

    CSS_LC_plot_dir = lc_plt_dir+"CSS/"
    if not os.path.exists(CSS_LC_plot_dir):
        os.mkdir(CSS_LC_plot_dir)
    ZTF_LC_plot_dir = lc_plt_dir+"ZTF/"
    if not os.path.exists(ZTF_LC_plot_dir):
        os.mkdir(ZTF_LC_plot_dir)

    for ZTF_filter in ZTF_filters:
        filter_lc_dir = ZTF_LC_plot_dir+ZTF_filter+"/"
        if not os.path.exists(filter_lc_dir):
            os.mkdir(filter_lc_dir)

    prop_out_dir = Vi_dir+datestr+"/"
    return prop_out_dir, CSS_LC_plot_dir, ZTF_LC_plot_dir, Vi_plots_dir, datestr


def checkViRun(Vi_dir="/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/WORKING_DIRECTORY/Vi/"):
    from pathlib import Path
    datestr = check_output(["/bin/date","+%F"])
    datestr = datestr.decode().replace('\n', '')
    prop_out_dir = Vi_dir + datestr + "/"
    my_file = Path(prop_out_dir+"completed_Vi_prop_"+datestr+".fits")
    if my_file.is_file():
        properties = Table.read(prop_out_dir+"completed_Vi_prop_"+datestr+".fits")
        prop_id_last = np.where(properties['ViCompleted']==0.0)[0][0]
        return True, prop_id_last, properties
    else:
        return False, 0, None


class TDSSprop:
    Vi_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/WORKING_DIRECTORY/Vi/"

    def __init__(self, nbins):
        from scipy.stats import kde
        import numpy as np
        from astropy import units as u
        from astropy.io import fits
        from astropy import coordinates as coords
        from astropy.table import Table
        self.nbins = nbins
        TDSS_prop = Table.read("/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/PROGRAM_SAMPLE/2021-07-27/FINAL_FILES/TDSS_SES+PREV_DR16DR12griLT20_GaiaEDR3_Drake2014PerVar_CSSID_ZTFIDs_LCpointer_PyHammer.fits")
        TDSS_prop['CSS_Nepochs'] = TDSS_prop['CSS_Nepochs'].filled(np.nan)
        TDSS_prop['ZTF_g_Nepochs'] = TDSS_prop['ZTF_g_Nepochs'].filled(np.nan)
        TDSS_prop['ZTF_r_Nepochs'] = TDSS_prop['ZTF_r_Nepochs'].filled(np.nan)
        self.data = TDSS_prop
        self.TDSS_cssid = TDSS_prop['CSSID'].astype(int)
        self.gaia_bp_rp = TDSS_prop['bp_rp']
        self.gaia_g = TDSS_prop['phot_g_mean_mag']
        self.gaia_dist = TDSS_prop['rpgeo']
        self.gaia_dist_lo = TDSS_prop['b_rpgeo_GaiaEDR3']
        self.gaia_dist_hi = TDSS_prop['B_rpgeo_GaiaEDR3a']
        self.gaia_parallax = TDSS_prop['parallax']
        self.gaia_parallax_error = TDSS_prop['parallax_error']
        self.gaia_pmra = TDSS_prop['pmra_GaiaEDR3']
        self.gaia_pmra_error = TDSS_prop['pmra_error']
        self.gaia_pmdec = TDSS_prop['pmdec']
        self.gaia_pmdec_error = TDSS_prop['pmdec_error']
        self.gaia_pmTOT = np.sqrt(self.gaia_pmra**2 + self.gaia_pmdec**2)
        self.gaia_pmTOT_error = np.sqrt((self.gaia_pmra*self.gaia_pmra_error)**2 + (self.gaia_pmdec*self.gaia_pmdec_error)**2) / self.gaia_pmTOT
        self.gaia_Mg = self.gaia_g + 5.0 - 5.0*np.log10(self.gaia_dist)
        self.gaia_l = TDSS_prop['l_GaiaEDR3']
        self.gaia_b = TDSS_prop['b_GaiaEDR3']
        self.gaia_Z = TDSS_prop['Z_GaiaEDR3']
        self.gaia_U = TDSS_prop['U_GaiaEDR3']
        self.gaia_V = TDSS_prop['V_GaiaEDR3']
        self.gaia_W = TDSS_prop['W_GaiaEDR3']
        self.SDSS_g = TDSS_prop['gmag']
        self.SDSS_g_err = TDSS_prop['e_gmag']
        self.SDSS_r = TDSS_prop['rmag']
        self.SDSS_r_err = TDSS_prop['e_rmag']
        self.SDSS_i = TDSS_prop['imag']
        self.SDSS_i_err = TDSS_prop['e_imag']
        self.SDSS_gmr = self.SDSS_g - self.SDSS_r
        self.SDSS_gmi = self.SDSS_g - self.SDSS_i
        self.SDSS_M_r = self.SDSS_r + 5.0 - 5.0*np.log10(self.gaia_dist)
        self.SDSS_M_r_lo = self.SDSS_r + 5.0 - 5.0*np.log10(self.gaia_dist_lo)
        self.SDSS_M_r_hi = self.SDSS_r + 5.0 - 5.0*np.log10(self.gaia_dist_hi)
        self.SDSS_M_i = self.SDSS_i + 5.0 - 5.0*np.log10(self.gaia_dist)
        self.SDSS_M_i_lo = self.SDSS_i + 5.0 - 5.0*np.log10(self.gaia_dist_lo) #lo means closser
        self.SDSS_M_i_hi = self.SDSS_i + 5.0 - 5.0*np.log10(self.gaia_dist_hi) #hi mean further
        self.SDSS_M_i_lo_err = np.abs(self.SDSS_M_i - self.SDSS_M_i_lo)
        self.SDSS_M_i_hi_err = np.abs(self.SDSS_M_i - self.SDSS_M_i_hi)
        self.SDSS_gmi_err = np.sqrt(self.SDSS_g_err**2 + self.SDSS_i_err**2)
        self.gaia_cmd_data = [self.gaia_bp_rp[~np.isnan(self.gaia_bp_rp)],self.gaia_Mg[~np.isnan(self.gaia_bp_rp)]]
        self.SDSS_gmr_cmd_data = [self.SDSS_gmr[~np.isnan(self.SDSS_gmr)],self.SDSS_M_r[~np.isnan(self.SDSS_gmr)]]
        self.SDSS_gmi_cmd_data = [self.SDSS_gmi[~np.isnan(self.SDSS_gmi)],self.SDSS_M_i[~np.isnan(self.SDSS_gmi)]]
        self.cmd_data = self.SDSS_gmi_cmd_data
        self.k = kde.gaussian_kde(self.cmd_data)
        self.xi, self.yi = np.mgrid[-1:4.5:self.nbins*1j, -1.0:16.5:self.nbins*1j]
        self.zi = self.k(np.vstack([self.xi.flatten(), self.yi.flatten()]))
        self.zi = np.sqrt(self.zi)
        self.TDSS_ra = TDSS_prop['ra_GaiaEDR3']
        self.TDSS_dec = TDSS_prop['dec_GaiaEDR3']
        self.TDSS_plate = TDSS_prop['plate'].astype(int)
        self.TDSS_mjd = TDSS_prop['mjd'].astype(int)
        self.TDSS_fibderid = TDSS_prop['fiber'].astype(int)
        self.TDSS_coords = coords.SkyCoord(ra=self.TDSS_ra, dec=self.TDSS_dec, frame='icrs')       
        self.Drake_index = np.where(TDSS_prop['Drake_Per'])[0]
        # self.Drake_num_to_vartype = np.genfromtxt("sup_data/"+"darke_var_types.txt", dtype="U", comments="#", delimiter=",")
        self.Drake_num_to_vartype = Table.read("sup_data/"+"darke_var_types.txt", format='ascii.commented_header')
        self.D_Per = TDSS_prop['Drake_Per']
        self.D_Amp = TDSS_prop['Drake_Vamp']
        self.vartype_num = TDSS_prop['Drake_Cl']
        self.pyhammer_RV = TDSS_prop['PyHammerRV']
        # self.pyhammerChanged = TDSS_prop['PyHammerDiff']
        self.upperLimDist = np.sqrt(600.0**2 - self.pyhammer_RV **2) / (4.74e-3*self.gaia_pmTOT )
        self.lowerLimSDSS_M_i = self.SDSS_i + 5.0 - 5.0*np.log10(self.upperLimDist)
        self.lowerLim_gaia_Mg = self.gaia_g + 5.0 - 5.0*np.log10(self.upperLimDist)
        self.lowerLimSDSS_M_r = self.SDSS_r + 5.0 - 5.0*np.log10(self.upperLimDist)


class latestFullVartoolsRun:
    def __init__(self, latestFullVartoolsRun_filename, nbins=50):
        self.latestFullVartoolsRun_filename = latestFullVartoolsRun_filename
        self.nbins = nbins
        self.latestFullVartoolsRun = pd.read_csv(self.latestFullVartoolsRun_filename)
        self.dataFrame_all_Index = np.where(self.latestFullVartoolsRun['ZTF_r_lc_id'].values != 0.0)[0]
        self.lc_id = self.latestFullVartoolsRun['ZTF_r_lc_id'].values[self.dataFrame_all_Index]
        self.all_Per_ls = self.latestFullVartoolsRun['ZTF_r_P'].values[self.dataFrame_all_Index]
        self.all_logProb_ls = self.latestFullVartoolsRun['ZTF_r_logProb'].values[self.dataFrame_all_Index]
        self.all_Amp_ls = self.latestFullVartoolsRun['ZTF_r_Amp'].values[self.dataFrame_all_Index]
        self.all_a95 = self.latestFullVartoolsRun['ZTF_r_a95'].values[self.dataFrame_all_Index]
        self.all_ChiSq = self.latestFullVartoolsRun['ZTF_r_Chi2'].values[self.dataFrame_all_Index]
        self.all_skewness = self.latestFullVartoolsRun['ZTF_r_lc_skew'].values[self.dataFrame_all_Index]
        self.Mt = self.latestFullVartoolsRun['ZTF_r_Mt'].values[self.dataFrame_all_Index]
        self.times_is_alias = self.latestFullVartoolsRun['ZTF_r_isAlias'].values[self.dataFrame_all_Index]
        self.where_periodic = np.where(self.all_logProb_ls <= -15.0)[0]
        self.where_not_periodic = np.where(self.all_logProb_ls > -15.0)[0]
        self.log_all_ChiSq = np.log(self.all_ChiSq[self.where_not_periodic])
        self.all_a95_nonan = self.all_a95[self.where_not_periodic]
        self.log_all_ChiSq = self.log_all_ChiSq[~np.isnan(self.log_all_ChiSq)]
        self.all_a95_nonan = self.all_a95_nonan[~np.isnan(self.log_all_ChiSq)]
        self.all_a95_nonan = self.all_a95_nonan[np.isfinite(self.log_all_ChiSq)]
        self.log_all_ChiSq = self.log_all_ChiSq[np.isfinite(self.log_all_ChiSq)]
        self.a95_chi_data = [self.log_all_ChiSq, self.all_a95_nonan]
        self.k2 = kde.gaussian_kde(self.a95_chi_data)
        self.xi_2, self.yi_2 = np.mgrid[-0.5:2.5:self.nbins*1j, 0.0:3.0:self.nbins*1j]
        self.zi_2 = self.k2(np.vstack([self.xi_2.flatten(), self.yi_2.flatten()]))
        self.zi_2 = np.sqrt(self.zi_2)


def get_Vi_panels(ra, dec, runDate, copy=False, moveDir='/Users/benjaminroulston/Desktop/'):
    Vi_plots_dir = f'/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/WORKING_DIRECTORY/Vi/{runDate}/Vi_plots/'
    all_filenames = []
    for ii in tqdm.tqdm(range(ra.size)):
        object_ra = ra[ii]
        object_dec = dec[ii]
        ra_string = '{:0>9.5f}'.format(object_ra)
        dec_string = '{:0=+9.5f}'.format(object_dec)

        this_Viplot_filename = f"{Vi_plots_dir}{ra_string}{dec_string}_Vi.pdf"
        all_filenames.append(this_Viplot_filename)

        if copy:
            new_Viplot_filename = f"{moveDir}{ra_string}{dec_string}_Vi.pdf"
            cp_cmd = os.system(f'cp {this_Viplot_filename} {new_Viplot_filename}')

    all_filenames = np.array(all_filenames)
    return all_filenames
