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
from subprocess import *
import os
import glob
import re

from astropy.table import Table
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from astropy import coordinates as coords

import mimic_alpha as ma

def plot_CSS_LC_noDrake(css_id, LC_OutDir, vartools_command, vartools_command_whitten, vartools_command_whitten2, vt_outdir, main_lc_data_files_path, plt_ax, runVartools=True, latestFullVartoolsRun=None):
    vt_dir = '/usr/local/bin/'
    col_names = ['MJD', 'mag', 'mag_err']
    #css_id = the full path AND file name to the raw LC file from CSS
    lc_data_pre_check = pd.read_csv(css_id, delim_whitespace = True, names = col_names)
    lc_data = lc_data_pre_check.dropna(subset = col_names)
    lc_data.iterrows()
    lc_id = np.int(css_id.rstrip(".dat").lstrip(main_lc_data_files_path))
    lc_mag = lc_data['mag']
    lc_mjd = lc_data['MJD']
    lc_err = lc_data['mag_err']
    nmag = len(lc_mag)
    errmn = np.mean(lc_err)
    brt10per = np.percentile(lc_mag,10)
    fnt10per = np.percentile(lc_mag,90)       
    brt10data = lc_data[ lc_mag < brt10per ]
    fnt10data = lc_data[ lc_mag > fnt10per ]
    fntmags = fnt10data['mag']
    fnterr = fnt10data['mag_err']
    medmagfnt10 = np.median(fntmags)
    mederrfnt10 = np.median(fnterr)
    brtmags = brt10data['mag']
    brterr = brt10data['mag_err']
    medmagbrt10 = np.median(brtmags)
    mederrbrt10 = np.median(brterr)
    brtcutoff=medmagbrt10-(2*mederrbrt10)
    fntcutoff=medmagfnt10+(2*mederrfnt10)
    filter_data = lc_data[ (lc_mag >= brtcutoff) & (lc_mjd >= 0.0) ]
    flc_data = filter_data[ filter_data['mag'] <= fntcutoff ] # same columns as lc_data
    flc_mag = flc_data['mag']
    flc_mjd = flc_data['MJD']
    flc_err = flc_data['mag_err']
    nfmag = len(flc_mag)
    fmagmed = np.median(flc_mag)
    fmagmn = np.mean(flc_mag)
    fmagmax = np.max(flc_mag)
    fmagmin = np.min(flc_mag)
    ferrmed = np.median(flc_err)
    ferrmn = np.mean(flc_err)
    fmag_stdev = np.std(flc_mag)
    rejects = nmag - nfmag 
    mag_above = np.mean(lc_mag)-3*ferrmn
    mag_below = np.mean(lc_mag)+3*ferrmn
    nabove = np.where(lc_mag <= mag_above)[0].size
    nbelow = np.where(lc_mag >= mag_below)[0].size
    Mt = (fmagmax - fmagmed)/(fmagmax - fmagmin)
    minpercent = np.percentile(flc_mag,5) # finds the 5th percentile mag value
    maxpercent = np.percentile(flc_mag,95) # finds the 95th percentile mag value
    a95 = maxpercent - minpercent
    lc_skew = flc_mag.skew()
    summation_eqn1 = np.sum( ((flc_mag - fmagmn)**2)/(flc_err**2) )
    Chi2 = (1./(nfmag-1))*summation_eqn1
    lc_file=LC_OutDir+str(lc_id)+'.lc'
    z = flc_data[['MJD', 'mag', 'mag_err']].copy()
    np.savetxt(lc_file, z.values, fmt="%12.5f %8.3f %8.3f ")#*******!
    if runVartools:
        vt_callLS = vt_dir+'vartools -i '+lc_file+vartools_command#*******!
        vt_result = check_output(vt_callLS, shell=True)#*******!
        Per_ls = ('%6.3f' % float(vt_result.split()[1]))
        logProb_ls = ('%7.3f' % float(vt_result.split()[2]))
        Amp_ls = ('%6.3f' % float(vt_result.split()[-1]))
        Per_ls_num = float(vt_result.split()[1])
        log10_P = np.log10(Per_ls_num)
        sample_around_logP_region = 0.05
        is_alias = ((log10_P >= np.log10(0.5)-sample_around_logP_region) & (log10_P <= np.log10(0.5)+sample_around_logP_region)) or ((log10_P >= np.log10(1.0)-sample_around_logP_region) & (log10_P <= np.log10(1.0)+sample_around_logP_region))
        if is_alias:
            vt_callLS = vt_dir+'vartools -i '+lc_file+vartools_command_whitten#*******!
            vt_result = check_output(vt_callLS, shell=True)#*******!
            Per_ls = ('%6.3f' % float(vt_result.split()[5]))
            logProb_ls = ('%7.3f' % float(vt_result.split()[6]))
            Amp_ls = ('%6.3f' % float(vt_result.split()[-1]))
        Per_ls_num = float(vt_result.split()[5])
        log10_P = np.log10(Per_ls_num)
        is_alias2 = ((log10_P >= np.log10(0.5)-sample_around_logP_region) & (log10_P <= np.log10(0.5)+sample_around_logP_region)) or ((log10_P >= np.log10(1.0)-sample_around_logP_region) & (log10_P <= np.log10(1.0)+sample_around_logP_region))
        if is_alias2:
            vt_callLS = vt_dir+'vartools -i '+lc_file+vartools_command_whitten2#*******!
            vt_result = check_output(vt_callLS, shell=True)#*******!
            Per_ls = ('%6.3f' % float(vt_result.split()[9]))
            logProb_ls = ('%7.3f' % float(vt_result.split()[10]))
            Amp_ls = ('%6.3f' % float(vt_result.split()[-1]))
        #print('\n CURRENTLY RUNNING VARTOOLS ON: '+str(lc_id))#*******!
        #print('--'*30)#*******!
        #print('[COMMAND]--',vt_callLS)#*******!
        #print('[RESULT]--',vt_result)#*******!
    else:
        dataFrameIndex = np.where(latestFullVartoolsRun.lc_id == lc_id)[0][0]
        #latestFullVartoolsRun_index = np.where(latestFullVartoolsRun[' dec'].values == 0.0)[0][0]
        Per_ls = latestFullVartoolsRun.all_Per_ls[dataFrameIndex]
        logProb_ls = latestFullVartoolsRun.all_logProb_ls[dataFrameIndex]
        Amp_ls = latestFullVartoolsRun.all_Amp_ls[dataFrameIndex]
        #all_a95 = latestFullVartoolsRun[' a95'].values[dataFrameIndex]
        #all_ChiSq = latestFullVartoolsRun[' Chi2'].values[dataFrameIndex]
        #all_skewness = latestFullVartoolsRun[' lc_skew'].values[dataFrameIndex]

    if np.float64(logProb_ls) < -10.0:
        # ---------------------------
        # Plot Phased LC from -LS
        # ---------------------------
        # read in the file
        phLCdata = vt_outdir+str(lc_id)+'.lc.killharm.model'
        phLC_colnames = ['MJD', 'Mag', 'PredMag', 'Magerr']
        pLC = pd.read_csv(phLCdata, sep=' ', index_col=False, header=None,  names=phLC_colnames)
        # assign lists to columns
        plc_mjd= pLC['MJD']
        nxt_mjd = [ ph+1 for ph in plc_mjd ] # add 1 to the phase to be able to plot phase from 0 to 2
        plc_mag = pLC['Mag']
        plc_predmag = pLC['PredMag']
        plc_err = pLC['Magerr']
        # plot the phased light curve
        #fig_size = (8,6)
        #plt_ax.figure(figsize=fig_size)  
        #plt_ax.clf()
        plt_ax.plot(plc_mjd,plc_mag, 'k.',ms=6, label='Phased LC') # phase 0 to 1
        plt_ax.plot(nxt_mjd,plc_mag, 'k.',ms=6) # phase 1 to 2
        # plot the predicted light curve with green open diamonds
        plt_ax.plot(plc_mjd,plc_predmag,'D',ms=8,mfc='None',mec='green',mew=0.2, label='Predicted LC')  # green open Diamonds
        plt_ax.plot(nxt_mjd,plc_predmag,'D',ms=8,mfc='None',mec='green',mew=0.2)  # green open Diamonds (for phase 1 to 2)
        plt_ax.axhline(fmagmn, color='r', ls='-', lw=2, label='Mean Mag')
        plt_ax.axhline(fmagmn+3*ferrmn, color='g', ls='-.', lw=2 ,alpha=0.5, label='3X Mag Err')
        plt_ax.axhline(fmagmn-3*ferrmn, color='g', ls='-.', lw=2 ,alpha=0.5)
        plt_ax.axhline(fmagmn+3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5, label='3X Mag StDev')
        plt_ax.axhline(fmagmn-3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5)
        if is_alias & is_alias2:
            title_line1 = 'CSS ID: {!s} | P={!s} \n logProb={!s} | Amp={!s} \n ngood={!s} | nreject={!s} \n nabove={!s} ({!s}%) | nbelow={!s} ({!s}%) \n'.format(lc_id, Per_ls+"ww", logProb_ls, Amp_ls, nfmag, rejects, nabove, np.round((nabove/nmag)*100,2), nbelow, np.round((nbelow/nmag)*100,2))
        elif is_alias:
            title_line1 = 'CSS ID: {!s} | P={!s} \n logProb={!s} | Amp={!s} \n ngood={!s} | nreject={!s} \n nabove={!s} ({!s}%) | nbelow={!s} ({!s}%) \n'.format(lc_id, Per_ls+"w", logProb_ls, Amp_ls, nfmag, rejects, nabove, np.round((nabove/nmag)*100,2), nbelow, np.round((nbelow/nmag)*100,2))
        else:
            title_line1 = 'CSS ID: {!s} | P={!s} \n logProb={!s} | Amp={!s} \n ngood={!s} | nreject={!s} \n nabove={!s} ({!s}%) | nbelow={!s} ({!s}%) \n'.format(lc_id, Per_ls, logProb_ls, Amp_ls, nfmag, rejects, nabove, np.round((nabove/nmag)*100,2), nbelow, np.round((nbelow/nmag)*100,2))
        #title_line2 = 'Drake: P={!s} | Amp={!s} | VarType={!s} | Subclass={!s}'.format(D_Per, D_Amp, D_Vartype, D_sub)
        title_str = title_line1 #+title_line2
        plt_ax.set_title(title_str, fontsize=12)
        plt_ax.set_xlabel('Phase')
        plt_ax.set_ylabel('mag')
        plt_ax.invert_yaxis() # flip the y-axis so fainter mags are on bottom
        # save plot
        #plotname = plt_dir+lc_id+'_plc'+'.eps'
        #plt_ax.savefig(plotname,dpi=600,bbox_inches='tight')
        #plt_ax.clf()
        #plt_ax.close()
    else:
        # ------------------
        # Plot Raw LC:
        # ------------------
        #plot(lc_mjd,lc_mag, 'k.',ms=6)  use the filtered data below
        plt_ax.plot(flc_mjd, flc_mag, 'k.', ms=6, label='Raw Data') 
        # draw a y=constant line with plt.axhline(y, xmin, xmax) 
        # plot mean mag. as solid line
        plt_ax.axhline(fmagmn, color='r', ls='-', lw=2, label='Mag. Mean')
        # plot 3X errmn as dashed line
        plt_ax.axhline(fmagmn+3*ferrmn, color='g', ls='--', lw=2 ,alpha=0.5, label='3x Mean Mag. Err')
        plt_ax.axhline(fmagmn-3*ferrmn, color='g', ls='--', lw=2 ,alpha=0.5)
        # plot 3X stdev as dotted line
        plt_ax.axhline(fmagmn+3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5, label='3x StDev Mag.')
        plt_ax.axhline(fmagmn-3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5)
        # set plot labels
        #title_str = f'{lc_id} \n mean mag. = {fmagmn:0.2f}'
        if is_alias:
            title_line1 = 'CSS ID: {!s} | P={!s} \n logProb={!s} | Amp={!s} \n ngood={!s} | nreject={!s} \n nabove={!s} ({!s}%) | nbelow={!s} ({!s}%) \n'.format(lc_id, Per_ls+"w", logProb_ls, Amp_ls, nfmag, rejects, nabove, np.round((nabove/nmag)*100,2), nbelow, np.round((nbelow/nmag)*100,2))
        else:
            title_line1 = 'CSS ID: {!s} | P={!s} \n logProb={!s} | Amp={!s}  \n ngood={!s} | nreject={!s} \n nabove={!s} ({!s}%) | nbelow={!s} ({!s}%) \n'.format(lc_id, Per_ls, logProb_ls, Amp_ls, nfmag, rejects, nabove, np.round((nabove/nmag)*100,2), nbelow, np.round((nbelow/nmag)*100,2))
        plt_ax.set_title(title_line1, fontsize=12)
        plt_ax.set_xlabel('MJD')
        plt_ax.set_ylabel('Mag')
        #plt_ax.legend(loc='upper right', fontsize=8)
        plt_ax.invert_yaxis() # flip the y-axis so fainter mags are on bottom

    prop_header = "lc_id, Per_ls, logProb_ls, Amp_ls, Mt, a95, lc_skew, Chi2, brtcutoff, brt10per, fnt10per, fntcutoff, errmn, ferrmn, ngood, nrejects, nabove, nbelow"
    if runVartools:
        properties = np.array([lc_id, Per_ls, logProb_ls, Amp_ls, Mt, a95, lc_skew, Chi2, 
                               brtcutoff, brt10per, fnt10per, fntcutoff, errmn, ferrmn, nfmag, rejects, nabove, nbelow])
    else:
        dataFrameIndex = np.where(latestFullVartoolsRun[' lc_id'].values == lc_id)[0][0]
        properties = latestFullVartoolsRun.latestFullVartoolsRun.values[dataFrameIndex,2:]
        #properties = latestFullVartoolsRun.values[dataFrameIndex, 2:]
    return properties

def plot_CSS_LC_Drake(css_id, LC_OutDir, vartools_command, vartools_command_whitten, vartools_command_whitten2, vt_outdir, main_lc_data_files_path, D_Per, D_Amp, D_Vartype, plt_ax, runVartools=True, latestFullVartoolsRun=None):
    vt_dir = '/usr/local/bin/'
    col_names = ['MJD', 'mag', 'mag_err']
    #css_id = the full path AND file name to the raw LC file from CSS
    lc_data_pre_check = pd.read_csv(css_id, delim_whitespace = True, names = col_names)
    lc_data = lc_data_pre_check.dropna(subset = col_names)
    lc_data.iterrows()
    lc_id = np.int(css_id.rstrip(".dat").lstrip(main_lc_data_files_path))
    lc_mag = lc_data['mag']
    lc_mjd = lc_data['MJD']
    lc_err = lc_data['mag_err']
    nmag = len(lc_mag)
    errmn = np.mean(lc_err)
    brt10per = np.percentile(lc_mag,10)
    fnt10per = np.percentile(lc_mag,90)       
    brt10data = lc_data[ lc_mag < brt10per ]
    fnt10data = lc_data[ lc_mag > fnt10per ]
    fntmags = fnt10data['mag']
    fnterr = fnt10data['mag_err']
    medmagfnt10 = np.median(fntmags)
    mederrfnt10 = np.median(fnterr)
    brtmags = brt10data['mag']
    brterr = brt10data['mag_err']
    medmagbrt10 = np.median(brtmags)
    mederrbrt10 = np.median(brterr)
    brtcutoff=medmagbrt10-(2*mederrbrt10)
    fntcutoff=medmagfnt10+(2*mederrfnt10)
    filter_data = lc_data[ (lc_mag >= brtcutoff) & (lc_mjd >= 0.0) ]
    flc_data = filter_data[ filter_data['mag'] <= fntcutoff ] # same columns as lc_data
    flc_mag = flc_data['mag']
    flc_mjd = flc_data['MJD']
    flc_err = flc_data['mag_err']
    nfmag = len(flc_mag)
    fmagmed = np.median(flc_mag)
    fmagmn = np.mean(flc_mag)
    fmagmax = np.max(flc_mag)
    fmagmin = np.min(flc_mag)
    ferrmed = np.median(flc_err)
    ferrmn = np.mean(flc_err)
    fmag_stdev = np.std(flc_mag)
    rejects = nmag - nfmag 
    mag_above = np.mean(lc_mag)-3*ferrmn
    mag_below = np.mean(lc_mag)+3*ferrmn
    nabove = np.where(lc_mag <= mag_above)[0].size
    nbelow = np.where(lc_mag >= mag_below)[0].size
    Mt = (fmagmax - fmagmed)/(fmagmax - fmagmin)
    minpercent = np.percentile(flc_mag,5) # finds the 5th percentile mag value
    maxpercent = np.percentile(flc_mag,95) # finds the 95th percentile mag value
    a95 = maxpercent - minpercent
    lc_skew = flc_mag.skew()
    summation_eqn1 = np.sum( ((flc_mag - fmagmn)**2)/(flc_err**2) )
    Chi2 = (1./(nfmag-1))*summation_eqn1
    lc_file=LC_OutDir+str(lc_id)+'.lc'
    z = flc_data[['MJD', 'mag', 'mag_err']].copy()
    np.savetxt(lc_file, z.values, fmt="%12.5f %8.3f %8.3f ")#*******!
    if runVartools:
        vt_callLS = vt_dir+'vartools -i '+lc_file+vartools_command#*******!
        vt_result = check_output(vt_callLS, shell=True)#*******!
        Per_ls = ('%6.3f' % float(vt_result.split()[1]))
        logProb_ls = ('%7.3f' % float(vt_result.split()[2]))
        Amp_ls = ('%6.3f' % float(vt_result.split()[-1]))
        Per_ls_num = float(vt_result.split()[1])
        log10_P = np.log10(Per_ls_num)
        sample_around_logP_region = 0.05
        is_alias = ((log10_P >= np.log10(0.5)-sample_around_logP_region) & (log10_P <= np.log10(0.5)+sample_around_logP_region)) or ((log10_P >= np.log10(1.0)-sample_around_logP_region) & (log10_P <= np.log10(1.0)+sample_around_logP_region))
        if is_alias:
            vt_callLS = vt_dir+'vartools -i '+lc_file+vartools_command_whitten#*******!
            vt_result = check_output(vt_callLS, shell=True)#*******!
            Per_ls = ('%6.3f' % float(vt_result.split()[5]))
            logProb_ls = ('%7.3f' % float(vt_result.split()[6]))
            Amp_ls = ('%6.3f' % float(vt_result.split()[-1]))
        Per_ls_num = float(vt_result.split()[5])
        log10_P = np.log10(Per_ls_num)
        is_alias2 = ((log10_P >= np.log10(0.5)-sample_around_logP_region) & (log10_P <= np.log10(0.5)+sample_around_logP_region)) or ((log10_P >= np.log10(1.0)-sample_around_logP_region) & (log10_P <= np.log10(1.0)+sample_around_logP_region))
        if is_alias2:
            vt_callLS = vt_dir+'vartools -i '+lc_file+vartools_command_whitten2#*******!
            vt_result = check_output(vt_callLS, shell=True)#*******!
            Per_ls = ('%6.3f' % float(vt_result.split()[9]))
            logProb_ls = ('%7.3f' % float(vt_result.split()[10]))
            Amp_ls = ('%6.3f' % float(vt_result.split()[-1]))
        #print('\n CURRENTLY RUNNING VARTOOLS ON: '+str(lc_id))#*******!
        #print('--'*30)#*******!
        #print('[COMMAND]--',vt_callLS)#*******!
        #print('[RESULT]--',vt_result)#*******!
    else:
        dataFrameIndex = np.where(latestFullVartoolsRun.lc_id == lc_id)[0][0]
        #latestFullVartoolsRun_index = np.where(latestFullVartoolsRun[' dec'].values == 0.0)[0][0]
        Per_ls = latestFullVartoolsRun.all_Per_ls[dataFrameIndex]
        logProb_ls = latestFullVartoolsRun.all_logProb_ls[dataFrameIndex]
        Amp_ls = latestFullVartoolsRun.all_Amp_ls[dataFrameIndex]
        #all_a95 = latestFullVartoolsRun[' a95'].values[dataFrameIndex]
        #all_ChiSq = latestFullVartoolsRun[' Chi2'].values[dataFrameIndex]
        #all_skewness = latestFullVartoolsRun[' lc_skew'].values[dataFrameIndex]
    if np.float64(logProb_ls) < -10.0:
        # ---------------------------
        # Plot Phased LC from -LS
        # ---------------------------
        # read in the file
        phLCdata = vt_outdir+str(lc_id)+'.lc.killharm.model'
        phLC_colnames = ['MJD', 'Mag', 'PredMag', 'Magerr']
        pLC = pd.read_csv(phLCdata, sep=' ', index_col=False, header=None,  names=phLC_colnames)
        # assign lists to columns
        plc_mjd= pLC['MJD']
        nxt_mjd = [ ph+1 for ph in plc_mjd ] # add 1 to the phase to be able to plot phase from 0 to 2
        plc_mag = pLC['Mag']
        plc_predmag = pLC['PredMag']
        plc_err = pLC['Magerr']
        # plot the phased light curve
        #fig_size = (8,6)
        #plt_ax.figure(figsize=fig_size)  
        #plt_ax.clf()
        plt_ax.plot(plc_mjd,plc_mag, 'k.',ms=6, label='Phased LC') # phase 0 to 1
        plt_ax.plot(nxt_mjd,plc_mag, 'k.',ms=6) # phase 1 to 2
        # plot the predicted light curve with green open diamonds
        plt_ax.plot(plc_mjd,plc_predmag,'D',ms=8,mfc='None',mec='green',mew=0.2, label='Predicted LC')  # green open Diamonds
        plt_ax.plot(nxt_mjd,plc_predmag,'D',ms=8,mfc='None',mec='green',mew=0.2)  # green open Diamonds (for phase 1 to 2)
        plt_ax.axhline(fmagmn, color='r', ls='-', lw=2, label='Mean Mag')
        plt_ax.axhline(fmagmn+3*ferrmn, color='g', ls='-.', lw=2 ,alpha=0.5, label='3X Mag Err')
        plt_ax.axhline(fmagmn-3*ferrmn, color='g', ls='-.', lw=2 ,alpha=0.5)
        plt_ax.axhline(fmagmn+3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5, label='3X Mag StDev')
        plt_ax.axhline(fmagmn-3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5)
        if is_alias:
            title_line1 = 'CSS ID: {!s} | P={!s} \n logProb={!s} | Amp={!s}  \n ngood={!s} | nreject={!s} \n nabove={!s} ({!s}%) | nbelow={!s} ({!s}%) \n'.format(lc_id, Per_ls+"w", logProb_ls, Amp_ls, nfmag, rejects, nabove, np.round((nabove/nmag)*100,2), nbelow, np.round((nbelow/nmag)*100,2))
        else:
            title_line1 = 'CSS ID: {!s} | P={!s} \n logProb={!s} | Amp={!s}  \n ngood={!s} | nreject={!s} \n nabove={!s} ({!s}%) | nbelow={!s} ({!s}%) \n'.format(lc_id, Per_ls, logProb_ls, Amp_ls, nfmag, rejects, nabove, np.round((nabove/nmag)*100,2), nbelow, np.round((nbelow/nmag)*100,2))
        #title_line2 = 'Drake: P={!s} | Amp={!s} | VarType={!s} | Subclass={!s}'.format(D_Per, D_Amp, D_Vartype, D_sub)
        title_str = title_line1 #+title_line2
        plt_ax.set_title(title_str, fontsize=12)
        plt_ax.set_xlabel('Phase')
        plt_ax.set_ylabel('mag')
        plt_ax.invert_yaxis() # flip the y-axis so fainter mags are on bottom
        # save plot
        #plotname = plt_dir+lc_id+'_plc'+'.eps'
        #plt_ax.savefig(plotname,dpi=600,bbox_inches='tight')
        #plt_ax.clf()
        #plt_ax.close()
    else:
        # ------------------
        # Plot Raw LC:
        # ------------------
        #plot(lc_mjd,lc_mag, 'k.',ms=6)  use the filtered data below
        plt_ax.plot(flc_mjd, flc_mag, 'k.', ms=6, label='Raw Data') 
        # draw a y=constant line with plt.axhline(y, xmin, xmax) 
        # plot mean mag. as solid line
        plt_ax.axhline(fmagmn, color='r', ls='-', lw=2, label='Mag. Mean')
        # plot 3X errmn as dashed line
        plt_ax.axhline(fmagmn+3*ferrmn, color='g', ls='--', lw=2 ,alpha=0.5, label='3x Mean Mag. Err')
        plt_ax.axhline(fmagmn-3*ferrmn, color='g', ls='--', lw=2 ,alpha=0.5)
        # plot 3X stdev as dotted line
        plt_ax.axhline(fmagmn+3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5, label='3x StDev Mag.')
        plt_ax.axhline(fmagmn-3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5)
        # set plot labels
        #title_str = f'{lc_id} \n mean mag. = {fmagmn:0.2f}'
        if is_alias & is_alias2:
            title_line1 = 'CSS ID: {!s} | P={!s} \n logProb={!s} | Amp={!s}  \n ngood={!s} | nreject={!s} \n nabove={!s} ({!s}%) | nbelow={!s} ({!s}%) \n'.format(lc_id, Per_ls+"ww", logProb_ls, Amp_ls, nfmag, rejects, nabove, np.round((nabove/nmag)*100,2), nbelow, np.round((nbelow/nmag)*100,2))
        elif is_alias:
            title_line1 = 'CSS ID: {!s} | P={!s} \n logProb={!s} | Amp={!s}  \n ngood={!s} | nreject={!s} \n nabove={!s} ({!s}%) | nbelow={!s} ({!s}%) \n'.format(lc_id, Per_ls+"w", logProb_ls, Amp_ls, nfmag, rejects, nabove, np.round((nabove/nmag)*100,2), nbelow, np.round((nbelow/nmag)*100,2))
        else:
            title_line1 = 'CSS ID: {!s} | P={!s} \n logProb={!s} | Amp={!s}  \n ngood={!s} | nreject={!s} \n nabove={!s} ({!s}%) | nbelow={!s} ({!s}%) \n'.format(lc_id, Per_ls, logProb_ls, Amp_ls, nfmag, rejects, nabove, np.round((nabove/nmag)*100,2), nbelow, np.round((nbelow/nmag)*100,2))
        title_line2 = 'Drake: P={!s} | Amp={!s} | VarType={!s} '.format(D_Per, D_Amp, D_Vartype)
        #title_line2 = 'Drake: P={!s} | Amp={!s} | VarType={!s} | Subclass={!s}'.format(D_Per, D_Amp, D_Vartype, D_sub)
        plt_ax.set_title(title_line1+" \n "+title_line2, fontsize=12)
        plt_ax.set_xlabel('MJD')
        plt_ax.set_ylabel('Mag')
        #plt_ax.legend(loc='upper right', fontsize=8)
        plt_ax.invert_yaxis() # flip the y-axis so fainter mags are on bottom

    prop_header = "lc_id, Per_ls, logProb_ls, Amp_ls, Mt, a95, lc_skew, Chi2, brtcutoff, brt10per, fnt10per, fntcutoff, errmn, ferrmn, rejects, nabove, nbelow"
    if runVartools:
        properties = np.array([lc_id, Per_ls, logProb_ls, Amp_ls, Mt, a95, lc_skew, Chi2, 
                               brtcutoff, brt10per, fnt10per, fntcutoff, errmn, ferrmn, rejects, nabove, nbelow])
    else:
        dataFrameIndex = np.where(latestFullVartoolsRun.lc_id == lc_id)[0][0]
        #properties = latestFullVartoolsRun.values[dataFrameIndex, 2:]
        properties = latestFullVartoolsRun.latestFullVartoolsRun.values[dataFrameIndex,2:]

    return properties

def plot_SDSS_DR_spec(plate_string, mjd_string, fiberid_string, object_color, object_SDSS_Mr, TDSSprop, TDSS_file_index, box_size, plt_ax):
    raw_SDSS_fits_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/SDSS_spec/getting_DR14_spec/RAW_spec/"
    IRAF_SDSS_fits_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/SDSS_spec/getting_DR14_spec/IRAF_FITS/"
    line_list_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/General/SpecLineLists/"
    spectral_type_prop_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/General/SpecTypeProp/"

    plate = np.int(plate_string)
    mjd = np.int(mjd_string)
    fiberid = np.int(fiberid_string)

    short_spec_filename = "spec-"+plate_string+"-"+mjd_string+"-"+fiberid_string+".fits"
    spec = fits.open(raw_SDSS_fits_dir+short_spec_filename)

    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='valid')
        return y_smooth

    xmin = 3800
    xmax = 10000
    sig_range = 3.0
    major_tick_space = 1000
    minor_tick_space = 100

    spectral_type_prop = np.genfromtxt(spectral_type_prop_dir+"tab5withMvSDSScolors.dat",comments="#",dtype='U')
    spectral_types = spectral_type_prop[:,0]
    gmr_spectral_types = spectral_type_prop[:,14]

    close_color_match_index = np.where(np.abs(object_color-np.float64(gmr_spectral_types)) == np.abs(object_color-np.float64(gmr_spectral_types)).min())[0][0]
    matched_spec_type = spectral_types[close_color_match_index][0]

    pyhammerResults = np.genfromtxt("sup_data/PyHammerResults.csv", delimiter=",", comments="#", dtype="U")
    filenames = pyhammerResults[:,0]
    filenames = np_f.replace(filenames, ".txt", ".fits")
    filenames = [name.lstrip("/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/SDSS_spec/ALL_VARSTAR_SPEC/ASCII/") for name in filenames]
    filenames = ["spec-"+name.replace("_"," ").split()[-1] for name in filenames]
    filenames = np.array(filenames)

    specTypeMatch_Index = np.where(filenames == short_spec_filename)[0][0]
    specTypeMatch = pyhammerResults[specTypeMatch_Index,3]
    specTypeMatch_code = re.split('(\d+)',specTypeMatch)[0]
    specTypeMatch_subType_code = re.split('(\d+)',specTypeMatch)[1]
    pyhammer_RV = pyhammerResults[specTypeMatch_Index,2]
    pyhammer_RV = np.float64(pyhammer_RV)
    pyhammer_RV = np.round(pyhammer_RV, 2)
    pyhammer_RV = str(pyhammer_RV)
    pyhammer_FeH_string = pyhammerResults[specTypeMatch_Index,4]
    pyhammer_FeH = np.float64(pyhammerResults[specTypeMatch_Index,4])

    spec_code_alph = np.array(['O','B','A','F','G','K','M','L','C','WD'])
    spec_code_num = np.arange(10)

    this_spec_num_code = np.where(spec_code_alph == specTypeMatch_code)[0][0]

    template_file_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/WORKING_DIRECTORY/Spectral_fitting/PyHammer/PyHammer-master/resources/templates/"
    if this_spec_num_code == 0:
        tempName = 'O' + str(specTypeMatch_subType_code) + '.fits'
    elif this_spec_num_code == 1: 
        tempName = 'B' + str(specTypeMatch_subType_code) + '.fits'
    elif this_spec_num_code == 2 and float(specTypeMatch_subType_code) < 3:
        tempName = 'A' + str(specTypeMatch_subType_code) + '.fits'
    elif this_spec_num_code == 2 and float(specTypeMatch_subType_code) > 2: 
        tempName = 'A' + str(specTypeMatch_subType_code) + '_-1.0_Dwarf.fits'
    elif this_spec_num_code == 3: 
        tempName = 'F' + str(specTypeMatch_subType_code) + '_-1.0_Dwarf.fits'
    elif this_spec_num_code == 4: 
        tempName = 'G' + str(specTypeMatch_subType_code) + '_+0.0_Dwarf.fits'
    elif this_spec_num_code == 5: 
        tempName = 'K' + str(specTypeMatch_subType_code) + '_+0.0_Dwarf.fits'
    elif this_spec_num_code == 6 and float(specTypeMatch_subType_code) < 9: 
        tempName = 'M' + str(specTypeMatch_subType_code) + '_+0.0_Dwarf.fits'
    elif this_spec_num_code == 6 and float(specTypeMatch_subType_code) == 9: 
        tempName = 'M' + str(specTypeMatch_subType_code) + '.fits'
    elif this_spec_num_code == 7: 
        tempName = 'L' + str(specTypeMatch_subType_code) + '.fits'
    elif this_spec_num_code == 8: 
        tempName = 'C' + str(specTypeMatch_subType_code) + '.fits'
    elif this_spec_num_code == 9: 
        tempName = 'WD' + str(specTypeMatch_subType_code) + '.fits'

    temp = fits.open(template_file_dir+tempName)
    temp_loglam = temp[1].data.field('LogLam')
    temp_lam = 10.0**temp_loglam
    temp_flux = temp[1].data.field('Flux')
    #line_lis_all = np.genfromtxt("aaaLineList_2.list",comments='#',dtype="S")
    #line_lis_all = np.genfromtxt(line_list_dir+"H_lines.list",comments='#',dtype="S")
    #line_lis_all = np.genfromtxt(line_list_dir+"spec_types/"+matched_spec_type+"star_lines.list",comments='#',dtype="S")
    line_lis_all = np.genfromtxt(line_list_dir+"spec_types/"+specTypeMatch_code+"star_lines.list",comments='#',dtype="S")

    lineList_wavelength = np.float64(line_lis_all[:,0])
    lineList_labels = np.empty(lineList_wavelength.size,dtype="U60")
    for ii in range(lineList_wavelength.size):
        lineList_labels[ii] = line_lis_all[ii,1].decode(encoding="utf-8", errors="strict")


    ra_string = '{:0>9.5f}'.format(spec[2].data.field('plug_ra')[0])
    dec_string = '{:0=+10.5f}'.format(spec[2].data.field('plug_dec')[0])
    plate_string = '{:0>4}'.format(str(np.int(spec[2].data.field('plate')[0])))
    mjd_string = '{:0>5}'.format(str(np.int(spec[2].data.field('mjd')[0])))
    fiberid_string = '{:0>4}'.format(str(np.int(spec[2].data.field('fiberid')[0])))
    new_filename = ra_string+dec_string+"_"+plate_string+"-"+mjd_string+"-"+fiberid_string
    flux = spec[1].data.field('flux')
    loglam = spec[1].data.field('loglam')
    wavelength = 10**loglam
    flux = removeSdssStitchSpike(wavelength, flux)
    cz = np.round(const.c.to(u.km/u.s).value*spec[2].data.field('Z_NOQSO'),2)[0]
    cz_err = np.round(const.c.to(u.km/u.s).value*spec[2].data.field('Z_ERR_NOQSO'),2)[0]
    subclass = spec[2].data.field('SUBCLASS_NOQSO')[0]
    if subclass == '':
        subclass = 'None'
    #ELODIE_BV = spec[2].data.field('ELODIE_BV')[0]
    #ELODIE_TEFF = spec[2].data.field('ELODIE_TEFF')[0]
    #ELODIE_LOGG = spec[2].data.field('ELODIE_LOGG')[0]
    #ELODIE_FEH = spec[2].data.field('ELODIE_FEH')[0]

    trim_spectrum_left = 10 #number of pixels to trim from left side
    smooth_flux = smooth(flux[trim_spectrum_left:],box_size)
    smooth_wavelength = smooth(wavelength[trim_spectrum_left:],box_size)

    plotted_region = np.where( (smooth_wavelength >= xmin) & (smooth_wavelength <= xmax))[0]
    ymin = smooth_flux[plotted_region].min()
    ymax = smooth_flux[plotted_region].max()

    np.where(smooth_flux == ymax)[0]

    this_EqW = eqw(wavelength, flux)
    if np.isnan(this_EqW):
        EqW_string = ""
        plot_title = str("RA: "+ra_string+", DEC: "+dec_string+" | cz = "+str(cz)+"$\pm$"+str(cz_err)+" km s$^{-1}$ | SDSS Subclass = "
                    +str(subclass.split()[0])+"\n PyHammer = "+specTypeMatch+EqW_string+", RV = "+pyhammer_RV+" km s$^{-1}$"+"\n "
                    +"DR | Plate = "+plate_string+" MJD = "+mjd_string+" Fiberid = "+fiberid_string+" | GaiaDR2 Dist = "+str(np.round(TDSSprop.gaia_dist[TDSS_file_index],2))
                    +" pc ("+str(np.round(TDSSprop.gaia_parallax[TDSS_file_index]/TDSSprop.gaia_parallax_error[TDSS_file_index],2))+") | GaiaDR2 PMtot = "+str(np.round(TDSSprop.gaia_pmTOT[TDSS_file_index],2))
                    +" mas/yr ("+str(np.round(TDSSprop.gaia_pmTOT[TDSS_file_index]/TDSSprop.gaia_pmTOT_error[TDSS_file_index], 2))+")")
    elif this_EqW > -2.0:
        EqW_string = ""
        plot_title = str("RA: "+ra_string+", DEC: "+dec_string+" | cz = "+str(cz)+"$\pm$"+str(cz_err)+" km s$^{-1}$ | SDSS Subclass = "
                    +str(subclass.split()[0])+"\n PyHammer = "+specTypeMatch+EqW_string+", RV = "+pyhammer_RV+" km s$^{-1}$"+"\n "
                    +"DR | Plate = "+plate_string+" MJD = "+mjd_string+" Fiberid = "+fiberid_string+" | GaiaDR2 Dist = "+str(np.round(TDSSprop.gaia_dist[TDSS_file_index],2))
                    +" pc ("+str(np.round(TDSSprop.gaia_parallax[TDSS_file_index]/TDSSprop.gaia_parallax_error[TDSS_file_index],2))+") | GaiaDR2 PMtot = "+str(np.round(TDSSprop.gaia_pmTOT[TDSS_file_index],2))
                    +" mas/yr ("+str(np.round(TDSSprop.gaia_pmTOT[TDSS_file_index]/TDSSprop.gaia_pmTOT_error[TDSS_file_index], 2))+")")
    else:
        EqW_string = "e"
        this_EqW_str = str(np.round(this_EqW,2))
        plot_title = str("RA: "+ra_string+", DEC: "+dec_string+" | cz = "+str(cz)+"$\pm$"+str(cz_err)+" km s$^{-1}$ | SDSS Subclass = "
                    +str(subclass.split()[0])+"\n PyHammer = "+specTypeMatch+EqW_string+", RV = "+pyhammer_RV+" km s$^{-1}$, EQW = "+this_EqW_str+"\n "
                    +"DR | Plate = "+plate_string+" MJD = "+mjd_string+" Fiberid = "+fiberid_string+" | GaiaDR2 Dist = "+str(np.round(TDSSprop.gaia_dist[TDSS_file_index],2))
                    +" pc ("+str(np.round(TDSSprop.gaia_parallax[TDSS_file_index]/TDSSprop.gaia_parallax_error[TDSS_file_index],2))+") | GaiaDR2 PMtot = "+str(np.round(TDSSprop.gaia_pmTOT[TDSS_file_index],2))
                    +" mas/yr ("+str(np.round(TDSSprop.gaia_pmTOT[TDSS_file_index]/TDSSprop.gaia_pmTOT_error[TDSS_file_index], 2))+")")

    lam8000_index =  np.where(np.abs(smooth_wavelength-8000.0) == np.abs(smooth_wavelength-8000.0).min())[0][0]
    current_spec_flux_at_8000 = smooth_flux[lam8000_index]
    temp_flux_scaled = temp_flux * current_spec_flux_at_8000
    #smooth_flux = smooth_flux/current_spec_flux_at_8000

    plt_ax.plot(smooth_wavelength,smooth_flux,color='black',linewidth=0.5)
    plt_ax.plot(temp_lam, temp_flux_scaled, color='red', alpha=0.3, linewidth=0.5)
    plt_ax.set_xlabel(r"Wavelength [$\AA$]")#, fontdict=font)
    plt_ax.set_ylabel(r"Flux [10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA$$^{-1}$]")#, fontdict=font)
    plt_ax.set_title(plot_title)
    plt_ax.set_xlim([xmin,xmax])
    plt_ax.set_ylim([ymin,ymax])
    #plt_ax.axvspan(5550, 5604, facecolor=ma.colorAlpha_to_rgb('grey', 0.5)[0])#, alpha=0.3)
    plt_ax.xaxis.set_major_locator(ticker.MultipleLocator(major_tick_space))
    plt_ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_tick_space))
    for ll in range(lineList_wavelength.size):
        plt_ax.axvline(x=lineList_wavelength[ll],ls='dashed',c=ma.colorAlpha_to_rgb('k', 0.1)[0])
        x_bounds = plt_ax.get_xlim()
        vlineLabel_value = lineList_wavelength[ll] + 20.0
        #plt_ax.annotate(s=lineList_labels[ll], xy =(((vlineLabel_value-x_bounds[0])/(x_bounds[1]-x_bounds[0])),0.01), 
        #                xycoords='axes fraction', verticalalignment='right', horizontalalignment='right bottom' , rotation = 90)
        plt_ax.text(lineList_wavelength[ll]+20.0,plt_ax.get_ylim()[0]+0.50,lineList_labels[ll],rotation=90, color=ma.colorAlpha_to_rgb('k', 0.2)[0])

    spec.close()
    return this_EqW

def plot_SDSS_prop_spec(plate, mjd, fiberid, object_color, object_SDSS_Mr, TDSSprop, TDSS_file_index, box_size, spAll, plt_ax):  
    plate = np.int(plate)
    mjd = np.int(mjd)
    fiberid = np.int(fiberid)
    line_list_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/General/SpecLineLists/"
    ascii_data_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/SDSS_spec/getting_prop_spec/propDATA_ASCII/"
    spectral_type_prop_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/General/SpecTypeProp/"

    spAll_plate = spAll[1].data.field('plate')
    spAll_mjd = spAll[1].data.field('mjd')
    spAll_fiberid = spAll[1].data.field('fiberid')

    cz = np.round(const.c.to(u.km/u.s).value*spAll[1].data.field('Z_NOQSO'),2)
    cz_err = np.round(const.c.to(u.km/u.s).value*spAll[1].data.field('Z_ERR_NOQSO'),2)
    subclass = spAll[1].data.field('SUBCLASS_NOQSO')
    # ELODIE_BV = spAll[1].data.field('ELODIE_BV')
    # ELODIE_TEFF = spAll[1].data.field('ELODIE_TEFF')
    # ELODIE_LOGG = spAll[1].data.field('ELODIE_LOGG')
    # ELODIE_FEH = spAll[1].data.field('ELODIE_FEH')

    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='valid')
        return y_smooth

    xmin = 3800
    xmax = 10000
    sig_range = 3.0
    major_tick_space = 1000
    minor_tick_space = 100

    index = np.where( (spAll_plate == plate) & (spAll_mjd ==mjd) & (spAll_fiberid == fiberid)  )[0][0]
    ra_string = '{:0>9.5f}'.format(spAll[1].data.field('plug_ra')[index])
    dec_string = '{:0=+10.5f}'.format(spAll[1].data.field('plug_dec')[index])
    plate_string = '{:0>4}'.format(str(np.int(spAll[1].data.field('plate')[index])))
    mjd_string = '{:0>5}'.format(str(np.int(spAll[1].data.field('mjd')[index])))
    fiberid_string = '{:0>4}'.format(str(np.int(spAll[1].data.field('fiberid')[index])))
    new_filename = ra_string+dec_string+"_"+plate_string+"-"+mjd_string+"-"+fiberid_string
    short_spec_filename = "spec-"+plate_string+"-"+mjd_string+"-"+fiberid_string+".fits"
    try:
        file_data = np.loadtxt(ascii_data_dir+new_filename+".txt",skiprows=1) # cols are wavelength,flux
        wavelength = file_data[:,0]
        flux = file_data[:,1]
    except IOError:
    #except:
        throw_error[ii] = 1
        print(ii,plates[ii],mjds[ii],fiberids[ii])

    flux = removeSdssStitchSpike(wavelength, flux)

    spectral_type_prop = np.genfromtxt(spectral_type_prop_dir+"tab5withMvSDSScolors.dat",comments="#",dtype='U')
    spectral_types = spectral_type_prop[:,0]
    gmr_spectral_types = spectral_type_prop[:,14]

    close_color_match_index = np.where(np.abs(object_color-np.float64(gmr_spectral_types)) == np.abs(object_color-np.float64(gmr_spectral_types)).min())[0][0]
    matched_spec_type = spectral_types[close_color_match_index][0]

    pyhammerResults = np.genfromtxt("sup_data/PyHammerResults.csv", delimiter=",", comments="#", dtype="U")
    filenames = pyhammerResults[:,0]
    filenames = np_f.replace(filenames, ".txt", ".fits")
    filenames = [name.lstrip("/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/SDSS_spec/ALL_VARSTAR_SPEC/ASCII/") for name in filenames]
    filenames = ["spec-"+name.replace("_"," ").split()[-1] for name in filenames]
    filenames = np.array(filenames)

    specTypeMatch_Index = np.where(filenames == short_spec_filename)[0][0]
    specTypeMatch = pyhammerResults[specTypeMatch_Index,3]
    specTypeMatch_code = re.split('(\d+)',specTypeMatch)[0]
    specTypeMatch_subType_code = re.split('(\d+)',specTypeMatch)[1]
    pyhammer_RV = pyhammerResults[specTypeMatch_Index,2]
    pyhammer_RV = np.float64(pyhammer_RV)
    pyhammer_RV = np.round(pyhammer_RV, 2)
    pyhammer_RV = str(pyhammer_RV)
    pyhammer_FeH_string = pyhammerResults[specTypeMatch_Index,4]
    pyhammer_FeH = np.float64(pyhammerResults[specTypeMatch_Index,4])

    spec_code_alph = np.array(['O','B','A','F','G','K','M','L','C','WD'])
    spec_code_num = np.arange(10)

    this_spec_num_code = np.where(spec_code_alph == specTypeMatch_code)[0][0]

    template_file_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/WORKING_DIRECTORY/Spectral_fitting/PyHammer/PyHammer-master/resources/templates/"
    if this_spec_num_code == 0:
        tempName = 'O' + str(specTypeMatch_subType_code) + '.fits'
    #Spectral type B
    elif this_spec_num_code == 1: 
        tempName = 'B' + str(specTypeMatch_subType_code) + '.fits'
    #Spectral types A0, A1, A2 (where there are no metallicity changes)
    elif this_spec_num_code == 2 and float(specTypeMatch_subType_code) < 3:
        tempName = 'A' + str(specTypeMatch_subType_code) + '.fits'
    #Spectral type A3 through A9
    elif this_spec_num_code == 2 and float(specTypeMatch_subType_code) > 2: 
        tempName = 'A' + str(specTypeMatch_subType_code) + '_-1.0_Dwarf.fits'
    #Spectral type F
    elif this_spec_num_code == 3: 
        tempName = 'F' + str(specTypeMatch_subType_code) + '_-1.0_Dwarf.fits'
    #Spectral type G
    elif this_spec_num_code == 4: 
        tempName = 'G' + str(specTypeMatch_subType_code) + '_+0.0_Dwarf.fits'
    #Spectral type K 
    elif this_spec_num_code == 5: 
        tempName = 'K' + str(specTypeMatch_subType_code) + '_+0.0_Dwarf.fits'
    #Spectral type M (0 through 8) 
    elif this_spec_num_code == 6 and float(specTypeMatch_subType_code) < 9: 
        tempName = 'M' + str(specTypeMatch_subType_code) + '_+0.0_Dwarf.fits'
    #Spectral type M9 (no metallicity)
    elif this_spec_num_code == 6 and float(specTypeMatch_subType_code) == 9: 
        tempName = 'M' + str(specTypeMatch_subType_code) + '.fits'
    #Spectral type L
    elif this_spec_num_code == 7: 
        tempName = 'L' + str(specTypeMatch_subType_code) + '.fits'
    elif this_spec_num_code == 8: 
        tempName = 'C' + str(specTypeMatch_subType_code) + '.fits'
    elif this_spec_num_code == 9: 
        tempName = 'WD' + str(specTypeMatch_subType_code) + '.fits'
    # Open the template
    temp = fits.open(template_file_dir+tempName)
    temp_loglam = temp[1].data.field('LogLam')
    temp_lam = 10.0**temp_loglam
    temp_flux = temp[1].data.field('Flux')
    #line_lis_all = np.genfromtxt("aaaLineList_2.list",comments='#',dtype="S")
    #line_lis_all = np.genfromtxt(line_list_dir+"H_lines.list",comments='#',dtype="S")
    #line_lis_all = np.genfromtxt(line_list_dir+"spec_types/"+matched_spec_type+"star_lines.list",comments='#',dtype="S")
    line_lis_all = np.genfromtxt(line_list_dir+"spec_types/"+specTypeMatch_code+"star_lines.list",comments='#',dtype="S")

    lineList_wavelength = np.float64(line_lis_all[:,0])
    lineList_labels = np.empty(lineList_wavelength.size,dtype="U60")
    for ii in range(lineList_wavelength.size):
        lineList_labels[ii] = line_lis_all[ii,1].decode(encoding="utf-8", errors="strict")

    trim_spectrum_left = 10 #number of pixels to trim from left side
    smooth_flux = smooth(flux[trim_spectrum_left:],box_size)
    smooth_wavelength = smooth(wavelength[trim_spectrum_left:],box_size)

    plotted_region = np.where( (smooth_wavelength >= xmin) & (smooth_wavelength <= xmax))[0]
    ymin = smooth_flux[plotted_region].min()
    ymax = smooth_flux[plotted_region].max()

    this_EqW = eqw(wavelength, flux)
    if np.isnan(this_EqW):
        EqW_string = ""
        plot_title = str("RA: "+ra_string+", DEC: "+dec_string+" | cz = "+str(cz[index])+"$\pm$"+str(cz_err[index])+" km s$^{-1}$ | SDSS Subclass = "
                    +str(subclass[index]).split()[0]+"\n PyHammer = "+specTypeMatch+EqW_string+", RV = "+pyhammer_RV+" km s$^{-1}$"+"\n "
                    +"prop. | Plate = "+plate_string+" MJD = "+mjd_string+" Fiberid = "+fiberid_string+" | GaiaDR2 Dist = "+str(np.round(TDSSprop.gaia_dist[TDSS_file_index],2))
                    +" pc ("+str(np.round(TDSSprop.gaia_parallax[TDSS_file_index]/TDSSprop.gaia_parallax_error[TDSS_file_index],2))+") | GaiaDR2 PMtot = "+str(np.round(TDSSprop.gaia_pmTOT[TDSS_file_index],2))
                    +" mas/yr ("+str(np.round(TDSSprop.gaia_pmTOT[TDSS_file_index]/TDSSprop.gaia_pmTOT_error[TDSS_file_index], 2))+")")
    elif this_EqW > -2.0:
        EqW_string = ""
        plot_title = str("RA: "+ra_string+", DEC: "+dec_string+" | cz = "+str(cz[index])+"$\pm$"+str(cz_err[index])+" km s$^{-1}$ | SDSS Subclass = "
                    +str(subclass[index]).split()[0]+"\n PyHammer = "+specTypeMatch+EqW_string+", RV = "+pyhammer_RV+" km s$^{-1}$"+"\n "
                    +"prop. | Plate = "+plate_string+" MJD = "+mjd_string+" Fiberid = "+fiberid_string+" | GaiaDR2 Dist = "+str(np.round(TDSSprop.gaia_dist[TDSS_file_index],2))
                    +" pc ("+str(np.round(TDSSprop.gaia_parallax[TDSS_file_index]/TDSSprop.gaia_parallax_error[TDSS_file_index],2))+") | GaiaDR2 PMtot = "+str(np.round(TDSSprop.gaia_pmTOT[TDSS_file_index],2))
                    +" mas/yr ("+str(np.round(TDSSprop.gaia_pmTOT[TDSS_file_index]/TDSSprop.gaia_pmTOT_error[TDSS_file_index], 2))+")")
    else:
        EqW_string = "e"
        this_EqW_str = str(np.round(this_EqW,2))
        plot_title = str("RA: "+ra_string+", DEC: "+dec_string+" | cz = "+str(cz[index])+"$\pm$"+str(cz_err[index])+" km s$^{-1}$ | SDSS Subclass = "
                    +str(subclass[index]).split()[0]+"\n PyHammer = "+specTypeMatch+EqW_string+", RV = "+pyhammer_RV+" km s$^{-1}$, EQW = "+this_EqW_str+"\n "
                    +"prop. | Plate = "+plate_string+" MJD = "+mjd_string+" Fiberid = "+fiberid_string+" | GaiaDR2 Dist = "+str(np.round(TDSSprop.gaia_dist[TDSS_file_index],2))
                    +" pc ("+str(np.round(TDSSprop.gaia_parallax[TDSS_file_index]/TDSSprop.gaia_parallax_error[TDSS_file_index],2))+") | GaiaDR2 PMtot = "+str(np.round(TDSSprop.gaia_pmTOT[TDSS_file_index],2))
                    +" mas/yr ("+str(np.round(TDSSprop.gaia_pmTOT[TDSS_file_index]/TDSSprop.gaia_pmTOT_error[TDSS_file_index], 2))+")")

    lam8000_index =  np.where(np.abs(smooth_wavelength-8000.0) == np.abs(smooth_wavelength-8000.0).min())[0][0]
    current_spec_flux_at_8000 = smooth_flux[lam8000_index]
    temp_flux_scaled = temp_flux * current_spec_flux_at_8000
    #smooth_flux = smooth_flux/current_spec_flux_at_8000

    plt_ax.plot(smooth_wavelength,smooth_flux,color='black',linewidth=0.5)
    plt_ax.plot(temp_lam, temp_flux_scaled, color='red', alpha=0.3, linewidth=0.5)
    plt_ax.set_xlabel(r"Wavelength [$\AA$]")#, fontdict=font)
    plt_ax.set_ylabel(r"Flux [10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA$$^{-1}$]")#, fontdict=font)
    plt_ax.set_title(plot_title)
    plt_ax.set_xlim([xmin,xmax])
    plt_ax.set_ylim([ymin,ymax])
    #plot.axvspan(5550, 5604, facecolor=ma.colorAlpha_to_rgb('grey', 0.5)[0])#, alpha=0.3)
    plt_ax.xaxis.set_major_locator(ticker.MultipleLocator(major_tick_space))
    plt_ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_tick_space))
    for ll in range(lineList_wavelength.size):
        plt_ax.axvline(x=lineList_wavelength[ll],ls='dashed',c=ma.colorAlpha_to_rgb('k', 0.1)[0])
        x_bounds = plt_ax.get_xlim()
        vlineLabel_value = lineList_wavelength[ll] + 20.0
       # plt_ax.annotate(s=lineList_labels[ll], xy =(((vlineLabel_value-x_bounds[0])/(x_bounds[1]-x_bounds[0])),0.01),
                         #xycoords='axes fraction', verticalalignment='right', horizontalalignment='right bottom' , rotation = 90)
        plt_ax.text(lineList_wavelength[ll]+20.0,plt_ax.get_ylim()[0]+0.50,lineList_labels[ll],rotation=90, color=ma.colorAlpha_to_rgb('k', 0.2)[0])
    return this_EqW

def plot_SDSS_photo(ra, dec, image_dir, plt_ax):
    ra_string = '{:0>9.5f}'.format(ra)
    dec_string = '{:0=+10.5f}'.format(dec)

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

def plot_CMD(xi, yi, zi, object_color, object_color_errs, object_absM, object_absM_errs, upperLimDist, lowerLim_M, plt_ax):
    sdss_zams_prop_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/General/SpecTypeProp/"
    sdss_zams_prop =  np.genfromtxt(sdss_zams_prop_dir+"tab5withMvSDSScolors.dat")

    #M_r = sdss_zams_prop[:,3]  
    M_i = sdss_zams_prop[:,4]  
    g_r = sdss_zams_prop[:,14]
    g_i = g_r + sdss_zams_prop[:,15]
    #M_r_zabms = -0.75 + M_r
    M_i_zabms = -0.75 + M_i

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

    title_str = "M$_i$ = {!s} \n g-i = {!s} \n UpperLim Dist = {!s} pc \n LowerLim Mi = {!s}".format(np.round(object_absM,2), np.round(object_color,2),np.round(upperLimDist,2), np.round(lowerLim_M,2))
    plt_ax.set_title(title_str, fontsize=12)

def plot_middle(css_id, latestFullVartoolsRun, xi, yi, zi, plt_ax):
    all_Per_ls = latestFullVartoolsRun.all_Per_ls
    all_logProb_ls = latestFullVartoolsRun.all_logProb_ls
    all_Amp_ls = latestFullVartoolsRun.all_Amp_ls
    all_a95 = latestFullVartoolsRun.all_a95
    all_ChiSq = latestFullVartoolsRun.all_ChiSq
    all_skewness = latestFullVartoolsRun.all_skewness

    this_object_index = np.where(latestFullVartoolsRun.lc_id == css_id)[0][0]

    where_periodic = np.where(all_logProb_ls <= -10.0)[0]
    where_not_periodic = np.where(all_logProb_ls > -10.0)[0]
    is_periodic = all_logProb_ls[this_object_index] <= -10.0

    sample_around_logP_region = 0.05
    if is_periodic:
        cm = plt.cm.get_cmap('viridis')
        log_allPer = np.log10(all_Per_ls[where_periodic])
        log_allAmp = np.log10(all_Amp_ls[where_periodic])
        where_notPlot = ((log_allPer >= np.log10(0.5)-sample_around_logP_region) & (log_allPer <= np.log10(0.5)+sample_around_logP_region)) | ((log_allPer >= np.log10(1.0)-sample_around_logP_region) & (log_allPer <= np.log10(1.0)+sample_around_logP_region))
        this_all_skewness = all_skewness[where_periodic]
        sc = plt_ax.scatter(log_allPer[~where_notPlot], log_allAmp[~where_notPlot], s=2.5, c=this_all_skewness[~where_notPlot], cmap=cm, vmin=-1, vmax=1)
        divider1 = make_axes_locatable(plt_ax)  
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(sc, cax=cax1)
        cbar1.ax.get_yaxis().labelpad = 0
        cbar1.ax.set_ylabel('Skewness', rotation=270)
        #cbar1.set_clim(-1.0, 1.0)
        single_point_color = cbar1.to_rgba(all_skewness[this_object_index])#np.array(cbar1.to_rgba(all_skewness[this_object_index], bytes=True)).reshape((1,4))
        plt_ax.scatter(np.log10(all_Per_ls[this_object_index]), np.log10(all_Amp_ls[this_object_index]), s=150.0, marker="X", color=single_point_color, edgecolors='red')
        plt_ax.set_xlabel("log(P / d)")
        plt_ax.set_ylabel("log(A / mag)")
        plt_ax.set_xlim([-1.0, 0.5])
        plt_ax.set_ylim([-1.2, 0.5])
        title_str = "log10(P / day) = "+str(np.round(np.log10(all_Per_ls[this_object_index]),2))+"\n log10(Amp / mag) = "+str(np.round(np.log10(all_Amp_ls[this_object_index]),2))+"\n Skewness = "+str(np.round(all_skewness[this_object_index],2))
        plt_ax.set_title(title_str, fontsize=12)
    else:
        #plt_ax.scatter(np.log10(all_ChiSq[where_not_periodic]), all_a95[where_not_periodic], s=1.0, c='grey')
        plt_ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.viridis)
        plt_ax.set_xlabel("log(ChiSq)")
        plt_ax.set_ylabel("a95")
        plt_ax.set_xlim([-0.5, 2.5])
        plt_ax.set_ylim([0.0, 3.0])
        plt_ax.scatter(np.log10(all_ChiSq[this_object_index]), all_a95[this_object_index], s=20.0, marker="X", color='red')
        title_str = "log10($\chi^2$) = "+str(np.round(np.log10(all_ChiSq[this_object_index]),2))+"\n a95 = "+str(np.round(all_a95[this_object_index],2))
        plt_ax.set_title(title_str, fontsize=12)

def eqw(wavelength, flux, line=6562.800, cont1=6500.0, cont2=6650.0):
    region = np.where( (wavelength >= cont1) & (wavelength <= cont2))[0]
    if region.size >0:
        contRegion1 = np.where( (wavelength >= cont1) & (wavelength <= 6540.0))[0]
        contRegion2 = np.where( (wavelength >= 6580.0) & (wavelength <= cont2))[0]
        contRegion = np.concatenate((contRegion1, contRegion2), axis=0)

        cont_wavelength = wavelength[contRegion]
        cont_flux = flux[contRegion]

        new_wavelength = wavelength[region]
        new_flux = flux[region]

        m, b = np.polyfit(cont_wavelength, cont_flux, 1)
        y = m*new_wavelength + b

        intergrand = (y - new_flux ) / y

        new_region1 = np.where( (new_wavelength >= cont1) & (new_wavelength <= 6540.0))[0]
        new_region2 = np.where( (new_wavelength >= 6580.0) & (new_wavelength <= cont2))[0]
        new_region = np.concatenate((new_region1, new_region2), axis=0)

        new_region_inbetween = np.where( (new_wavelength >= 6540.0) & (new_wavelength <= 6580.0))[0]

        cont_var = np.nanvar(new_flux[new_region])
        cont_flux = np.nanmedian(new_flux[new_region_inbetween])
        SNR = cont_flux / np.sqrt(cont_var)

        if SNR >= 3.0:
            return intergrand.sum()
        else:
            return np.nan
    else:
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

def makeViDirs(Vi_dir="/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/WORKING_DIRECTORY/Vi/"):
    datestr = check_output(["/bin/date","+%F"])
    datestr = datestr.decode().replace('\n', '')
    if not os.path.exists(Vi_dir+datestr):
        os.mkdir(Vi_dir+datestr)
    vt_outdir = Vi_dir+datestr+"/varout/"
    lc_dir = Vi_dir+datestr+"/LC/"
    lc_plt_dir = Vi_dir+datestr+"/LC_plots/"
    Vi_plots_dir = Vi_dir+datestr+"/Vi_plots/"
    #photo_img_dir = Vi_dir+datestr+"/photo_img/"
    if not os.path.exists(vt_outdir):
        os.mkdir(vt_outdir)
    if not os.path.exists(lc_dir):
        os.mkdir(lc_dir)
    # if not os.path.exists(lc_plt_dir):
    #     os.mkdir(lc_plt_dir)
    if not os.path.exists(Vi_plots_dir):
        os.mkdir(Vi_plots_dir)
    # if not os.path.exists(photo_img_dir):
    #     os.mkdir(photo_img_dir)
    prop_out_dir = Vi_dir+datestr+"/"
    return prop_out_dir, vt_outdir, lc_dir, Vi_plots_dir, datestr

def checkViRun(TDSS_cssid, Vi_dir="/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/WORKING_DIRECTORY/Vi/"):
    from pathlib import Path
    datestr = check_output(["/bin/date","+%F"])
    datestr = datestr.decode().replace('\n', '')
    prop_out_dir = Vi_dir+datestr+"/"
    my_file = Path(prop_out_dir+"completed_Vi_prop_"+datestr+".csv")
    TDSS_cssid_copy = TDSS_cssid.copy()
    if my_file.is_file():
        properties = np.loadtxt(prop_out_dir+"completed_Vi_prop_"+datestr+".csv", delimiter=",")
        index_where_left_off = np.where(properties[:,0]==0.0)[0][0] 
        last_CSS_ID = properties[index_where_left_off-1, 2].astype(int)
        last_CSS_ID_index = np.where(TDSS_cssid_copy == last_CSS_ID)[0][0]
        TDSS_cssid_copy = TDSS_cssid_copy[last_CSS_ID_index+1:]
        prop_id = index_where_left_off
        return True, prop_id, TDSS_cssid_copy
    else:
        return False, 0, TDSS_cssid

def getLCs(main_lc_data_files_path="/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/CSS_LCs/csvs/"):
    csv_paths = [file for file in glob.glob(main_lc_data_files_path+"*.dat")]
    csv_raw_ids = [CSVS.rstrip(".dat") for CSVS in csv_paths]
    csv_raw_ids = [CSVS.lstrip(main_lc_data_files_path) for CSVS in csv_raw_ids]
    csv_raw_ids = np.array(csv_raw_ids).astype(int)
    col_names = ['MJD', 'mag', 'mag_err']
    CSS_LCs = iter(csv_paths)
    return csv_raw_ids, CSS_LCs, col_names

class TDSSprop:
    main_TDSS_file_path = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/"
    Vi_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/WORKING_DIRECTORY/Vi/"
    def __init__(self, nbins):
        from scipy.stats import kde
        import numpy as np
        from astropy import units as u
        from astropy.io import fits
        from astropy import coordinates as coords
        self.nbins = nbins
        TDSS_prop = fits.open(self.main_TDSS_file_path+"TDSS_SES+PREV_DR12griLT20_GaiaDR2_CSSPerVar_spTypes2_postVI_dist_nsc_CSSid_Vartools_PyHammer_2019-03-04.fits")
        TDSS_prop = TDSS_prop[1]
        self.TDSS_cssid = TDSS_prop.data.field('CSS_ID').astype(int)
        self.gaia_bp_rp = TDSS_prop.data.field('bp_rp')
        self.gaia_g = TDSS_prop.data.field('phot_g_mean_mag')
        self.gaia_dist = TDSS_prop.data.field('r_est')
        self.gaia_dist_lo = TDSS_prop.data.field('r_lo')
        self.gaia_dist_hi = TDSS_prop.data.field('r_hi')
        self.gaia_parallax = TDSS_prop.data.field('parallax')
        self.gaia_parallax_error = TDSS_prop.data.field('parallax_error')
        self.gaia_pmra = TDSS_prop.data.field('pmra')
        self.gaia_pmra_error = TDSS_prop.data.field('pmra_error')
        self.gaia_pmdec = TDSS_prop.data.field('pmdec')
        self.gaia_pmdec_error = TDSS_prop.data.field('pmdec_error')
        self.gaia_pmTOT = np.sqrt(self.gaia_pmra**2 + self.gaia_pmdec**2)
        self.gaia_pmTOT_error = np.sqrt((self.gaia_pmra*self.gaia_pmra_error)**2 + (self.gaia_pmdec*self.gaia_pmdec_error)**2) / self.gaia_pmTOT
        self.gaia_Mg = self.gaia_g + 5.0 - 5.0*np.log10(self.gaia_dist)
        self.SDSS_g =  TDSS_prop.data.field('gmag')
        self.SDSS_g_err =  TDSS_prop.data.field('e_gmag')
        self.SDSS_r =  TDSS_prop.data.field('rmag')
        self.SDSS_i =  TDSS_prop.data.field('imag')
        self.SDSS_i_err =  TDSS_prop.data.field('e_imag')
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
        self.TDSS_ra = TDSS_prop.data.field('RAdeg')
        self.TDSS_dec = TDSS_prop.data.field('DEdeg')
        self.TDSS_plate = TDSS_prop.data.field('plate').astype(int)
        self.TDSS_mjd = TDSS_prop.data.field('mjd').astype(int)
        self.TDSS_fibderid = TDSS_prop.data.field('fiber').astype(int)
        self.TDSS_plate_dr14 = TDSS_prop.data.field('plateDR14').astype(int)
        self.TDSS_mjd_dr14 = TDSS_prop.data.field('mjdDR14').astype(int)
        self.TDSS_fibderid_dr14 = TDSS_prop.data.field('fiberDR14').astype(int)
        self.TDSS_plates = self.TDSS_plate
        self.TDSS_plates[self.TDSS_plate_dr14 > 0] = self.TDSS_plate_dr14[self.TDSS_plate_dr14 > 0]
        self.TDSS_mjds = self.TDSS_mjd
        self.TDSS_mjds[self.TDSS_plate_dr14 > 0] = self.TDSS_mjd_dr14[self.TDSS_plate_dr14 > 0]
        self.TDSS_fiberids = self.TDSS_fibderid
        self.TDSS_fiberids[self.TDSS_plate_dr14 > 0] = self.TDSS_fibderid_dr14[self.TDSS_plate_dr14 > 0]
        self.TDSS_coords = coords.SkyCoord(ra=self.TDSS_ra*u.degree, dec=self.TDSS_dec*u.degree, frame='icrs')
        self.DR14_spec_filenames = np.genfromtxt("list_of_all_DR14_spec.txt",dtype="U")
        self.prop_spec_filenames = np.genfromtxt("list_of_all_prop_spec.txt",dtype="U")        
        self.Drake_index = np.where(np.isnan(TDSS_prop.data.field('Period_(days)')) == False)[0]
        self.Drake_num_to_vartype = np.genfromtxt(self.Vi_dir+"Vi_program/"+"darke_var_types.txt", dtype="U", comments="#", delimiter=",")
        self.D_Per = TDSS_prop.data.field('Period_(days)')
        self.D_Amp = TDSS_prop.data.field('Amplitude')
        self.vartype_num = TDSS_prop.data.field('Var_Type')
        self.pyhammer_RV = TDSS_prop.data.field('PyHammer_RV')
        self.upperLimDist = np.sqrt(600.0**2 - self.pyhammer_RV **2) / (4.74e-3*self.gaia_pmTOT )
        self.lowerLimSDSS_M_i = self.SDSS_i + 5.0 - 5.0*np.log10(self.upperLimDist)
        self.lowerLim_gaia_Mg = self.gaia_g + 5.0 - 5.0*np.log10(self.upperLimDist)
        self.lowerLimSDSS_M_r = self.SDSS_r + 5.0 - 5.0*np.log10(self.upperLimDist)

class latestFullVartoolsRun:
    def __init__(self, latestFullVartoolsRun_filename, nbins=50):
        import numpy as np
        import pandas as pd
        from scipy.stats import kde
        self.latestFullVartoolsRun_filename = latestFullVartoolsRun_filename
        self.nbins = nbins
        self.latestFullVartoolsRun = pd.read_csv(self.latestFullVartoolsRun_filename)
        self.dataFrame_all_Index = np.where(self.latestFullVartoolsRun[' dec'].values == 0.0)[0][0]
        self.lc_id = self.latestFullVartoolsRun[' lc_id'].values[:self.dataFrame_all_Index]
        self.all_Per_ls = self.latestFullVartoolsRun[' Per_ls'].values[:self.dataFrame_all_Index]
        self.all_logProb_ls = self.latestFullVartoolsRun[' logProb_ls'].values[:self.dataFrame_all_Index]
        self.all_Amp_ls = self.latestFullVartoolsRun[' Amp_ls'].values[:self.dataFrame_all_Index]
        self.all_a95 = self.latestFullVartoolsRun[' a95'].values[:self.dataFrame_all_Index]
        self.all_ChiSq = self.latestFullVartoolsRun[' Chi2'].values[:self.dataFrame_all_Index]
        self.all_skewness = self.latestFullVartoolsRun[' lc_skew'].values[:self.dataFrame_all_Index]
        self.where_periodic = np.where(self.all_logProb_ls <= -10.0)[0]
        self.where_not_periodic = np.where(self.all_logProb_ls > -10.0)[0]
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