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

import mimic_alpha as ma
import VarStar_Vi_plot_functions as vi
import importlib

vt_dir = '/usr/local/bin/'
Vi_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/WORKING_DIRECTORY/Vi/"

spAll_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/SDSS_spec/getting_prop_spec/"
spAll  = fits.open(spAll_dir+'spAll-v5_10_10_propSPEC.fits')

main_lc_data_files_path="/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/CSS_LCs/csvs/"
#***********************************************
prop_out_dir, vt_outdir, lc_dir, Vi_plots_dir, datestr = vi.makeViDirs()
csv_raw_ids, CSS_LCs, col_names = vi.getLCs()
#***********************************************
#Set paramters for running Vi
box_size = 10
nbins=50

vartools_command = " -LS 0.1 10.0 0.1 1 0 -Phase ls  -Killharm fix 1 1.0 1 1 1 "+vt_outdir+" fitonly "
vartools_command_header = "Name LS_Period_1_0 Log10_LS_Prob_1_0 LS_Periodogram_Value_1_0 LS_SNR_1_0 \
                           Killharm_Mean_Mag_2 Killharm_Period_1_2 Killharm_Per1_Subharm_2_Sincoeff_2 \
                           Killharm_Per1_Subharm_2_Coscoeff_2 Killharm_Per1_Fundamental_Sincoeff_2 \
                           Killharm_Per1_Fundamental_Coscoeff_2 Killharm_Per1_Harm_2_Sincoeff_2 \
                           Killharm_Per1_Harm_2_Coscoeff_2 Killharm_Per1_Amplitude_2 \n"

vartools_command_whitten = " -LS 0.1 10.0 0.1 2 0 -Phase ls  -Killharm fix 1 1.0 1 1 1 "+vt_outdir+" fitonly "
vartools_command_header_whitten = "Name LS_Period_1_0 Log10_LS_Prob_1_0 LS_Periodogram_Value_1_0 LS_SNR_1_0 \
                                LS_Period_1_1 Log10_LS_Prob_1_1 LS_Periodogram_Value_1_1 LS_SNR_1_1 \
                                Killharm_Mean_Mag_2 Killharm_Period_1_2 Killharm_Per1_Subharm_2_Sincoeff_2 \
                                Killharm_Per1_Subharm_2_Coscoeff_2 Killharm_Per1_Fundamental_Sincoeff_2 \
                                Killharm_Per1_Fundamental_Coscoeff_2 Killharm_Per1_Harm_2_Sincoeff_2 \
                                Killharm_Per1_Harm_2_Coscoeff_2 Killharm_Per1_Amplitude_2 \n"

vartools_command_whitten2 = " -LS 0.1 10.0 0.1 3 0 -Phase ls  -Killharm fix 1 1.0 1 1 1 "+vt_outdir+" fitonly "
vartools_command_header_whitten2 = "Name LS_Period_1_0 Log10_LS_Prob_1_0 LS_Periodogram_Value_1_0 LS_SNR_1_0 \
                                LS_Period_1_1 Log10_LS_Prob_1_1 LS_Periodogram_Value_1_1 LS_SNR_1_1 \
                                Killharm_Mean_Mag_2 Killharm_Period_1_2 Killharm_Per1_Subharm_2_Sincoeff_2 \
                                Killharm_Per1_Subharm_2_Coscoeff_2 Killharm_Per1_Fundamental_Sincoeff_2 \
                                Killharm_Per1_Fundamental_Coscoeff_2 Killharm_Per1_Harm_2_Sincoeff_2 \
                                Killharm_Per1_Harm_2_Coscoeff_2 Killharm_Per1_Amplitude_2 \n"

#***********************************************
#***********************************************
ra_dec_css_ID = np.genfromtxt("sup_data/ra_dec_to_CSS_ID.txt")
css_ids = ra_dec_css_ID[:,0].astype(int)
ra = ra_dec_css_ID[:,1]
dec = ra_dec_css_ID[:,2]
#***********************************************
#***********************************************
TDSSprop = vi.TDSSprop(nbins)
#***********************************************
latestFullVartoolsRun_filename = "completed_Vi_prop_2019-02-04.csv"
latestFullVartoolsRun = vi.latestFullVartoolsRun(latestFullVartoolsRun_filename=prop_out_dir+latestFullVartoolsRun_filename)
#***********************************************
TDSS_cssid_orginal = TDSSprop.TDSS_cssid
prop_header = "ra, dec, lc_id, Per_ls, logProb_ls, Amp_ls, Mt, a95, lc_skew, Chi2, brtcutoff, brt10per,\
               fnt10per, fntcutoff, errmn, ferrmn, ngood, nrejects, nabove, nbelow, Eqw"

hasViRun, prop_id, TDSS_cssid, properties = vi.checkViRun(TDSS_cssid_orginal)#if Vi has run, this will find where it let off and continue propid from there

if hasViRun:
    pass
else:
    properties = np.empty((csv_raw_ids.size,21))
    prop_id = 0
    TDSS_cssid = TDSS_cssid_orginal.copy()

#***********************************************
#random_index_to_plot = np.random.randint(low=0, high=TDSS_cssid.size, size=500)
#from_here_TDSS_cssid = TDSS_cssid[random_index_to_plot][194:]
runVartools = True
importlib.reload(vi)  
for css_id_num in TDSS_cssid:
    css_id = main_lc_data_files_path+str(css_id_num)+".dat"
    #css_id_num = np.int(css_id.rstrip(".dat").lstrip(main_lc_data_files_path))
    object_index = np.where(css_ids == css_id_num)[0][0]
    object_ra = ra[object_index]
    object_dec = dec[object_index]
    TDSS_file_index = np.where(TDSS_cssid_orginal == css_id_num)[0][0]
    is_Drake = np.isin(TDSS_file_index,TDSSprop.Drake_index)
    ra_string = '{:0>9.5f}'.format(object_ra)
    dec_string = '{:0=+10.5f}'.format(object_dec)
    if ~np.isin(css_id_num, csv_raw_ids):
        continue
    lc_data_pre_check = pd.read_csv(css_id, delim_whitespace = True, names = col_names)
    lc_data = lc_data_pre_check.dropna(subset = col_names)
    if len(lc_data)<50:
        continue  
    #start = timeit.default_timer()
    #importlib.reload(vi) 
    try:
        dataFrameIndex = np.where(latestFullVartoolsRun.lc_id == css_id_num)[0][0]
    except IndexError:
        continue
    fig = plt.figure(figsize=(12,9), constrained_layout=True)
    gs = GridSpec(2, 7, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1, 0.4, 1, 1])#, hspace=0.3, wspace=0.5)
    ax1 = fig.add_subplot(gs[0, :2])#LC
    ax2 = fig.add_subplot(gs[0, 2:4])#SDSS DR12 Image
    ax3 = fig.add_subplot(gs[0, 5:])#CMD?
    ax4 = fig.add_subplot(gs[1, :])#spectra with lines
    if is_Drake:
        D_Per = TDSSprop.D_Per[TDSS_file_index]
        D_Amp = TDSSprop.D_Amp[TDSS_file_index]
        vartype_num = str(TDSSprop.vartype_num[TDSS_file_index])
        vartype_index = np.where(TDSSprop.Drake_num_to_vartype[:,0] == vartype_num)[0][0]
        D_Vartype = TDSSprop.Drake_num_to_vartype[vartype_index,1].strip()
        #D_sub = TDSS_prop.data.field('SUBCLASS')[TDSS_file_index].replace("+"," ").split()[0]
        properties[prop_id,2:-1] = vi.plot_CSS_LC_Drake(css_id, lc_dir, vartools_command, vartools_command_whitten, vartools_command_whitten2, vt_outdir, main_lc_data_files_path, D_Per, D_Amp, D_Vartype, ax1, runVartools=runVartools, latestFullVartoolsRun=latestFullVartoolsRun)
    else:
        properties[prop_id,2:-1] = vi.plot_CSS_LC_noDrake(css_id, lc_dir, vartools_command, vartools_command_whitten, vartools_command_whitten2, vt_outdir, main_lc_data_files_path, ax1, runVartools=runVartools, latestFullVartoolsRun=latestFullVartoolsRun)
    properties[prop_id,0] = object_ra
    properties[prop_id,1] = object_dec
    plate = TDSSprop.TDSS_plates[TDSS_file_index]
    mjd = TDSSprop.TDSS_mjds[TDSS_file_index]
    fiberid = TDSSprop.TDSS_fiberids[TDSS_file_index]
    plate_string = '{:0>4}'.format(str(np.int(plate)))
    mjd_string = '{:0>5}'.format(str(np.int(mjd)))
    fiberid_string = '{:0>4}'.format(str(np.int(fiberid)))
    short_filename = plate_string+"-"+mjd_string+"-"+fiberid_string+".txt"
    long_filename = "spec-"+short_filename[:-4]+".fits"
    # object_bp_rp = gaia_bp_rp[TDSS_file_index]
    # object_M_G = gaia_Mg[TDSS_file_index]
    object_SDSS_gmr = TDSSprop.SDSS_gmr[TDSS_file_index]
    object_SDSS_Mr = TDSSprop.SDSS_M_r[TDSS_file_index]
    object_SDSS_gmi = TDSSprop.SDSS_gmi[TDSS_file_index]
    object_SDSS_Mi = TDSSprop.SDSS_M_i[TDSS_file_index]
    object_SDSS_Mi_lo_err = TDSSprop.SDSS_M_i_lo_err[TDSS_file_index]
    object_SDSS_Mi_hi_err = TDSSprop.SDSS_M_i_hi_err[TDSS_file_index]
    if np.isin(short_filename, TDSSprop.prop_spec_filenames):
        this_EqW = vi.plot_SDSS_prop_spec(plate, mjd, fiberid, object_SDSS_gmr, object_SDSS_Mr, TDSSprop, TDSS_file_index, box_size, spAll, ax4)
    elif np.isin(long_filename, TDSSprop.DR14_spec_filenames):
        this_EqW = vi.plot_SDSS_DR_spec(plate_string, mjd_string, fiberid_string, object_SDSS_gmr, object_SDSS_Mr, TDSSprop, TDSS_file_index, box_size, ax4)
    else:
        print("Error, spec isn't in DR14 OR prop lists.")
        print("ra =",ra_string)
        print("dec =",dec_string)
        print(long_filename)   
    properties[prop_id, -1] =  this_EqW
    #vi.plot_SDSS_photo(object_ra, object_dec, photo_img_dir, ax2)
    vi.plot_middle(css_id_num, latestFullVartoolsRun, latestFullVartoolsRun.xi_2, latestFullVartoolsRun.yi_2, latestFullVartoolsRun.zi_2, ax2)
    lowerlim_Mi = TDSSprop.lowerLimSDSS_M_i #object_SDSS_Mi
    object_SDSS_Mi_lo_err = np.abs(object_SDSS_Mi - lowerlim_Mi[TDSS_file_index])
    object_absM_errs = [[object_SDSS_Mi_lo_err], [object_SDSS_Mi_hi_err]]
    object_color_errs = TDSSprop.SDSS_gmi_err[TDSS_file_index]
    vi.plot_CMD(TDSSprop.xi, TDSSprop.yi, TDSSprop.zi, object_SDSS_gmi, object_color_errs, object_SDSS_Mi, object_absM_errs, TDSSprop.upperLimDist[TDSS_file_index], TDSSprop.lowerLimSDSS_M_i[TDSS_file_index], ax3)
    #plt.savefig(Vi_plots_dir+ra_string+dec_string+"_Vi.eps",dpi=600,bbox_inches='tight')
    plt.savefig(Vi_plots_dir+ra_string+dec_string+"_Vi.png",dpi=600,bbox_inches='tight')
    #plt.show()
    plt.clf()
    plt.close()
    np.savetxt(prop_out_dir+"completed_Vi_prop_"+datestr+".csv", properties, delimiter=",", header=prop_header, fmt="%f, %f, %i, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f")
    prop_id += 1


np.savetxt(prop_out_dir+"completed_Vi_prop_"+datestr+".csv", properties, delimiter=",", header=prop_header,     fmt="%f, %f, %i, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f")



























