# %load_ext autoreload
# %autoreload 2

import matplotlib
import matplotlib.style as mplstyle
matplotlib.use('TkAGG')
mplstyle.use('fast')

matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0
matplotlib.rcParams['agg.path.chunksize'] = 10000

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
from matplotlib import animation, rc
from scipy.stats import f
from scipy.signal import find_peaks
from sklearn.cluster import MeanShift, estimate_bandwidth
from astropy.io import fits
from astropy.timeseries import LombScargle

from ResearchTools import LCtools

from astropy.table import Table
from astropy.coordinates import ICRS
import astropy.units as u
from astropy.coordinates import SkyCoord

import warnings
import tqdm.autonotebook as tqdm
# import tqdm.notebook as tqdm
from subprocess import *
import os
import pickle
import lzma
import bz2
import gzip
import blosc

from multiprocessing import Pool



output_dir = "Analysis_Results/"

datestr = check_output(["/bin/date","+%F"])
datestr = datestr.decode().replace('\n', '')
# datestr = '2021-06-15'
if not os.path.exists(output_dir+datestr):
    os.mkdir(output_dir+datestr)

lc_dir0 = output_dir+datestr+"/ZTF/"
lc_dir_CSS = output_dir+datestr+"/CSS/"
lc_dir_ZTFg = output_dir+datestr+"/ZTF/g"
lc_dir_ZTFr = output_dir+datestr+"/ZTF/r"
lc_dir_ZTFi = output_dir+datestr+"/ZTF/i"

if not os.path.exists(lc_dir0):
    os.mkdir(lc_dir0)
if not os.path.exists(lc_dir_CSS):
    os.mkdir(lc_dir_CSS)
if not os.path.exists(lc_dir_ZTFg):
    os.mkdir(lc_dir_ZTFg)
if not os.path.exists(lc_dir_ZTFr):
    os.mkdir(lc_dir_ZTFr)
if not os.path.exists(lc_dir_ZTFi):
    os.mkdir(lc_dir_ZTFi)
    
if not os.path.exists(output_dir+datestr+"/RAW_LC_ANALYSIS/"):
    os.mkdir(output_dir+datestr+"/RAW_LC_ANALYSIS/")

raw_lc_analysis_dir_ZTF = output_dir+datestr+"/RAW_LC_ANALYSIS/"+"/ZTF/"
raw_LC_analysis_dir_CSS = output_dir+datestr+"/RAW_LC_ANALYSIS/"+"/CSS/"
raw_LC_analysis_dir_ZTFg = output_dir+datestr+"/RAW_LC_ANALYSIS/"+"/ZTF/g/"
raw_LC_analysis_dir_ZTFr = output_dir+datestr+"/RAW_LC_ANALYSIS/"+"/ZTF/r/"
raw_LC_analysis_dir_ZTFi = output_dir+datestr+"/RAW_LC_ANALYSIS/"+"/ZTF/i/"   
    
if not os.path.exists(raw_lc_analysis_dir_ZTF):
    os.mkdir(raw_lc_analysis_dir_ZTF)
if not os.path.exists(raw_LC_analysis_dir_CSS):
    os.mkdir(raw_LC_analysis_dir_CSS)
if not os.path.exists(raw_LC_analysis_dir_ZTFg):
    os.mkdir(raw_LC_analysis_dir_ZTFg)
if not os.path.exists(raw_LC_analysis_dir_ZTFr):
    os.mkdir(raw_LC_analysis_dir_ZTFr)
if not os.path.exists(raw_LC_analysis_dir_ZTFi):
    os.mkdir(raw_LC_analysis_dir_ZTFi)        



checkHarmonic = False
log10FAP = -5.0
logFAP_limit = log10FAP
polyfit_deg = 3

CSS_LC_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/CSS_LCs/csvs/"
ZTF_LC_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/ZTF/DATA/07-27-2021/"

ZTF_LC_data = Table.read("/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/ZTF/DATA/07-27-2021/TDSS_VarStar_ZTFDR6_gri_GroupID.fits")
TDSS_prop = Table.read("/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/PROGRAM_SAMPLE/2021-07-27/FINAL_FILES/TDSS_SES+PREV_DR16DR12griLT20_GaiaEDR3_Drake2014PerVar_CSSID_ZTFIDs_LCpointer_PyHammer_EqW.fits")


def low_order_poly(mag, a, b, c, d, e, f_, g):
    return a + b * mag + c * mag**2 + d * mag**3 + e * mag**4 + f_ * mag**5 + g * mag**5


def TDSS_LC_ANALYSIS(prop_id):
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

    this_filename_base = f"{ra_string}{dec_string}_"
    
    if is_CSS:
        if ROW[f"CSS_Nepochs"] > 10:        
            lc_file = CSS_LC_dir+str(ROW['CSSID'])+'.dat'
            CSS_lc_data = Table.read(lc_file, format='ascii', names=['mjd', 'mag', 'magerr'])
            popt = np.array([-2.61242938e+01,  1.93636204e+00,  4.45971381e-01, -6.49419310e-02, 2.99231126e-03,  2.40758201e-01, -2.40805035e-01])
            magerr_resid_mean = 0.008825118765717422
            shift_const = 1.5 * magerr_resid_mean
            pred_magerr = low_order_poly(CSS_lc_data['mag'], popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
            bad_err_index = np.where(CSS_lc_data['magerr'] - (pred_magerr - shift_const) < 0.0)[0]
            CSS_lc_data['magerr'][bad_err_index] = pred_magerr[bad_err_index]
            
            mean_mag = np.nanmean(CSS_lc_data['mag'])
            
            flc_data, LC_stat_properties = LCtools.process_LC(CSS_lc_data, fltRange=5.0, detrend=True)
            if flc_data['QualFlag'].sum() > 10:
                try:
                    LC_period_properties, all_period_properties = LCtools.perdiodSearch(flc_data, minP=0.1, maxP=100.0, log10FAP=log10FAP, checkHarmonic=False)
                    all_period_properties = {**LC_stat_properties, **all_period_properties}
                    #LC_prop = {**LC_stat_properties, **LC_period_properties}
                
                    FAP_power_peak = all_period_properties['ls'].false_alarm_level(10**log10FAP)
                    #new_props = { ii: all_period_properties[ii] for ii in selected_props }
                    new_props = {**all_period_properties, **{'FAP_power_peak':FAP_power_peak}}
              
                    #pickle_filename = raw_LC_analysis_dir_CSS + f"{prop_id}_CSS_{ROW['CSSID']}.xz"
                    #with open(pickle_filename, 'wb') as f:
                    #    pickle.dump(new_props, f)   
                    #with lzma.open(pickle_filename, "wb") as f:
                    #    pickle.dump(new_props, f)
                        
                    pickled_data = pickle.dumps(new_props)  # returns data as a bytes object
                    compressed_pickle = blosc.compress(pickled_data)
                    pickle_filename = raw_LC_analysis_dir_CSS + f"{prop_id}_CSS_{ROW['CSSID']}.dat"
                    with open(pickle_filename, "wb") as f:
                        f.write(compressed_pickle)
                except np.linalg.LinAlgError:
                    print("Singular Matrix", ra_string, dec_string)
                    LC_period_properties, all_period_properties = LCtools.perdiodSearch_linalgfail(flc_data, minP=0.1, maxP=100.0, log10FAP=log10FAP, checkHarmonic=False)
                    all_period_properties = {**LC_stat_properties, **all_period_properties}
                    #LC_prop = {**LC_stat_properties, **LC_period_properties}
                
                    FAP_power_peak = all_period_properties['ls'].false_alarm_level(10**log10FAP)
                    #new_props = { ii: all_period_properties[ii] for ii in selected_props }
                    new_props = {**all_period_properties, **{'FAP_power_peak':FAP_power_peak}}
              
                    #pickle_filename = raw_LC_analysis_dir_CSS + f"{prop_id}_CSS_{ROW['CSSID']}.xz"
                    #with open(pickle_filename, 'wb') as f:
                    #    pickle.dump(new_props, f)   
                    #with lzma.open(pickle_filename, "wb") as f:
                    #    pickle.dump(new_props, f)
                        
                    pickled_data = pickle.dumps(new_props)  # returns data as a bytes object
                    compressed_pickle = blosc.compress(pickled_data)
                    pickle_filename = raw_LC_analysis_dir_CSS + f"{prop_id}_CSS_{ROW['CSSID']}.dat"
                    with open(pickle_filename, "wb") as f:
                        f.write(compressed_pickle)

    if is_ZTF:
        for ii, this_ZTF_filter in enumerate(['g', 'r', 'i']):
            if ROW[f"ZTF_{this_ZTF_filter}_Nepochs"] > 10:
                lc_index = (ZTF_LC_data['ZTF_GroupID'] == ROW['ZTF_GroupID']) & (ZTF_LC_data['filtercode'] == 'z'+this_ZTF_filter)
                lc_data = ZTF_LC_data[lc_index]
                
                mean_mag = np.nanmean(lc_data['mag'])
            
                flc_data, LC_stat_properties = LCtools.process_LC(lc_data, fltRange=5.0, detrend=True)
                if flc_data['QualFlag'].sum() > 10:
                    try:
                        LC_period_properties, all_period_properties = LCtools.perdiodSearch(flc_data, minP=0.1, maxP=100.0, log10FAP=log10FAP, checkHarmonic=False)
                        all_period_properties = {**LC_stat_properties, **all_period_properties}
                        #LC_prop = {**LC_stat_properties, **LC_period_properties}
                        
                        FAP_power_peak = all_period_properties['ls'].false_alarm_level(10**log10FAP)
                        #new_props = { ii: all_period_properties[ii] for ii in selected_props }
                        new_props = {**all_period_properties, **{'FAP_power_peak':FAP_power_peak}}

                        #pickle_filename = raw_lc_analysis_dir_ZTF + f"/{this_ZTF_filter}/" + f"{prop_id}_ZTF{this_ZTF_filter}_{ROW['ZTF_GroupID']}.xz"
                        #with open(pickle_filename, 'wb') as f:
                        #    pickle.dump(new_props, f) 
                        #with lzma.open(pickle_filename, "wb") as f:
                        #    pickle.dump(new_props, f)
                            
                        pickled_data = pickle.dumps(new_props)  # returns data as a bytes object
                        compressed_pickle = blosc.compress(pickled_data)
                        pickle_filename = raw_lc_analysis_dir_ZTF + f"/{this_ZTF_filter}/" + f"{prop_id}_ZTF{this_ZTF_filter}_{ROW['ZTF_GroupID']}.xz"
                        with open(pickle_filename, "wb") as f:
                            f.write(compressed_pickle)
                    except np.linalg.LinAlgError:
                        print("Singular Matrix", ra_string, dec_string)
                        LC_period_properties, all_period_properties = LCtools.perdiodSearch_linalgfail(flc_data, minP=0.1, maxP=100.0, log10FAP=log10FAP, checkHarmonic=False)
                        all_period_properties = {**LC_stat_properties, **all_period_properties}
                        #LC_prop = {**LC_stat_properties, **LC_period_properties}
                        
                        FAP_power_peak = all_period_properties['ls'].false_alarm_level(10**log10FAP)
                        #new_props = { ii: all_period_properties[ii] for ii in selected_props }
                        new_props = {**all_period_properties, **{'FAP_power_peak':FAP_power_peak}}

                        #pickle_filename = raw_lc_analysis_dir_ZTF + f"/{this_ZTF_filter}/" + f"{prop_id}_ZTF{this_ZTF_filter}_{ROW['ZTF_GroupID']}.xz"
                        #with open(pickle_filename, 'wb') as f:
                        #    pickle.dump(new_props, f) 
                        #with lzma.open(pickle_filename, "wb") as f:
                        #    pickle.dump(new_props, f)
                            
                        pickled_data = pickle.dumps(new_props)  # returns data as a bytes object
                        compressed_pickle = blosc.compress(pickled_data)
                        pickle_filename = raw_lc_analysis_dir_ZTF + f"/{this_ZTF_filter}/" + f"{prop_id}_ZTF{this_ZTF_filter}_{ROW['ZTF_GroupID']}.xz"
                        with open(pickle_filename, "wb") as f:
                            f.write(compressed_pickle)

start_index = 18840
if __name__ == '__main__':
    with Pool(os.cpu_count()-2) as pool:
        r = list(tqdm.tqdm(pool.imap(TDSS_LC_ANALYSIS, range(start_index, len(TDSS_prop))), total=len(TDSS_prop)-start_index))
        

