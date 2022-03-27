from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.style as mplstyle
import matplotlib
# mplstyle.use('fast')
# matplotlib.rcParams['path.simplify'] = True
# matplotlib.rcParams['path.simplify_threshold'] = 1.0
# matplotlib.rcParams['agg.path.chunksize'] = 10000

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
import fitz
from PIL import Image
import io
import sys
import glob

############################################################################
############################################################################
############################################################################
############################################################################
############################################################################

prop_id_start = np.int64(sys.argv[1])
# prop_id_start = 17563

############################################################################
############################################################################
############################################################################
############################################################################
############################################################################


def get_filePDF(prop_id, Pfrac):
    ROW = TDSS_prop[prop_id]
    is_CSS = ROW['CSSLC']
    is_ZTF = ROW['ZTFLC']

    is_CSS = False
    is_ZTFg = False
    is_ZTFr = False

    CSS_base_filename = lc_dir_CSS + f"{prop_id}_CSS_{ROW['CSSID']}"
    ZTFg_base_filename = lc_dir_ZTFg + f"/{prop_id}_ZTFg_{ROW['ZTF_GroupID']}"
    ZTFr_base_filename = lc_dir_ZTFr + f"/{prop_id}_ZTFr_{ROW['ZTF_GroupID']}"

    all_data = [None, None, None]
    if os.path.isfile(CSS_base_filename + f"_{Pfrac}P.pdf"):
        doc = fitz.open(CSS_base_filename  + f"_{Pfrac}P.pdf", width=720, height=432)
        all_data[0] = doc[0].getDisplayList().getPixmap(alpha=False).tobytes()
        doc.close()
    if os.path.isfile(ZTFg_base_filename + f"_{Pfrac}P.pdf"):
        doc = fitz.open(ZTFg_base_filename  + f"_{Pfrac}P.pdf", width=720, height=432)
        all_data[1] = doc[0].getDisplayList().getPixmap(alpha=False).tobytes()
        doc.close()
    if os.path.isfile(ZTFr_base_filename + f"_{Pfrac}P.pdf"):
        doc = fitz.open(ZTFr_base_filename  + f"_{Pfrac}P.pdf", width=720, height=432)
        all_data[2] = doc[0].getDisplayList().getPixmap(alpha=False).tobytes()
        doc.close()

    return all_data


def get_filenames(prop_id, survey, Pfrac):
    ROW = TDSS_prop[prop_id]

    if survey=='CRTS':
        filename  = lc_dir_CSS + f"{prop_id}_CSS_{ROW['CSSID']}" + f"_{Pfrac}P.pdf"
    elif survey=='ZTF-g':
        filename = lc_dir_ZTFg + f"/{prop_id}_ZTFg_{ROW['ZTF_GroupID']}" + f"_{Pfrac}P.pdf"
    elif survey=='ZTF-r':
        filename = lc_dir_ZTFr + f"/{prop_id}_ZTFr_{ROW['ZTF_GroupID']}" + f"_{Pfrac}P.pdf"

    return filename


def get_pdf_data(filename, imageScale=2):
    image_matrix = fitz.Matrix(fitz.Identity)
    image_matrix.preScale(imageScale, imageScale)

    doc = fitz.open(filename)
    return doc[0].getDisplayList().getPixmap(alpha = False, matrix=image_matrix).tobytes()



def check_avaiable_survey(prop_id):
    ROW = TDSS_prop[prop_id]
    is_CSS = ROW['CSSLC']

    is_ZTFg = (~TDSS_prop['ZTF_g_Nepochs'].mask)[prop_id]
    is_ZTFr = (~TDSS_prop['ZTF_r_Nepochs'].mask)[prop_id]

    return is_CSS, is_ZTFg, is_ZTFr

def turn_off_survey_buttons(prop_id, window):
    window['CRTS'].update(value=False, disabled=True)
    window['ZTF-g'].update(value=False, disabled=True)
    window['ZTF-r'].update(value=False, disabled=True)

    window['CRTS-done'].update(value=False, disabled=True)
    window['ZTF-g-done'].update(value=False, disabled=True)
    window['ZTF-r-done'].update(value=False, disabled=True)

    is_CSS, is_ZTFg, is_ZTFr = check_avaiable_survey(prop_id)

    if is_CSS:
        window['CRTS'].update(disabled=False)
        window['CRTS-done'].update(value=False, disabled=False)

    if is_ZTFg:
        window['ZTF-g'].update(disabled=False)
        window['ZTF-g-done'].update(value=False, disabled=False)

    if is_ZTFr:
        window['ZTF-r'].update(value=True, disabled=False)
        window['ZTF-r-done'].update(value=False, disabled=False)
    elif is_ZTFg: 
        window['ZTF-g'].update(value=True)
    elif is_CSS:
        window['CRTS'].update(value=True)


def get_current_survey_selected(window):
    if window['CRTS'].get():
        return "CRTS", [True, False, False]
    elif window['ZTF-g'].get():
        return "ZTF-g", [False, True, False]
    elif window['ZTF-r'].get():
        return "ZTF-r", [False, False, True]


def get_current_Pfrac_selected(window):
    if window['(1/5)P'].get():
        return "0.2"
    elif window['(1/4)P'].get():
        return "0.25"
    elif window['(1/3)P'].get():
        return "0.33"
    elif window['(1/2)P'].get():
        return "0.5"
    elif window['P'].get():
        return "1.0"
    elif window['2P'].get():
        return "2.0"
    elif window['3P'].get():
        return "3.0"
    elif window['4P'].get():
        return "4.0"
    elif window['5P'].get():
        return "5.0"

############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
output_dir = "./Analysis_Results/"
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

survey_list = np.array(['CRTS', 'ZTF-g', 'ZTF-r'])
Pfrac_list = np.array(['(1/5)P', '(1/4)P', '(1/3)P', '(1/2)P', 'P', '2P', '3P', '4P', '5P'])
P_fracs = LCtools.createHarmonicFrac(5)
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
Surveydone_col = [[sg.Text("Survey Checked")],
              [sg.Checkbox('CRTS', key='CRTS-done', default=False, enable_events=True), sg.Text(), sg.Text()],
              [sg.Checkbox('ZTF-g', key='ZTF-g-done', default=False, enable_events=True), sg.Text(), sg.Text()],
              [sg.Checkbox('ZTF-r', key='ZTF-r-done', default=False, enable_events=True), sg.Text(), sg.Text()]]


Survey_col = [[sg.Text("Current Survey")],
              [sg.Radio('CRTS', 'SurveyChoice', key='CRTS', default=True, enable_events=True), sg.Text(), sg.Text()],
              [sg.Radio('ZTF-g', 'SurveyChoice', key='ZTF-g', default=False, enable_events=True), sg.Text(), sg.Text()],
              [sg.Radio('ZTF-r', 'SurveyChoice', key='ZTF-r', default=False, enable_events=True), sg.Text(), sg.Text()]]

Pfrac_col = [[sg.Radio('P', 'PeriodFrac', key='P', default=True, enable_events=True), sg.Text(), sg.Text()],
             [sg.Radio('(1/2)P', 'PeriodFrac', key='(1/2)P', default=False, enable_events=True), sg.Text(), sg.Radio('2P', 'PeriodFrac', key='2P', default=False, enable_events=True)],
             [sg.Radio('(1/3)P', 'PeriodFrac', key='(1/3)P', default=False, enable_events=True), sg.Text(), sg.Radio('3P', 'PeriodFrac', key='3P', default=False, enable_events=True)],
             [sg.Radio('(1/4)P', 'PeriodFrac', key='(1/4)P', default=False, enable_events=True), sg.Text(), sg.Radio('4P', 'PeriodFrac', key='4P', default=False, enable_events=True)],
             [sg.Radio('(1/5)P', 'PeriodFrac', key='(1/5)P', default=False, enable_events=True), sg.Text(), sg.Radio('5P', 'PeriodFrac', key='5P', default=False, enable_events=True)]]

varType_col = [[sg.Radio('Periodic', 'isPeriodic', key='Periodic', default=False, enable_events=True)],
               [sg.Radio('Non-Periodic', 'isPeriodic', key='Non-Periodic', default=True, enable_events=True)],
               [sg.Checkbox('Long term trends', key='Long term trends', default=False, enable_events=True)],
               [sg.Text()],
               [sg.Text()]]

periodicType_col = [[sg.Radio('None', 'PeriodType', key='None', default=True, enable_events=True),   sg.Text(), sg.Radio('single-min', 'PeriodType', key='single-min', default=False, enable_events=True)],
                    [sg.Radio('RRab', 'PeriodType', key='RRab', default=False, enable_events=True),  sg.Text(), sg.Radio('other', 'PeriodType', key='other', default=False, enable_events=True)],
                    [sg.Radio('RRc', 'PeriodType', key='RRc', default=False, enable_events=True),   sg.Text(), sg.Text()],
                    [sg.Radio('EA', 'PeriodType', key='EA', default=False, enable_events=True),    sg.Text(), sg.Text()],
                    [sg.Radio('EB/EW', 'PeriodType', key='EB/EW', default=False, enable_events=True), sg.Text(), sg.Text()]]

button_col = [[sg.Text(), sg.Text()],
              [sg.Text(), sg.Text()],
              [sg.Text(), sg.Text()],
              [sg.Button("Previous"), sg.Button("Next")],
              [sg.Button('Save'), sg.Button('Save/Exit')]]

comment_col = [[sg.Text("Comments")],
               [sg.Multiline(default_text='', size=(35, 5))]]




is_CSS, is_ZTFg, is_ZTFr = check_avaiable_survey(prop_id_start)

if is_ZTFr:
    filename = get_filenames(prop_id_start, 'ZTF-r', '1.0')
elif is_ZTFg:
    filename = get_filenames(prop_id_start, 'ZTF-g', '1.0')
elif is_CSS:
    filename = get_filenames(prop_id_start, 'CRTS', '1.0')

data = get_pdf_data(filename)
image_elem = sg.Image(data=data)#, size=(1440, 864))


layout = [[image_elem],
          [sg.Column(Surveydone_col), sg.Column(Survey_col), sg.VerticalSeparator(), sg.Column(Pfrac_col), sg.VerticalSeparator(), sg.Column(varType_col), sg.VerticalSeparator(), sg.Column(periodicType_col), sg.VerticalSeparator(), sg.Column(comment_col), sg.VerticalSeparator(), sg.Column(button_col)]]


window = sg.Window('TDSS Variable Star Light Curve Inspection',
                            layout,
                            finalize=True,
                            resizable=True,
                            location=(100, 100),
                            element_justification="center",
                            font='Helvetica 18',
                            size=(2000, 1200))


turn_off_survey_buttons(prop_id_start, window)

window['P'].update(value=True)
window['Non-Periodic'].update(value=True)
window['Long term trends'].update(value=False)
window['None'].update(value=True)

############################################################################
############################################################################
############################################################################
############################################################################
############################################################################

# ROW = TDSS_prop[prop_id]

# object_ra = ROW['ra_GaiaEDR3']
# object_dec = ROW['dec_GaiaEDR3']

# c = ICRS(object_ra*u.degree, object_dec*u.degree)
# rahmsstr = c.ra.to_string(u.hour, precision=2, pad=True)
# decdmsstr = c.dec.to_string(u.degree, alwayssign=True, precision=2, pad=True)

# base_filename = lc_dir_CSS + f"{prop_id}_CSS_{ROW['CSSID']}"
# base_filename = lc_dir_ZTFg + f"/{prop_id}_ZTFg_{ROW['ZTF_GroupID']}"
# base_filename = lc_dir_ZTFg + f"/{prop_id}_ZTFr_{ROW['ZTF_GroupID']}"

# base_filename + f"_{P_fracs.round(2)[ii]}P.pdf"


############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
prop_id = prop_id_start

while True:
    event, values = window.read()
    #print(event, values)
    
    if event == sg.WIN_CLOSED or event == 'Save/Exit':
        break
    elif event in Pfrac_list:
        this_survey_name, this_survey_bool = get_current_survey_selected(window)
        data = get_pdf_data(get_filenames(prop_id, this_survey_name, get_current_Pfrac_selected(window)))
        image_elem.update(data=data)
    elif event in survey_list:
        this_survey_name, this_survey_bool = get_current_survey_selected(window)
        data = get_pdf_data(get_filenames(prop_id, this_survey_name, get_current_Pfrac_selected(window)))
        image_elem.update(data=data)
    elif event == 'Previous':
        # save previous selections to Table

        ########################################
        prop_id -= 1
        is_CSS, is_ZTFg, is_ZTFr = check_avaiable_survey(prop_id)

        if is_ZTFr:
            filename = get_filenames(prop_id, 'ZTF-r', '1.0')
        elif is_ZTFg:
            filename = get_filenames(prop_id, 'ZTF-g', '1.0')
        elif is_CSS:
            filename = get_filenames(prop_id, 'CRTS', '1.0')

        data = get_pdf_data(filename)
        image_elem.update(data=data)

        turn_off_survey_buttons(prop_id, window)

        window['P'].update(value=True)
        window['Non-Periodic'].update(value=True)
        window['Long term trends'].update(value=False)
        window['None'].update(value=True)

    elif event == 'Next': 
        # save previous selections to Table

        ########################################
        prop_id += 1
        is_CSS, is_ZTFg, is_ZTFr = check_avaiable_survey(prop_id)

        if is_ZTFr:
            filename = get_filenames(prop_id, 'ZTF-r', '1.0')
        elif is_ZTFg:
            filename = get_filenames(prop_id, 'ZTF-g', '1.0')
        elif is_CSS:
            filename = get_filenames(prop_id, 'CRTS', '1.0')

        data = get_pdf_data(filename)
        image_elem.update(data=data)

        turn_off_survey_buttons(prop_id, window)

        window['P'].update(value=True)
        window['Non-Periodic'].update(value=True)
        window['Long term trends'].update(value=False)
        window['None'].update(value=True)

window.close()