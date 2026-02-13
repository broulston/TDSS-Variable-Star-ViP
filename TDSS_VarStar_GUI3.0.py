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
    # ROW = TDSS_prop[prop_id]
    # is_CSS = ROW['CSSLC']

    # is_ZTFg = (~TDSS_prop['ZTF_g_Nepochs'].mask)[prop_id]
    # is_ZTFr = (~TDSS_prop['ZTF_r_Nepochs'].mask)[prop_id]

    is_CSS = (~TDSS_P_prop['CSS_lc_id'].mask)[prop_id]
    is_ZTFg = (~TDSS_P_prop['ZTF_g_lc_id'].mask)[prop_id]
    is_ZTFr = (~TDSS_P_prop['ZTF_r_lc_id'].mask)[prop_id]

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


def get_current_VarTpe_selected(window):
    if window['None'].get():
        return 'None'
    elif window['RRab'].get():
        return 'RRab'
    elif window['RRc'].get():
        return 'RRc'
    elif window['EA'].get():
        return 'EA'
    elif window['EB/EW'].get():
        return 'EB/EW'
    elif window['single-min'].get():
        return 'single-min'
    elif window['other'].get():
        return 'other'


def store_Pfrac(window, holding_list_check, prop_id, this_survey_bool):
    holding_Pfrac = P_fracs[np.where(Pfrac_list2 == get_current_Pfrac_selected(window))[0][0]]

    this_survey = np.where(this_survey_bool)[0][0]
    holding_list_check[1][prop_id][this_survey][1] = holding_Pfrac


def update_Pfrac(window, holding_list_check, prop_id):
    this_survey_name, this_survey_bool = get_current_survey_selected(window)

    this_survey = np.where(this_survey_bool)[0][0]
    holding_Pfrac = holding_list_check[1][prop_id][this_survey][1]


    this_Pfrac_key = Pfrac_list[np.where(P_fracs == holding_Pfrac)[0][0]]
    window[this_Pfrac_key].update(value=True)


def store_Periodic(window, holding_list_check, prop_id, this_survey_bool):
    this_survey = np.where(this_survey_bool)[0][0]
    holding_list_check[1][prop_id][this_survey][2] = window['Periodic'].get()


def update_Periodic(window, holding_list_check, prop_id):
    this_survey_name, this_survey_bool = get_current_survey_selected(window)

    this_survey = np.where(this_survey_bool)[0][0]
    window['Periodic'].update(value=holding_list_check[1][prop_id][this_survey][2])


def store_longterm(window, holding_list_check, prop_id, this_survey_bool):
    this_survey = np.where(this_survey_bool)[0][0]
    holding_list_check[1][prop_id][this_survey][3] = window['Long term trends'].get()


def update_longterm(window, holding_list_check, prop_id):
    this_survey_name, this_survey_bool = get_current_survey_selected(window)

    this_survey = np.where(this_survey_bool)[0][0]
    window['Long term trends'].update(value=holding_list_check[1][prop_id][this_survey][3])


def store_VarType(window, holding_list_check, prop_id, this_survey_bool):
    this_survey = np.where(this_survey_bool)[0][0]
    holding_list_check[1][prop_id][this_survey][4] = get_current_VarTpe_selected(window)


def update_VarType(window, holding_list_check, prop_id):
    this_survey_name, this_survey_bool = get_current_survey_selected(window)

    this_survey = np.where(this_survey_bool)[0][0]
    holding_VarType = holding_list_check[1][prop_id][this_survey][4]

    window[holding_VarType].update(value=True)


def store_comment(window, holding_list_check, prop_id, previous_survey_bool):
    holding_comment = window['Comments'].get()

    this_survey = np.where(previous_survey_bool)[0][0]
    holding_list_check[1][prop_id][this_survey][5] = holding_comment


def update_comment(window, holding_list_check, prop_id):
    this_survey_name, this_survey_bool = get_current_survey_selected(window)

    this_survey = np.where(this_survey_bool)[0][0]
    holding_comment = holding_list_check[1][prop_id][this_survey][5]

    window['Comments'].Update(holding_comment)


def update_surveyDone(window, holding_list_check, prop_id):
    if holding_list_check[1][prop_id][0] == True:
        window['Checked'].update(value=True)
    elif holding_list_check[1][prop_id][0] == False:
        window['Checked'].update(value=False)


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

TDSS_P_prop = Table.read("Analysis_Results/2021-09-08/TDSS_VarStar_LCprops_09-08-2021.fits")

is_one_periodic = ((TDSS_P_prop['CSS_logProb']<=log10FAP) |  (TDSS_P_prop['ZTF_g_logProb']<=log10FAP) | (TDSS_P_prop['ZTF_r_logProb']<=log10FAP)).data
is_one_periodic_index = np.where(is_one_periodic == True)[0]

is_all_periodic = ((TDSS_P_prop['CSS_logProb']<=log10FAP) &  (TDSS_P_prop['ZTF_g_logProb']<=log10FAP) & (TDSS_P_prop['ZTF_r_logProb']<=log10FAP)).data
is_all_periodic_index = np.where(is_all_periodic == True)[0]


# TDSS_prop = TDSS_prop[is_one_periodic].copy()

sample_size = len(TDSS_prop)

survey_checked_list = np.array(['CRTS-done', 'ZTF-g-done', 'ZTF-r-done'])
survey_list = np.array(['CRTS', 'ZTF-g', 'ZTF-r'])
Pfrac_list = np.array(['(1/5)P', '(1/4)P', '(1/3)P', '(1/2)P', 'P', '2P', '3P', '4P', '5P'])
Pfrac_list2 = np.array(['0.2', '0.25', '0.33', '0.5', '1.0', '2.0', '3.0', '4.0', '5.0'])
varType_list = np.array(['Periodic', 'Non-Periodic'])
periodicType_list = np.array(['None', 'RRab', 'RRc', 'EA', 'EB/EW', 'single-min', 'other'])

P_fracs = LCtools.createHarmonicFrac(5)

############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
init_row_survey = [np.NaN, 1.0, False, False, 'None', ''] #isChecked, Pfrac, nonP/P, longtermtrends, VarType, Comments

if os.path.isfile('period_inspect_data.pkl'):
    with open('period_inspect_data.pkl', 'rb') as fh:
        holding_list_check = pickle.load(fh)
else:
    holding_list_check =[0, [  [init_row_survey.copy(), init_row_survey.copy(), init_row_survey.copy()] for ii in range(sample_size)]]

    for this_prop_id in range(sample_size):
        is_CSS, is_ZTFg, is_ZTFr = check_avaiable_survey(this_prop_id)

        if is_CSS:
            holding_list_check[1][this_prop_id][0][0] = False
        if is_ZTFg:
            holding_list_check[1][this_prop_id][1][0] = False
        if is_ZTFr:
            holding_list_check[1][this_prop_id][2][0] = False

    with open('period_inspect_data.pkl', 'wb') as fh:
       pickle.dump(holding_list_check, fh)


# prop_id_start = np.int64(sys.argv[1])
prop_id_start = holding_list_check[0] #0 #17563
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
              [sg.Button('Save'), sg.Button('Save/Exit')],
              [sg.Text(f'{np.where(is_one_periodic_index == prop_id_start)[0][0]} / {is_one_periodic_index.size} = {np.round(100*np.where(is_one_periodic_index == prop_id_start)[0][0] / is_one_periodic_index.size,2)}', key='ProgText')],
              [sg.ProgressBar(is_one_periodic_index.size, orientation='h', size=(20,20), key='-PROG-')]]

comment_col = [[sg.Text("Comments")],
               [sg.Multiline(default_text='', key='Comments', size=(35, 5), do_not_clear=True)]]


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

############################################################################
############################################################################
############################################################################

def CheckIfAllSurveyChecked(window, holding_list_check, prop_id, this_survey_bool):
    which_selected = np.array([holding_list_check[1][prop_id][ii][0] for ii in range(3)])
    all_selected = np.all(which_selected)

    if all_selected:
        if prop_id >= holding_list_check[0]:
            holding_list_check[0] = prop_id


############################################################################
############################################################################

prop_id = prop_id_start
# prop_id = is_one_periodic_index[prop_id_start]

previous_survey_name, previous_survey_bool = get_current_survey_selected(window)

while True:
    event, values = window.read()
    #print(event, values)
    if (event == 'Save'):
        store_comment(window, holding_list_check, prop_id, previous_survey_bool)
        store_Pfrac(window, holding_list_check, prop_id, previous_survey_bool)

        with open('period_inspect_data.pkl', 'wb') as fh:
            pickle.dump(holding_list_check, fh)
    
    if (event == sg.WIN_CLOSED) or (event == 'Save/Exit'):
        store_comment(window, holding_list_check, prop_id, previous_survey_bool)
        store_Pfrac(window, holding_list_check, prop_id, previous_survey_bool)

        with open('period_inspect_data.pkl', 'wb') as fh:
            pickle.dump(holding_list_check, fh)
        break

    store_comment(window, holding_list_check, prop_id, previous_survey_bool)

    if event in Pfrac_list:
        this_survey_name, this_survey_bool = get_current_survey_selected(window)
        store_Pfrac(window, holding_list_check, prop_id, this_survey_bool)


    if event in varType_list:
        this_survey_name, this_survey_bool = get_current_survey_selected(window)
        store_Periodic(window, holding_list_check, prop_id, this_survey_bool)  


    if event == 'Long term trends':
        this_survey_name, this_survey_bool = get_current_survey_selected(window)
        store_longterm(window, holding_list_check, prop_id, this_survey_bool)


    if event in periodicType_list:
        this_survey_name, this_survey_bool = get_current_survey_selected(window)
        store_VarType(window, holding_list_check, prop_id, this_survey_bool)


    if event in survey_checked_list:
        if event == 'CRTS-done':
            holding_list_check[1][prop_id][0][0] = window['CRTS-done'].get()
        elif event == 'ZTF-g-done':
            holding_list_check[1][prop_id][1][0] = window['ZTF-g-done'].get()
        elif event == 'ZTF-r-done':
            holding_list_check[1][prop_id][2][0] = window['ZTF-r-done'].get()

        this_survey_name, this_survey_bool = get_current_survey_selected(window)
        CheckIfAllSurveyChecked(window, holding_list_check, prop_id, this_survey_bool)


    if ((event == 'Previous') or (event == 'Next')):
        if event == 'Previous':
            # prop_id -= 1
            prop_id = is_one_periodic_index[np.maximum(0, np.where(is_one_periodic_index == prop_id)[0][0] - 1)]

        if event == 'Next': 
            # prop_id += 1
            prop_id = is_one_periodic_index[np.minimum(6431, np.where(is_one_periodic_index == prop_id)[0][0] + 1)]

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

        current_percent = np.round(100 * np.where(is_one_periodic_index == prop_id)[0][0] / is_one_periodic_index.size, 1)
        window['-PROG-'].update(np.where(is_one_periodic_index == prop_id)[0][0])
        window['ProgText'].update(f'{np.where(is_one_periodic_index == prop_id)[0][0]} / {is_one_periodic_index.size} = {current_percent}')



    previous_survey_name, previous_survey_bool = get_current_survey_selected(window)

    update_comment(window, holding_list_check, prop_id)
    update_Pfrac(window, holding_list_check, prop_id)
    update_Periodic(window, holding_list_check, prop_id)
    update_longterm(window, holding_list_check, prop_id)
    update_VarType(window, holding_list_check, prop_id)

    update_surveyDone(window, holding_list_check, prop_id)

    data = get_pdf_data(get_filenames(prop_id, previous_survey_name, get_current_Pfrac_selected(window)))
    image_elem.update(data=data)

window.close()
