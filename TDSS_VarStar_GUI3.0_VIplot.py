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
sg.theme('LightGray1')
import numpy as np

import os
import io
import sys
import pickle

from astropy.table import Table
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from astropy import coordinates as coords
from astropy.coordinates import SkyCoord, ICRS

import blosc
import fitz
from PIL import Image

import glob

############################################################################
############################################################################
############################################################################
############################################################################
############################################################################


def get_filePDF(prop_id, Vi_panel_dir):
    filename = get_filenames(prop_id, Vi_panel_dir)
    if os.path.isfile(filename):
        doc = fitz.open(filename, width=720, height=432)
        data= doc[0].get_displaylist().get_pixmap(alpha=False).tobytes()
        doc.close()

    return data


def get_filenames(prop_id, Vi_panel_dir):
    ROW = TDSS_Table[prop_id]
    object_ra = ROW['ra_GaiaEDR3']
    object_dec = ROW['dec_GaiaEDR3']
    ra_string = '{:0>9.5f}'.format(object_ra)
    dec_string = '{:0=+9.5f}'.format(object_dec)

    return Vi_panel_dir + f"{prop_id}_{ra_string}{dec_string}_Vi.pdf"


def get_pdf_data(filename, imageScale=1):
    image_matrix = fitz.Matrix(fitz.Identity)
    image_matrix.prescale(imageScale, imageScale)

    doc = fitz.open(filename)
    return doc[0].get_displaylist().get_pixmap(alpha = False, matrix=image_matrix).tobytes()


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
    elif window['Rotation'].get():
        return 'Rotation'
    elif window['single-min'].get():
        return 'single-min'
    elif window['sinusoidal'].get():
        return 'sinusoidal'
    elif window['CV'].get():
        return 'CV'
    elif window['Flare'].get():
        return 'Flare'
    elif window['PCEB'].get():
        return 'PCEB'
    elif window['Other'].get():
        return 'Other'


def store_Periodic(window, holding_list_check, prop_id):
    holding_list_check[1][prop_id][1] = window['Periodic'].get()


def update_Periodic(window, holding_list_check, prop_id):
    window['Periodic'].update(value=holding_list_check[1][prop_id][1])


def store_longterm(window, holding_list_check, prop_id):
    holding_list_check[1][prop_id][2] = window['Long term trends'].get()


def update_longterm(window, holding_list_check, prop_id):
    window['Long term trends'].update(value=holding_list_check[1][prop_id][2])


def store_VarType(window, holding_list_check, prop_id):
    holding_list_check[1][prop_id][3] = get_current_VarTpe_selected(window)


def update_VarType(window, holding_list_check, prop_id):
    holding_VarType = holding_list_check[1][prop_id][3]
    window[holding_VarType].update(value=True)


def store_comment(window, holding_list_check, prop_id):
    holding_comment = window['Comments'].get()
    holding_list_check[1][prop_id][4] = holding_comment


def update_comment(window, holding_list_check, prop_id):
    holding_comment = holding_list_check[1][prop_id][4]
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

TDSS_Table = Table.read('/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/WORKING_DIRECTORY/Statistical_Testing/LC_stat_calculations/Control_StatsCalc/TDSS_VarStar_FINAL_Var_ALL_PROP_STATS_03-15-2022.fits')
selection_index = (TDSS_Table['ZTF_g_logProb']<=-5).data & (TDSS_Table['ZTF_r_logProb']<=-5).data & (TDSS_Table['ZTF_g_P']<100.0).data & (TDSS_Table['ZTF_g_P']<100.0).data
# np.ones(len(TDSS_Table)).astype(bool)
# TDSS_Table = TDSS_Table[selection_index]

output_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/WORKING_DIRECTORY/Vi/"
datestr = '2022-04-05'
Vi_panel_dir = output_dir + datestr + "/Vi_plots/"

sample_size = selection_index.sum()
selected_prop_ids = np.arange(len(TDSS_Table))[selection_index]

varType_list = np.array(['Periodic', 'Non-Periodic'])
periodicType_list =  np.array(['None', 'RRab', 'RRc', 'EA', 'EB/EW', 'Rotation', 'single-min', 'sinusoidal', 'CV', 'Flare', 'PCEB', 'Other'])

############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
init_row_survey = [False, True, False, 'None', ''] #isChecked, nonP/P, longtermtrends, VarType, Comments

saved_pickle_filename = 'Vi_inspect_data.pkl'

if os.path.isfile(saved_pickle_filename):
    with open(saved_pickle_filename, 'rb') as fh:
        holding_list_check = pickle.load(fh)
else:
    holding_list_check =[selected_prop_ids[0], [  init_row_survey.copy() for ii in range(len(TDSS_Table))]]

    with open(saved_pickle_filename, 'wb') as fh:
       pickle.dump(holding_list_check, fh)


# prop_id_start = np.int64(sys.argv[1])
prop_id_start = holding_list_check[0] #0 #17563
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################


Surveydone_col = [[sg.Text()],
                  [sg.Checkbox('Checked', key='Checked', default=holding_list_check[1][prop_id_start][0], enable_events=True)]]

varType_col = [[sg.Radio('Periodic', 'isPeriodic', key='Periodic', default=True, enable_events=True)],
               [sg.Radio('Non-Periodic', 'isPeriodic', key='Non-Periodic', default=False, enable_events=True)],
               [sg.Checkbox('Long term trends', key='Long term trends', default=holding_list_check[1][prop_id_start][2], enable_events=True)],
               [sg.Text()],
               [sg.Text()]]

periodicType_col1 = [[sg.Radio('None    ',     'PeriodType',      key='None',     default=True, enable_events=True)],
                    [sg.Radio('RRab    ',     'PeriodType',      key='RRab',     default=False, enable_events=True)],
                    [sg.Radio('RRc     ',      'PeriodType',      key='RRc',      default=False, enable_events=True)],
                    [sg.Radio('EA      ',       'PeriodType',      key='EA',       default=False, enable_events=True)],
                    [sg.Radio('EB/EW   ',    'PeriodType',      key='EB/EW',    default=False, enable_events=True)],
                    [sg.Radio('Rotation', 'PeriodType',      key='Rotation', default=False, enable_events=True)]]

periodicType_col2 = [[sg.Text(), sg.Radio('single-min', 'PeriodType', key='single-min', default=False, enable_events=True)],
                    [sg.Text(), sg.Radio('sinusoidal', 'PeriodType', key='sinusoidal', default=False, enable_events=True)],
                    [sg.Text(), sg.Radio('CV',         'PeriodType', key='CV',         default=False, enable_events=True)],
                    [sg.Text(), sg.Radio('Flare',      'PeriodType', key='Flare',      default=False, enable_events=True)],
                    [sg.Text(), sg.Radio('PCEB',       'PeriodType', key='PCEB',       default=False, enable_events=True)],
                    [sg.Text(), sg.Radio('Other',      'PeriodType', key='Other',      default=False, enable_events=True)]]


button_col = [[sg.Text(), sg.Text()],
              [sg.Text(), sg.Text()],
              [sg.Text(), sg.Text()],
              [sg.Button("Previous"), sg.Button("Next")],
              [sg.Button('Save'), sg.Button('Save/Exit')],
              [sg.Text(f'{np.where(selected_prop_ids == prop_id_start)[0][0]} / {sample_size} = {np.round(100*np.where(selected_prop_ids == prop_id_start)[0][0] / sample_size,2)}', key='ProgText')],
              [sg.ProgressBar(sample_size, orientation='h', size=(20,20), key='-PROG-')]]

comment_col = [[sg.Text("Comments")],
               [sg.Multiline(default_text=f'{holding_list_check[1][prop_id_start][4]}', key='Comments', size=(35, 5), do_not_clear=True)]]


filename = get_filenames(prop_id_start, Vi_panel_dir)
data = get_pdf_data(filename)
image_elem = sg.Image(data=data)#, size=(1440, 864))


layout = [[image_elem],
          [sg.Column(Surveydone_col), sg.VerticalSeparator(), sg.Column(varType_col), sg.VerticalSeparator(), sg.Column(periodicType_col1), sg.Column(periodicType_col2), sg.VerticalSeparator(), sg.Column(comment_col), sg.VerticalSeparator(), sg.Column(button_col)]]

window = sg.Window('TDSS Variable Star Vi',
                            layout,
                            finalize=True,
                            resizable=True,
                            location=(360, 150),
                            element_justification="center",
                            font='Helvetica 18',
                            size=(1300, 900),
                            enable_close_attempted_event=True)


update_Periodic(window, holding_list_check, prop_id_start)
update_VarType(window, holding_list_check, prop_id_start)

prop_id = prop_id_start

while True:
    event, values = window.read()
    #print(event, values)

    store_comment(window, holding_list_check, prop_id)
    if (event == 'Save'):
        #store_comment(window, holding_list_check, prop_id)

        with open(saved_pickle_filename, 'wb') as fh:
            pickle.dump(holding_list_check, fh)
    
    if (event == sg.WINDOW_CLOSE_ATTEMPTED_EVENT or event == 'Save/Exit') and sg.popup_yes_no('Do you really want to exit?') == 'Yes':
    # if (event == sg.WIN_CLOSED) or (event == 'Save/Exit'):
        #store_comment(window, holding_list_check, prop_id)

        with open(saved_pickle_filename, 'wb') as fh:
            pickle.dump(holding_list_check, fh)
        break

    if event in varType_list:
        store_Periodic(window, holding_list_check, prop_id)  


    if event == 'Long term trends':
        store_longterm(window, holding_list_check, prop_id)


    if event in periodicType_list:
        store_VarType(window, holding_list_check, prop_id)


    if event == 'Checked':
        holding_list_check[1][prop_id][0] = window['Checked'].get()

    if ((event == 'Previous') or (event == 'Next')):
        if window['Checked'].get() & (prop_id > holding_list_check[0]):
            holding_list_check[0] = prop_id

        if event == 'Previous':
            # prop_id -= 1
            prop_id = selected_prop_ids[np.maximum(0, np.where(selected_prop_ids == prop_id)[0][0] - 1)]

        if event == 'Next': 
            # prop_id += 1
            prop_id = selected_prop_ids[np.minimum(selected_prop_ids.max()-1, np.where(selected_prop_ids == prop_id)[0][0] + 1)]

        filename = get_filenames(prop_id, Vi_panel_dir)
        data = get_pdf_data(filename)
        image_elem.update(data=data)

        current_percent = np.round(100 * np.where(selected_prop_ids == prop_id)[0][0] / sample_size, 1)
        window['-PROG-'].update(np.where(selected_prop_ids == prop_id)[0][0])
        window['ProgText'].update(f'{np.where(selected_prop_ids == prop_id)[0][0]} / {sample_size} = {current_percent}')


    update_comment(window, holding_list_check, prop_id)
    update_Periodic(window, holding_list_check, prop_id)
    update_longterm(window, holding_list_check, prop_id)
    update_VarType(window, holding_list_check, prop_id)
    update_surveyDone(window, holding_list_check, prop_id)

window.close()
