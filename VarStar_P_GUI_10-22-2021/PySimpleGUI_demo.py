import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# VARS CONSTS:
# Upgraded dataSize to global...
_VARS = {'window': False,
         'fig_agg': False,
         'pltFig': False,
         'dataSize': 60}


plt.style.use('Solarize_Light2')

# Helper Functions


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


# \\  -------- PYSIMPLEGUI -------- //

AppFont = 'Any 16'
SliderFont = 'Any 14'
sg.theme('black')

# New layout with slider and padding

layout = [[sg.Canvas(key='figCanvas', background_color='#FDF6E3')],
          [sg.Text(text="Random sample size :",
                   font=SliderFont,
                   background_color='#FDF6E3',
                   pad=((0, 0), (10, 0)),
                   text_color='Black'),
           sg.Slider(range=(4, 1000), orientation='h', size=(34, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='-Slider-',
                     enable_events=True),
           sg.Button('Resample',
                     font=AppFont,
                     pad=((4, 0), (10, 0)))],
          # pad ((left, right), (top, bottom))
          [sg.Button('Exit', font=AppFont, pad=((540, 0), (0, 0)))]]

_VARS['window'] = sg.Window('Random Samples',
                            layout,
                            finalize=True,
                            resizable=True,
                            location=(100, 100),
                            element_justification="center",
                            background_color='#FDF6E3')

# \\  -------- PYSIMPLEGUI -------- //


# \\  -------- PYPLOT -------- //


def makeSynthData():
    xData = np.random.randint(100, size=_VARS['dataSize'])
    yData = np.linspace(0, _VARS['dataSize'],
                        num=_VARS['dataSize'], dtype=int)
    return (xData, yData)


def drawChart():
    _VARS['pltFig'] = plt.figure()
    dataXY = makeSynthData()
    plt.plot(dataXY[0], dataXY[1], '.k')
    _VARS['fig_agg'] = draw_figure(
        _VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])


def updateChart():
    _VARS['fig_agg'].get_tk_widget().forget()
    dataXY = makeSynthData()
    # plt.cla()
    plt.clf()
    plt.plot(dataXY[0], dataXY[1], '.k')
    _VARS['fig_agg'] = draw_figure(
        _VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])


def updateData(val):
    _VARS['dataSize'] = val
    updateChart()

# \\  -------- PYPLOT -------- //


drawChart()

# MAIN LOOP
while True:
    event, values = _VARS['window'].read(timeout=200)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Resample':
        updateChart()
    elif event == '-Slider-':
        updateData(int(values['-Slider-']))
        # print(values)
        # print(int(values['-Slider-']))
_VARS['window'].close()
