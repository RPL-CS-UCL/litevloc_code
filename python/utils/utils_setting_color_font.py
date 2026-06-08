#! /usr/bin/env python

import numpy as np
from matplotlib import pylab
from matplotlib import rc
from colorama import init

def acquire_color_palette():
    def spec(N):
        t = np.linspace(-510, 510, N)
        return np.round(np.clip(np.stack([-t, 510 - np.abs(t), t], axis=1), 0, 255)).astype("float32") / 255

    PALLETE = spec(60)

    # colormap: https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html
    PALLETE[0] = [0, 152 / 255, 83 / 255]  # green
    PALLETE[1] = [228 / 255, 53 / 255, 39 / 255]  # red
    PALLETE[2] = [140 / 255, 3 / 255, 120 / 255]  # purple
    PALLETE[3] = [0, 95 / 255, 129 / 255]  # blue
    PALLETE[4] = [0.9290, 0.6940, 0.1250]
    PALLETE[5] = [0.6350, 0.0780, 0.1840]
    PALLETE[6] = [0.494, 0.184, 0.556]
    PALLETE[7] = [0.850, 0.3250, 0.0980]
    PALLETE[8] = [0.466, 0.674, 0.188]
    PALLETE[9] = [0.3010, 0.7450, 0.9330]
    PALLETE[10] = [0.6350, 0.0780, 0.1840]
    PALLETE[11] = [0.494, 0.184, 0.556]
    PALLETE[12] = [0.850, 0.3250, 0.0980]
    PALLETE[13] = [0.466, 0.674, 0.188]
    for i in range(14, 60):
        PALLETE[i] = np.random.random(3)
    return PALLETE

def acquire_marker():
    MARKERS = ['o', 's', '^', 'D', 'X', '*', '+', 'o', 's', '^', 'D', '*', 'X']
    return MARKERS

def acquire_linestyle():
    LINES = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    return LINES

def acquire_bar_style():
    BAR_STYLE = ['', '///', 'xx', 'o', 'O', '.', '*', '-', '+', 'x']
    return BAR_STYLE

# Optional font_family:
# rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': fontsize})
# rc('font', **{'family': 'serif', 'serif': ['Palatino'], 'size': fontsize})
# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': fontsize}) # More smooth
def setting_font(fontsize=14, titlesize=14, legend_fontsize=14, font_family='Palatino'):
    try:
        init(autoreset=True)
        rc('font', **{'family': 'serif', 'serif': [font_family], 'size': fontsize})
        rc('text', usetex=True)
        params = {'axes.titlesize': titlesize, 'legend.fontsize': legend_fontsize, 'legend.numpoints': 1}
        pylab.rcParams.update(params)
    except Exception as e:
        init(autoreset=True)
        rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif'], 'size': fontsize})
        rc('text', usetex=False)
        params = {'axes.titlesize': titlesize, 'legend.fontsize': legend_fontsize, 'legend.numpoints': 1}
        pylab.rcParams.update(params)
