#! /usr/bin/env python

import numpy as np
from matplotlib import pylab
from matplotlib import rc
from colorama import init

def acquire_color_palette():
    def spec(N):
        t = np.linspace(-510, 510, N)
        return np.round(np.clip(np.stack([-t, 510 - np.abs(t), t], axis=1), 0, 255)).astype("float32") / 255

    PALLETE = spec(20)

    # colormap: https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html
    PALLETE[0] = [0, 1.0 * 152 / 255, 1.0 * 83 / 255]  # green
    PALLETE[1] = [1.0 * 228 / 255, 1.0 * 53 / 255, 1.0 * 39 / 255]  # red
    PALLETE[2] = [1.0 * 140 / 255, 1.0 * 3 / 255, 1.0 * 120 / 255]  # purple
    PALLETE[3] = [0, 1.0 * 95 / 255, 1.0 * 129 / 255]  # blue
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
    for i in range(14, 20):
        PALLETE[i] = np.random.random(3)
    return PALLETE

# init(autoreset=True)
# rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 10})
# rc('text', usetex=True)
# params = {'axes.titlesize': 12, 'legend.fontsize': 10, 'legend.numpoints': 1}
# pylab.rcParams.update(params)
