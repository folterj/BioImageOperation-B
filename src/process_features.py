import os
import glob
from math import sqrt, ceil
from textwrap import wrap
import numpy as np
import matplotlib.pyplot as plt

from file.bio import import_tracks_by_frame
from src.file.plain_csv import export_csv
from parameters import *


def get_norm_data(filename):
    data = import_tracks_by_frame(filename)
    v = np.asarray(list(data['v'].values()))
    v_angle = np.asarray(list(data['v_angle'].values()))
    meanl = np.mean(list(data['length_major'].values()))
    v_norm = v / meanl
    angle_norm = abs(v_angle) / VANGLE_NORM
    return v, v_angle, v_norm, angle_norm, meanl


def get_movement_type(filename):
    movement_type = []
    movement_n = []
    movement_time = {'search': 0, 'walk': 0, 'run': 0}
    data = import_tracks_by_frame(filename)
    v = np.asarray(list(data['v_projection'].values()))
    v_angle = np.asarray(list(data['v_angle'].values()))
    meanl = np.mean(list(data['length_major'].values()))
    v_norm = v / meanl
    angle_norm = abs(v_angle) / VANGLE_NORM
    for v, angle in zip(v_norm, angle_norm):
        if angle > 45 / VANGLE_NORM:
            type = 'search'
            n = 1
        elif v > 2:
            type = 'run'
            n = 3
        elif v > 0.5:
            type = 'walk'
            n = 2
        else:
            type = ''
            n = 0
        movement_type.append(type)
        movement_n.append(n)
        if type != '':
            movement_time[type] += 1
    for type in movement_time:
        print(f'{type}: {movement_time[type]} {movement_time[type] / len(v_norm) * 100:.1f}%')
    return v_norm, angle_norm, movement_type, movement_n


def get_hist_data(data, range):
    hist, bin_edges = np.histogram(data, bins=NBINS, range=(0, range))
    return hist / len(data)


def get_loghist_data(data, power_min, power_max, ax=None, title="", color="#1f77b4"):
    # assume symtric scale - middle bin is 10^0 = 1

    #bin_edges = [10 ** ((i - NBINS / 2) * factor) for i in range(NBINS + 1)]
    bin_edges = np.logspace(power_min, power_max, NBINS + 1)

    # manual histogram, ensuring values at histogram edges are counted
    hist = np.zeros(NBINS)
    factor = (power_max - power_min) / NBINS
    for x in data:
        if x != 0:
            bin = int(np.log10(abs(x) - power_min) / factor)
            bin = np.clip(bin, 0, NBINS)
            hist[bin] += 1

    #hist, _ = np.histogram(data, bins=bin_edges)

    if ax is not None:
        n = len(data)
        h, b, _ = ax.hist(data, bins=bin_edges, weights=[1/n]*n, color=color)
        ax.set_xscale('log')
        ax.title.set_text(title)

    return hist / len(data), bin_edges


def get_v_hists(filename, ax_v, ax_vangle):
    filetitle = os.path.splitext(os.path.basename(filename))[0].replace("_", " ")
    filetitle_plot = "\n".join(wrap(filetitle, 20))

    v, v_angle, normv, norm_angle, meanl = get_norm_data(filename)

    hist_v, _ = get_loghist_data(normv, -2, 2, ax_v, "v " + filetitle_plot, "#1f77b4")
    hist_vangle, _ = get_loghist_data(norm_angle, -3, 1, ax_vangle, "vangle " + filetitle_plot, "#ff7f0e")

    print(filetitle)
    print(round(v.max(), 3), round(meanl, 3), round(v.max() / meanl, 3))
    print(round(v_angle.max(), 3), round(v_angle.max() / VANGLE_NORM, 3))
    print()

    return hist_v, hist_vangle


def draw_hists(files):
    nfigs = len(files) * 2
    ncols = int(ceil(sqrt(nfigs) / 2) * 2)
    nrows = int(ceil(nfigs / ncols))

    plt.rcParams["axes.titlesize"] = 4
    plt.rcParams.update({'font.size': 4})

    fig, axs0 = plt.subplots(nrows, ncols, dpi=300)
    axs = np.asarray(axs0).flatten()

    for ax in axs:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    i = 0
    for filename in files:
        get_v_hists(filename, axs[i], axs[i + 1])
        i += 2

    plt.tight_layout()
    plt.show()


def draw_hist(filename):
    fig, axs = plt.subplots(1, 2, dpi=300)
    get_v_hists(filename, axs[0], axs[1])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    #draw_hist(BIO_TRACKING_FILE)
    #draw_hist(LIVING_EARTH_PATH + "tracking_GP029287_08016_DUSK_MILLIPEDE_LARGE.csv")

    #draw_hists(glob.glob(LIVING_EARTH_PATH + "*.csv"))
    data = get_movement_type(LIVING_EARTH_FILE)
    export_csv(LIVING_EARTH_FILE, 'D:/Video/Living_Earth/Foraging/tracks1/test.csv',
               ['v_projection_norm', 'v_angle_norm', 'movement_type', 'movement_n'], data)
