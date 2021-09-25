import os
import glob
from math import sqrt, ceil
from textwrap import wrap
import numpy as np
import matplotlib.pyplot as plt

from file.bio import import_tracks_by_frame
from src.file.plain_csv import export_csv
from parameters import *
from src.util import round_significants
from src.video import annotate_video


def get_norm_data(filename):
    data = import_tracks_by_frame(filename)
    v = np.asarray(list(data['v'].values()))
    v_angle = np.asarray(list(data['v_angle'].values()))
    meanl = np.mean(list(data['length_major'].values()))
    v_norm = v / meanl
    angle_norm = abs(v_angle) / VANGLE_NORM
    return v, v_angle, v_norm, angle_norm, meanl


def get_movement_type(filename):
    movement_time = {'': 0, 'brownian': 0, 'levi': 0, 'ballistic': 0}
    data = import_tracks_by_frame(filename)
    dtime = np.mean(np.diff(list(data['time'].values())))
    frames = data['frame']
    v_all = data['v_projection']
    v_angle_all = data['v_angle']
    v_norm_all = {}
    v_angle_norm_all = {}
    movement_type = {}
    movement_n = {}

    meanl = np.mean(list(data['length_major'].values()))
    for frame, v, v_angle in zip(frames.values(), v_all.values(), v_angle_all.values()):
        v_norm = v / meanl
        v_angle_norm = abs(v_angle) / VANGLE_NORM
        if v_norm > 3:
            type = 'ballistic'
            n = 3
        elif abs(v_norm) > 0.6:
            if v_norm > 0.6 and v_angle_norm < 0.05:
                type = 'levi'
                n = 2
            else:
                type = 'brownian'
                n = 1
        else:
            type = ''
            n = 0
        v_norm_all[frame] = v_norm
        v_angle_norm_all[frame] = v_angle_norm
        movement_type[frame] = type
        movement_n[frame] = n
        movement_time[type] += 1
    n = len(v_all)
    for type in movement_time:
        print(f'{type}: {movement_time[type] * dtime:.1f}s {movement_time[type] / n * 100:.1f}%')
    headers = ['v_projection_norm', 'v_angle_norm', 'movement_type', 'movement_n']
    return frames, headers, (v_norm_all, v_angle_norm_all, movement_type, movement_n)


def get_hist_data(data, range):
    hist, bin_edges = np.histogram(data, bins=NBINS, range=(0, range))
    return hist / len(data)


def get_loghist_data(data, power_min, power_max, ax=None, title="", color="#1f77b4"):
    # assume symtric scale - middle bin is 10^0 = 1
    n = len(data)

    #bin_edges = [10 ** ((i - NBINS / 2) * factor) for i in range(NBINS + 1)]
    bin_edges = np.logspace(power_min, power_max, NBINS + 1)

    # manual histogram, ensuring values at (positive) histogram edges are counted
    hist = np.zeros(NBINS)
    factor = (power_max - power_min) / NBINS
    for x in data:
        if x != 0:
            bin = (np.log10(abs(x)) - power_min) / factor
            if bin >= 0:
                # discard low values
                bin = np.clip(int(bin), 0, NBINS)
                hist[bin] += 1
    hist /= n

    #hist, _ = np.histogram(data, bins=bin_edges)

    if ax is not None:
        h, b, _ = ax.hist(data, bins=bin_edges, weights=[1/n]*n, color=color)
        ax.set_xscale('log')
        ax.title.set_text(title)

    return hist, bin_edges


def get_v_hists(filename, ax_v, ax_vangle):
    filetitle = os.path.splitext(os.path.basename(filename))[0].replace("_", " ")
    filetitle_plot = "\n".join(wrap(filetitle, 20))

    v, v_angle, normv, norm_angle, meanl = get_norm_data(filename)

    hist_v, hist_v_bin_edges = get_loghist_data(normv, -2, 2, ax_v, "v " + filetitle_plot, "#1f77b4")
    hist_vangle, hist_vangle_bin_edges = get_loghist_data(norm_angle, -3, 1, ax_vangle, "vangle " + filetitle_plot, "#ff7f0e")

    print(filetitle)
    print(f'v max: {v.max():.3f}\nmean length: {meanl:.3f}\nv max norm: {v.max() / meanl:.3f}')
    print(f'v angle max: {v_angle.max():.3f}\nv angle max norm: {v_angle.max() / VANGLE_NORM:.3f}')
    print()

    return (hist_v, hist_v_bin_edges), (hist_vangle, hist_vangle_bin_edges)


def draw_hists(files, show_pairs=True, show_grid=True):
    v_hists = []
    vangles_hists = []

    if show_grid:
        nfigs = len(files) * 2
        ncols = int(ceil(sqrt(nfigs) / 2) * 2)
        nrows = int(ceil(nfigs / ncols))

        plt.rcParams["axes.titlesize"] = 4
        plt.rcParams.update({'font.size': 4})

        fig, axs0 = plt.subplots(nrows, ncols, dpi=PLOT_DPI)
        axs = np.asarray(axs0).flatten()

        for ax in axs:
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        i = 0
        for filename in files:
            v_hist, vangle_hist = get_v_hists(filename, axs[i], axs[i + 1])
            v_hists.append(v_hist)
            vangles_hists.append(vangle_hist)
            i += 2
        plt.tight_layout()
        plt.show()

    elif show_pairs:
        for filename in files:
            fig, axs0 = plt.subplots(1, 2, dpi=PLOT_DPI)
            axs = np.asarray(axs0).flatten()
            v_hist, vangle_hist = get_v_hists(filename, axs[0], axs[1])
            v_hists.append(v_hist)
            vangles_hists.append(vangle_hist)
        plt.show()

    else:
        for filename in files:
            fig, axs1 = plt.subplots(1, 1, dpi=PLOT_DPI)
            fig, axs2 = plt.subplots(1, 1, dpi=PLOT_DPI)
            v_hist, vangle_hist = get_v_hists(filename, axs1, axs2)
            v_hists.append(v_hist)
            vangles_hists.append(vangle_hist)
            plt.show()

    return v_hists, vangles_hists


def draw_hist(filename):
    fig, axs = plt.subplots(1, 2, dpi=PLOT_DPI)
    get_v_hists(filename, axs[0], axs[1])
    plt.tight_layout()
    plt.show()


def print_hist_values(hist):
    values = hist[0]
    labels = hist[1]
    for i, value in enumerate(values):
        print(f'{round_significants(labels[i], 3):7} ... {round_significants(labels[i+1], 3):7}:'
              f' {value:.3f}')


if __name__ == '__main__':
    #draw_hist(BIO_TRACKING_FILE)
    #draw_hist(LIVING_EARTH_PATH + "tracking_GP029287_08016_DUSK_MILLIPEDE_LARGE.csv")

    #draw_hists(glob.glob(LIVING_EARTH_PATH + "*.csv"))
    v_hists, vangle_hists = draw_hists(glob.glob(LIVING_EARTH_INFILE), show_pairs=False, show_grid=False)

    for v_hist, vangle_hist in zip(v_hists, vangle_hists):
        print('v')
        print_hist_values(v_hist)
        print('v_angle')
        print_hist_values(vangle_hist)

    frames, headers, data = get_movement_type(LIVING_EARTH_INFILE)
    export_csv(LIVING_EARTH_INFILE, LIVING_EARTH_OUTFILE, headers, data)
    annotate_video(LIVING_EARTH_VIDEO_INFILE, LIVING_EARTH_VIDEO_OUTFILE, frames, headers, data)
