import csv
import glob
import os
from math import sqrt, ceil
from textwrap import wrap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.VideoInfo import VideoInfos
from src.file.bio import import_tracks_by_frame
from src.BioFeatures import BioFeatures
from src.parameters import PROFILE_HIST_BINS, VANGLE_NORM, PLOT_DPI
from src.util import round_significants, get_filetitle, list_to_str


def get_norm_data(filename):
    data = import_tracks_by_frame(filename)
    v = np.asarray(list(data['v'].values()))
    v_angle = np.asarray(list(data['v_angle'].values()))
    meanl = np.mean(list(data['length_major'].values()))
    v_norm = v / meanl
    angle_norm = abs(v_angle) / VANGLE_NORM
    return v, v_angle, v_norm, angle_norm, meanl


def extract_movement(filename, output_type='movement_type'):
    if output_type == 'movement_type':
        movement_time = {'': 0, 'brownian': 0, 'levi': 0, 'ballistic': 0}
    elif output_type == 'activity_type':
        movement_time = {'': 0, 'appendages': 0, 'moving': 0}

    data = import_tracks_by_frame(filename)
    dtime = np.mean(np.diff(list(data['time'].values())))
    frames = data['frame']
    positions = {frame: (x, y) for frame, x, y in zip(frames.values(), data['x'].values(), data['y'].values())}
    length_major = data['length_major1']
    length_minor = data['length_minor1']
    meanl = np.mean(list(length_major.values()))
    meanw = np.mean(list(length_minor.values()))
    v_all = data['v_projection']
    v_angle_all = data['v_angle']
    v_norm_all = {}
    v_angle_norm_all = {}
    length_major_delta_all = {}
    length_minor_delta_all = {}
    movement_type = {}

    lenl0 = meanl
    lenw0 = meanw
    for frame, v, v_angle, lenl, lenw in zip(frames.values(), v_all.values(), v_angle_all.values(), length_major.values(), length_minor.values()):
        v_norm = v / meanl
        length_major_delta = abs(lenl - lenl0) / meanl
        length_minor_delta = abs(lenw - lenw0) / meanw
        v_angle_norm = abs(v_angle) / VANGLE_NORM
        if output_type == 'movement_type':

            if v_norm > 3:
                type = 'ballistic'
            elif abs(v_norm) > 0.6:
                if v_norm > 0.6 and v_angle_norm < 0.05:
                    type = 'levi'
                else:
                    type = 'brownian'
            else:
                type = ''

        elif output_type == 'activity_type':

            if v_norm > 0.2:
                type = 'moving'
            elif length_major_delta + length_minor_delta > 0.01:
                type = 'appendages'
            else:
                type = ''

        lenl0 = lenl
        lenw0 = lenw

        v_norm_all[frame] = v_norm
        v_angle_norm_all[frame] = v_angle_norm
        length_major_delta_all[frame] = length_major_delta
        length_minor_delta_all[frame] = length_minor_delta
        movement_type[frame] = type
        movement_time[type] += 1
    n = len(v_all)
    for type in movement_time:
        print(f'{type}: {movement_time[type] * dtime:.1f}s {movement_time[type] / n * 100:.1f}%')
    headers = ['v_projection_norm', 'v_angle_norm', 'length_major_delta', 'length_minor_delta', 'movement_type']
    return frames, positions,\
           headers, (v_norm_all, v_angle_norm_all, length_major_delta_all, length_minor_delta_all, movement_type),\
           movement_time


def get_hist_data(data, range):
    hist, bin_edges = np.histogram(data, bins=PROFILE_HIST_BINS, range=(0, range))
    return hist / len(data)


def get_loghist_data(data, power_min, power_max, ax=None, title="", color="#1f77b4"):
    # assume symtric scale - middle bin is 10^0 = 1
    n = len(data)

    #bin_edges = [10 ** ((i - NBINS / 2) * factor) for i in range(NBINS + 1)]
    bin_edges = np.logspace(power_min, power_max, PROFILE_HIST_BINS + 1)

    # manual histogram, ensuring values at (positive) histogram edges are counted
    hist = np.zeros(PROFILE_HIST_BINS)
    factor = (power_max - power_min) / PROFILE_HIST_BINS
    for x in data:
        if x != 0:
            bin = (np.log10(abs(x)) - power_min) / factor
            if bin >= 0:
                # discard low values
                bin = np.clip(int(bin), 0, PROFILE_HIST_BINS)
                hist[bin] += 1
    hist /= n

    #hist, _ = np.histogram(data, bins=bin_edges)

    if ax is not None:
        h, b, _ = ax.hist(data, bins=bin_edges, weights=[1/n]*n, color=color)
        ax.set_xscale('log')
        ax.title.set_text(title)

    return hist, bin_edges


def get_v_percentiles(filename):
    data = import_tracks_by_frame(filename)
    v = np.asarray(list(data['v'].values()))
    return np.percentile(v, [25, 50, 75])


def get_v_hists(filename, ax_v=None, ax_vangle=None):
    filetitle = get_filetitle(filename).replace("_", " ")
    filetitle_plot = "\n".join(wrap(filetitle, 20))

    v, v_angle, normv, norm_angle, meanl = get_norm_data(filename)

    hist_v, hist_v_bin_edges = get_loghist_data(normv, -2, 2, ax_v, "v " + filetitle_plot, "#1f77b4")
    hist_vangle, hist_vangle_bin_edges = get_loghist_data(norm_angle, -3, 1, ax_vangle, "vangle " + filetitle_plot, "#ff7f0e")

    print(filetitle)
    print(f'v max: {v.max():.3f}\nmean length: {meanl:.3f}\nv max norm: {v.max() / meanl:.3f}')
    print(f'v angle max: {v_angle.max():.3f}\nv angle max norm: {v_angle.max() / VANGLE_NORM:.3f}')
    print()

    return (hist_v, hist_v_bin_edges), (hist_vangle, hist_vangle_bin_edges)


def draw_hists(filenames, show_pairs=True, show_grid=True):
    v_hists = []
    vangle_hists = []

    if show_grid:
        nfigs = len(filenames) * 2
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
        for filename in filenames:
            v_hist, vangle_hist = get_v_hists(filename, axs[i], axs[i + 1])
            v_hists.append(v_hist)
            vangle_hists.append(vangle_hist)
            i += 2
        plt.tight_layout()
        plt.show()

    elif show_pairs:
        for filename in filenames:
            fig, axs0 = plt.subplots(1, 2, dpi=PLOT_DPI)
            axs = np.asarray(axs0).flatten()
            v_hist, vangle_hist = get_v_hists(filename, axs[0], axs[1])
            v_hists.append(v_hist)
            vangle_hists.append(vangle_hist)
        plt.show()

    else:
        for filename in filenames:
            fig, axs1 = plt.subplots(1, 1, dpi=PLOT_DPI)
            fig, axs2 = plt.subplots(1, 1, dpi=PLOT_DPI)
            v_hist, vangle_hist = get_v_hists(filename, axs1, axs2)
            v_hists.append(v_hist)
            vangle_hists.append(vangle_hist)
            plt.show()

    return v_hists, vangle_hists


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


def extract_info(input_file):
    parts = get_filetitle(input_file).split('_')
    i = 0
    if not parts[i].isnumeric():
        i += 1
    date = parts[i]
    time = parts[i + 1].replace('-', ':')
    id = parts[-1]

    camera = 0
    s = input_file.lower().find('cam')
    if s >= 0:
        while s < len(input_file) and not input_file[s].isnumeric():
            s += 1
        e = s
        while e < len(input_file) and input_file[e].isnumeric():
            e += 1
        if e > s:
            camera = int(input_file[s:e])

    return [id, date, time, camera]


def extract_activity_features(params):
    base_dir = params['base_dir']
    input_files = glob.glob(os.path.join(base_dir, params['tracks_relabel_dir'], '*'))
    video_input_path = os.path.join(base_dir, params['video_input_path'])
    video_files = sorted(glob.glob(video_input_path))

    profile_v_output_filename = os.path.join(base_dir, params['profile_v_output_filename'])
    profile_vangle_output_filename = os.path.join(base_dir, params['profile_vangle_output_filename'])
    dataframe_output_filename = os.path.join(base_dir, params['dataframe_output_filename'])
    video_infos = VideoInfos(video_files)

    print('Reading & processing input files')
    datas = [BioFeatures(filename) for filename in tqdm(input_files)]

    print('Writing output files')
    header_standard = ['ID', 'Date', 'Time', 'Camera']
    header_v = header_standard + [str(x) for x in datas[0].v_hist[1]]
    header_vangle = header_standard + [str(x) for x in datas[0].vangle_hist[1]]
    with open(profile_v_output_filename, 'w', newline='') as csvfile_v, \
         open(profile_vangle_output_filename, 'w', newline='') as csvfile_vangle:

        csvwriter_v = csv.writer(csvfile_v)
        csvwriter_vangle = csv.writer(csvfile_vangle)

        csvwriter_v.writerow(header_v)
        csvwriter_vangle.writerow(header_vangle)

        for data in datas:
            csvwriter_v.writerow(list_to_str(data.info) + list_to_str(data.v_hist[0]))
            csvwriter_vangle.writerow(list_to_str(data.info) + list_to_str(data.vangle_hist[0]))

    header = header_standard + ['Appendage Movement [s]', 'Appendage Movement [%]',
                                'Body Movement [s]', 'Body Movement [%]',
                                'Speed 25 Percentile', 'Speed 50 Percentile', 'Speed 75 Percentile']

    with open(dataframe_output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        for data in datas:
            video_info = video_infos.find_match(data.filetitle)
            data.classify_movement(output_type='activity_type')
            output = data.info
            output.append(data.get_movement_time('appendages'))
            output.append(data.get_movement_fraction('appendages', video_info.total_frames))
            output.append(data.get_movement_time('moving'))
            output.append(data.get_movement_fraction('moving', video_info.total_frames))
            output.extend(data.v_percentiles)
            csvwriter.writerow(output)

    print('Done')
