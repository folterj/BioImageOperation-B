import csv
import os
import glob
from math import sqrt, ceil
from textwrap import wrap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from file.bio import import_tracks_by_frame
from src.BioData import BioData
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


def get_v_percentiles(filename):
    data = import_tracks_by_frame(filename)
    v = np.asarray(list(data['v'].values()))
    return np.percentile(v, [25, 50, 75])


def get_v_hists(filename, ax_v=None, ax_vangle=None):
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
    parts = os.path.splitext(os.path.basename(input_file))[0].split('_')
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


def to_str(lst):
    return [str(x) for x in lst]


def main_old():
    #draw_hist(BIO_TRACKING_FILE)
    #draw_hist(LIVING_EARTH_PATH + "tracking_GP029287_08016_DUSK_MILLIPEDE_LARGE.csv")

    input_files = glob.glob(TRACKS_PATH2)

    #v_hists, vangle_hists = draw_hists(glob.glob(LIVING_EARTH_INFILE), show_pairs=False, show_grid=False)

    datas = []
    v_hists = []
    vangle_hists = []
    v_percentiles = []
    for filename in input_files:
        datas.append(BioData(filename))
        v_hist, vangle_hist = get_v_hists(filename)
        v_percentile = get_v_percentiles(filename)
        v_hists.append(v_hist)
        vangle_hists.append(vangle_hist)
        v_percentiles.append(v_percentile)

    header_standard = ['ID', 'Date', 'Time', 'Camera']

    header_v = header_standard + [str(x) for x in v_hists[0][1]]
    header_vangle = header_standard + [str(x) for x in vangle_hists[0][1]]
    with open(OUTPUT_PROFILE_V, 'w', newline='') as csvfile_v, \
         open(OUTPUT_PROFILE_VANGLE, 'w', newline='') as csvfile_vangle:

        csvwriter_v = csv.writer(csvfile_v)
        csvwriter_vangle = csv.writer(csvfile_vangle)

        csvwriter_v.writerow(header_v)
        csvwriter_vangle.writerow(header_vangle)

        for input_file, v_hist, vangle_hist in zip(input_files, v_hists, vangle_hists):
            filetitle = os.path.splitext(os.path.basename(input_file))[0].replace("_", " ")
            print(f'v {filetitle}')
            print_hist_values(v_hist)
            print(f'v_angle {filetitle}')
            print_hist_values(vangle_hist)
            print()

            info = extract_info(input_file)
            csvwriter_v.writerow(to_str(info) + to_str(v_hist[0]))
            csvwriter_vangle.writerow(to_str(info) + to_str(vangle_hist[0]))

    #frames, positions, headers, data, movement_time = extract_movement(LIVING_EARTH_INFILE, type='movement_type')
    #export_csv(LIVING_EARTH_INFILE, LIVING_EARTH_OUTFILE, headers, data)
    #annotate_video(LIVING_EARTH_VIDEO_INFILE, LIVING_EARTH_VIDEO_OUTFILE, frames, [positions], [headers], [data])

    all_positions = []
    all_data = []
    all_headers = []
    header = header_standard + ['Appendage Movement [s]', 'Appendage Movement [%]',
                                'Body Movement [s]', 'Body Movement [%]',
                                'Speed 25 Percentile', 'Speed 50 Percentile', 'Speed 75 Percentile']

    with open(OUTPUT_DATAFRAME, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        for input_file, v_percentile in zip(input_files, v_percentiles):
            filetitle = os.path.splitext(os.path.basename(input_file))[0].replace("_", " ")
            print(filetitle)

            info = extract_info(input_file)
            frames, positions, headers, data, movement_time = extract_movement(input_file, output_type='activity_type')

            #export_csv(input_file, output_file, headers, data)
            csvwriter_v.writerow(to_str(info) + to_str(v_percentile))

            headers1 = [headers[-1]]
            data1 = [data[-1]]
            all_positions.append(positions)
            all_headers.append(headers1)
            all_data.append(data1)
            print()

    #annotate_video(LIVING_EARTH_VIDEO_INFILE, LIVING_EARTH_VIDEO_OUTFILE, frames, all_positions, all_headers, all_data)


if __name__ == '__main__':
    #draw_hist(BIO_TRACKING_FILE)
    #draw_hist(LIVING_EARTH_PATH + "tracking_GP029287_08016_DUSK_MILLIPEDE_LARGE.csv")

    input_files = glob.glob(TRACKS_PATH2)

    #v_hists, vangle_hists = draw_hists(glob.glob(LIVING_EARTH_INFILE), show_pairs=False, show_grid=False)

    datas = [BioData(filename) for filename in tqdm(input_files)]

    header_standard = ['ID', 'Date', 'Time', 'Camera']

    header_v = header_standard + [str(x) for x in datas[0].v_hist[1]]
    header_vangle = header_standard + [str(x) for x in datas[0].vangle_hist[1]]
    with open(OUTPUT_PROFILE_V, 'w', newline='') as csvfile_v, \
         open(OUTPUT_PROFILE_VANGLE, 'w', newline='') as csvfile_vangle:

        csvwriter_v = csv.writer(csvfile_v)
        csvwriter_vangle = csv.writer(csvfile_vangle)

        csvwriter_v.writerow(header_v)
        csvwriter_vangle.writerow(header_vangle)

        for data in datas:
            csvwriter_v.writerow(to_str(data.info) + to_str(data.v_hist[0]))
            csvwriter_vangle.writerow(to_str(data.info) + to_str(data.vangle_hist[0]))

    #frames, positions, headers, data, movement_time = extract_movement(LIVING_EARTH_INFILE, type='movement_type')
    #export_csv(LIVING_EARTH_INFILE, LIVING_EARTH_OUTFILE, headers, data)
    #annotate_video(LIVING_EARTH_VIDEO_INFILE, LIVING_EARTH_VIDEO_OUTFILE, frames, [positions], [headers], [data])

    all_positions = []
    all_data = []
    all_headers = []
    header = header_standard + ['Appendage Movement [s]', 'Appendage Movement [%]',
                                'Body Movement [s]', 'Body Movement [%]',
                                'Speed 25 Percentile', 'Speed 50 Percentile', 'Speed 75 Percentile']

    with open(OUTPUT_DATAFRAME, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        for data in datas:
            data.classify_movement(output_type='activity_type')
            output = data.info
            output.append(data.get_movement_time('appendages'))
            output.append(data.get_movement_fraction('appendages'))
            output.append(data.get_movement_time('moving'))
            output.append(data.get_movement_fraction('moving'))
            output.extend(data.v_percentiles)
            #export_csv(input_file, output_file, headers, data)
            csvwriter.writerow(output)

    #annotate_video(LIVING_EARTH_VIDEO_INFILE, LIVING_EARTH_VIDEO_OUTFILE, frames, all_positions, all_headers, all_data)
