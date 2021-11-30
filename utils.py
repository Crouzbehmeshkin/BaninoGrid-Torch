import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import os
from os import listdir
from os.path import isfile, join
from matplotlib.backends.backend_pdf import PdfPages

import torch

import ensembles


def load_datadic_from_tfrecords(path, _Datasets, env_name: str, feature_map: dict):
    ds_info = _Datasets[env_name]

    tfrecord_files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)

    def _parse_function(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_map)

    parsed_dataset = raw_dataset.map(_parse_function)

    datadic = dict()
    for key in feature_map.keys():
        datadic[key] = list()

    for parsed_record in parsed_dataset:
        datadic['init_pos'].append(parsed_record['init_pos'].numpy())
        datadic['init_hd'].append(parsed_record['init_hd'].numpy())
        datadic['ego_vel'].append(parsed_record['ego_vel'].numpy())
        datadic['target_pos'].append(parsed_record['target_pos'].numpy())
        datadic['target_hd'].append(parsed_record['target_hd'].numpy())
    return datadic


def get_place_cell_ensembles(
        env_size, neurons_seed, targets_type, lstm_init_type, n_pc, pc_scale, device='cpu'):
    """Create the ensembles for the Place cells."""
    place_cell_ensembles = [
        ensembles.PlaceCellEnsemble(
            n,
            device,
            stdev=s,
            pos_min=env_size / 2.0,
            pos_max=env_size / 0.2,
            seed=neurons_seed,
            soft_targets=targets_type,
            soft_init=lstm_init_type)
        for n, s in zip(n_pc, pc_scale)
    ]
    return place_cell_ensembles


def get_head_direction_ensembles(
        neurons_seed, targets_type, lstm_init_type, n_hdc, hdc_concentration, device='cpu'):
    """Create the ensembles for the Head direction cells."""
    head_direction_ensembles = [
        ensembles.HeadDirectionCellEnsemble(
            n,
            device,
            concentration=con,
            seed=neurons_seed,
            soft_targets=targets_type,
            soft_init=lstm_init_type)
        for n, con in zip(n_hdc, hdc_concentration)
    ]
    return head_direction_ensembles


def encode_initial_conditions(
        init_pos, init_hd, place_cell_ensembles, head_direction_ensembles):
    initial_conds = []
    for ens in place_cell_ensembles:
        cond_i = ens.get_init(init_pos[:, None, :])
        initial_conds.append(
            torch.squeeze(ens.get_init(init_pos[:, None, :])))
    for ens in head_direction_ensembles:
        cond_i = ens.get_init(init_hd[:, None, :])
        initial_conds.append(
            torch.squeeze(ens.get_init(init_hd[:, None, :])))
    return initial_conds


def encode_targets(
        target_pos, target_hd, place_cell_ensembles, head_direction_ensembles):
    ensembles_targets = []
    for ens in place_cell_ensembles:
        ensembles_targets.append(ens.get_targets(target_pos))
    for ens in head_direction_ensembles:
        ensembles_targets.append(ens.get_targets(target_hd))
    return ensembles_targets


def clip_all_gradients(g, var, limit):
    return (torch.clamp(g, -limit, limit), var)


def clip_bottleneck_gradient(g, var, limit):
    if ('bottleneck' in var.name) or ('pc_logits' in var.name):
        return (torch.clamp(g, -limit, limit), var)
    else:
        return (g, var)


def no_clipping(g, var):
    return (g, var)


def concat_dict(acc, new_data):
    """Dictionaty Concatentaion Function"""
    def to_array(kk):
        if isinstance(kk, np.ndarray):
            return kk
        else:
            return np.asarray([kk])

    for k, v in new_data.items():
        if isinstance(v, dict):
            if k in acc:
                acc[k] = concat_dict(acc[k], v)
            else:
                acc[k] = concat_dict(dict(), v)
        else:
            v = to_array(v)
            if k in acc:
                acc[k] = np.concatenate([acc[k], v])
            else:
                acc[k] = np.copy(v)
    return acc


def get_scores_and_plot(scorer,
                        data_abs_xy,
                        activations,
                        directory,
                        filename,
                        plot_graphs=True,
                        nbins=20,
                        cm="jet",
                        sort_by_score_60=True):
    """Plotting Function"""

    #Concatenate all trajectories
    xy = data_abs_xy.reshape(-1, data_abs_xy.shape[-1])
    act = activations.reshape(-1, activations.shape[-1])
    n_units = act.shape[1]

    #Get the rate-map for each unit
    s = [scorer.calculate_ratemap(xy[:, 0], xy[:, 1], act[:, i])
        for i in range(n_units)]

    #Get the scores
    score_60, score_90, max_60_mask, max_90_mask, sac = zip(
        *[scorer.get_scores(rate_map) for rate_map in s])

    #Seperations
    # sperations = map(np.mean, max_60_mask)

    #Sort by score if desired
    if sort_by_score_60:
        ordering = np.argsort(-np.array(score_60))
    else:
        ordering = range(n_units)

    #Plot
    cols = 16
    rows = int(np.ceil(n_units / cols))
    fig = plt.figure(figsize=(24, rows*4))
    for i in range(n_units):
        rf = plt.subplot(rows*2, cols, i+1)
        acr = plt.subplot(rows*2, cols, n_units + i + 1)
        if i < n_units:
            index = ordering[i]
            title = "%d (%.2f)" % (index, score_60(index))
            #Plotting the activation maps
            scorer.plot_ratemap(s[index], ax=rf, title=title, cmap=cm)
            #Plotting the autocorrelation of the activation maps
            scorer.plot_sac(
                sac[index],
                mask_params=max_60_mask[index],
                ax=acr,
                title=title,
                cmap=cm)
    #Save
    if plot_graphs:
        if not os.path.exists(directory):
            os.makedirs(directory)
        with PdfPages(os.path.join(directory, filename), 'w') as f:
            plt.savefig(f, format='pdf')
        plt.close(fig)
    return (np.asarray(score_60), np.asarray(score_90),
            np.asarray(map(np.mean, max_60_mask)),
            np.asarray(map(np.mean, max_90_mask)))