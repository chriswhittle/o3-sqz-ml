'''
Produce plots for paper. Run as

`python plotting.py config.yaml`

to run all plotting routines or

`python plotting.py config.yaml PLOT_TYPE`

to run a specific plotting routine where `PLOT_TYPE` is one of 'summary', 
'timeseries', 'genetic', 'gradient' or 'sobol'.
'''

import sys
from pathlib import Path
import logging
import yaml

import numpy as np
from scipy.stats import mode
import pandas as pd

import pathlib
curdir = pathlib.Path(__file__).parent.resolve()

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use(curdir / 'paper.mplstyle')
import seaborn as sns

from train_nn import SQZModel
from genetic import bitmask_str2np

TIME_STEP = 60

SMALL_FONT_SIZE = 8

def make_path(path_string):
    path = Path(path_string)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path

def remove_outliers(data, m=4.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev
    return data[s<m]

def fill_nans(df):
    # create array of contiguous timesteps
    full_times = np.arange(
        df.index[0], df.index[-1], TIME_STEP
    )

    # resample given dataframe with NaNs in gaps
    return df.reindex(full_times, fill_value=np.nan)

def summary(**config):
    sub_config = config['plotting']['summary']
    figure_path = make_path(sub_config['figure_path'])

    # read in O3 squeezing data and pick out nominal BLRMS
    data = pd.read_csv(config['processed_path'], index_col='gps_time')
    f = config['nominal_blrms_lims']
    sqz_levels = data[f'SQZ_dB {f[0]}-{f[1]}Hz']

    # find gap between O3a and O3b by finding biggest gap in data
    gap_ind = np.diff(sqz_levels.index).argmax()
    gap_gps_start = sqz_levels.index[gap_ind]
    gap_gps_end = sqz_levels.index[gap_ind+1]
    sqz_levels[gap_gps_start + TIME_STEP] = np.nan
    sqz_levels[gap_gps_end - TIME_STEP] = np.nan
    sqz_levels = sqz_levels.sort_index()

    # print mean, std and 90% CI of sqz level
    print(f'{sqz_levels.mean():.2f} dB')
    print(f'{sqz_levels.std():.2f} dB')
    print(np.nanpercentile(sqz_levels, [5, 95]))

    # make plot
    fig = plt.figure(figsize=(3.5, 2.25))
    ax1, ax2 = fig.subplots(1, 2, sharey=True, width_ratios=[7, 1])

    # fill in gap between O3a and O3b
    ax1.axvspan(
        gap_gps_start,
        gap_gps_end,
        color='gray',
        alpha=0.25
    )

    # plot squeezing level time series
    ax1.plot(sqz_levels)

    # get mean squeezing level to draw on plot
    mean_sqz = sqz_levels.mean()
    mean_style = {'ls': '--', 'c': 'k', 'alpha': 0.55}
    ax1.axhline(y=mean_sqz, **mean_style)

    # text labels for O3a and O3b
    for l, x in [('O3a', 0.18), ('O3b', 0.75)]:
        plt.text(x, 0.16, l, transform=ax1.transAxes)

    ax1.set_xticks(
        [sqz_levels.index[0], sqz_levels.index[-1]],
        labels=['April 2019', 'March 2020']
    )
    ax1.set_ylim([0, 3.4])
    ax1.set_ylabel('Squeezing level [dB]')
    ax1.set_xlabel('Time')

    # plot squeezing level histogram
    ax2.hist(sqz_levels, bins=sub_config['bins'], orientation='horizontal')
    ax2.axhline(y=mean_sqz, **mean_style)
    ax2.set_xticks([])
    ax2.set_xlim(0, ax2.get_xlim()[1]*1.08)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(figure_path)
    plt.close()

def timeseries(**config):
    config = config.copy()
    sub_config = config['plotting']['timeseries']
    figure_path = make_path(sub_config['figure_path'])

    ws = sub_config['window_size']

    # perform training with different model architectures
    architectures = sub_config['neural_networks']
    models = [None] * len(architectures)
    base_architecture = config['neural_network'].copy()
    config['cut_channels'] = config['plotting']['cut_channels']
    for i, architecture in enumerate(architectures):
        cur_architecture = base_architecture.copy()
        cur_architecture.update(**architecture['options'])
        config['neural_network'] = cur_architecture

        print(config['neural_network'])

        models[i] = SQZModel(
            architecture['path'],
            sub_config['sub_start_gps'],
            sub_config['sub_start_gps'] + sub_config['sub_duration'],
            **config
        )

    plt.figure()
    for arch, model in zip(architectures, models):
        l = plt.plot(model.loss_histories[0][:,0], label=arch['label'])
        plt.plot(model.loss_histories[0][:,1], c=l[0].get_color(), ls='--')
    plt.legend()
    plt.savefig(str(figure_path).replace('timeseries', 'timeseries_losses'))
    plt.close()

    # rescale time axis to days
    t_rescale = 3600*24

    plt.figure(figsize=(4, 4.8))
    ax = plt.gca()
    # plot raw sqz levels
    val_labels = models[0].raw_data.validation_labels
    val_labels = fill_nans(val_labels)
    plt.plot(
        (val_labels.index - val_labels.index[0])/t_rescale,
        val_labels.values,
        c='gray',
        alpha=0.7
    )

    # plot rolling average of sqz levels
    smooth_labels = val_labels.rolling(window=ws).mean().dropna()
    smooth_labels = fill_nans(smooth_labels)
    plt.plot(
        (smooth_labels.index - smooth_labels.index[0])/t_rescale,
        smooth_labels.values,
        c='k',
        alpha=1,
        label='True squeezing'
    )

    for arch, model in zip(architectures, models):
        # compute rolling average of sqz estimates
        predictions = pd.DataFrame(model.estimate_sqz(
            model.raw_data.validation_features
        ).flatten())
        predictions.index = model.raw_data.validation_features.index
        predictions = fill_nans(predictions.rolling(window=ws).mean().dropna())

        plt.plot(
            (predictions.index - predictions.index[0])/t_rescale,
            predictions.values,
            label=arch['label'],
            alpha=0.9
        )
    plt.xlabel('Time [days]')
    plt.ylabel('Squeezing level [dB]')
    plt.ylim(sub_config['sqz_limits'])
    
    box = ax.get_position()
    ax.set_position([box.x0 + box.width * 0.1, box.y0,
                    box.width * 0.9, box.height * 0.7])
    plt.legend(bbox_to_anchor=(0.05, 1.05))

    # plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

def genetic(**config):
    # get config options for genetic plot
    sub_config = config['plotting']['genetic']
    figure_path = make_path(sub_config['figure_path'])

    # get number of members in each generation
    G = config['genetic']['pop_size']
    
    # get total of possible channels to include
    num_features = len(config['channels']) - len(config['cut_channels'])
    channels = [c for c in list(config['channels'].keys()) 
                if c not in config['cut_channels']]

    # read in genetic data
    inclusions = []
    with open(config['genetic_path']) as genetic_file:
        data = np.loadtxt(genetic_file)
    mask_strings = data[:,:G]
    losses = data[:,G:]

    # calculate inclusion fractions as a function of time
    for i in range(mask_strings.shape[0]):
        gen_masks = mask_strings[i,:]
        generation = np.array(list(
            [bitmask_str2np(g, num_features) for g in gen_masks]
        ))
        inclusion = generation.mean(axis=0)
        inclusions += [inclusion]
    inclusions = np.array(inclusions[:-1])

    # remove rows with incomplete calculation
    losses = losses[~(losses==-1).any(axis=1)]
    epochs = np.arange(losses.shape[0])

    # print best channel set found so far
    optimal_ind = np.unravel_index(losses.argmin(), losses.shape)
    optimal_bitmask = bitmask_str2np(mask_strings[optimal_ind], num_features)
    optimal_channels = [
        c for i, c in enumerate(channels) if optimal_bitmask[i]==1
    ]
    optimal_cuts = [
        c for i, c in enumerate(channels) if optimal_bitmask[i]==0
    ]
    logging.warning(f'Optimal channel subset: {optimal_channels}')
    logging.warning(f'Cut channels: {optimal_cuts}')

    # calculate per-generation loss metrics as a function of epoch
    mean_loss = losses.mean(axis=1)
    min_loss = losses.min(axis=1)
    max_loss = losses.max(axis=1)

    # calculate running best loss
    cummin_loss = pd.Series(min_loss).cummin()

    # calculate cumulative inclusion
    sum_inclusions = inclusions.cumsum(axis=0)
    counts = np.tile(np.arange(losses.shape[0])[:, np.newaxis] + 1,
                        (1, num_features))
    cum_inclusions = sum_inclusions / counts
    
    # save channels with cumulative inclusion to text file
    with open(sub_config['table_path'], 'w') as table_file:
        table_file.write('\n'.join(
            [
                f'{c}\t{config["channels"][c]}\t{cum_inclusions[-1,i]:.2f}\t{optimal_bitmask[i]}'
                for i, c in enumerate(channels)
            ]
        ))

    # make plot
    plt.figure(figsize=(3.5, 3.2))

    # loss subplot
    ax = plt.subplot(211)
    plt.semilogx(mean_loss, label='Mean loss')
    plt.fill_between(epochs, min_loss, max_loss, alpha=0.1)
    plt.semilogx(cummin_loss, label='Best loss')

    plt.ylabel('Loss')
    plt.legend(loc='best')

    ax.set_xticklabels([])

    # inclusion fraction plot
    plt.subplot(212)
    for i, c in enumerate(channels):
        if c in sub_config['highlight_channels']:
            plt.semilogx(cum_inclusions[:,i], label=config['channels'][c])
        else:
            plt.semilogx(cum_inclusions[:,i], c='gray', alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Cumulative inclusion')
    plt.ylim([0, 1])
    
    plt.legend(
        loc='lower left', ncol=2, fontsize=8, framealpha=0.5
    )

    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(figure_path)
    plt.close()

def cluster(step_f=2e-3, **config):
    config = config.copy()

    # grab config data
    sub_config = config['plotting']['cluster']
    figure_path = make_path(sub_config['figure_path'])

    config['cut_channels'] = config['plotting']['cut_channels']

    # dummy model to get full dataset
    config['cluster_count'] = sub_config['cluster_count']
    full_model = SQZModel(
        Path(sub_config['model_path']),
        config['start_gps'],
        config['end_gps'],
        val_fraction=4e-6,
        **config
    )

    # do plotting
    plt.figure(figsize=(3.75, 4.5))

    # plot full cluster time series and one zoomed in on a region of interest
    axes = [None]*2
    for zoomed in [False, True]:
        axes[zoomed] = plt.subplot(4, 1, zoomed+1)

        # take times and cluster values from dummy model
        times = full_model.raw_data.training_features.index
        clusters = full_model.training_clusters

        # zoom in on specified time interval for zoomed subplot
        if zoomed:
            inds = np.logical_and(
                times>sub_config['zoom_start_gps'],
                times<sub_config['zoom_start_gps'] + sub_config['zoom_duration']
            )
            times = times[inds]
            clusters = clusters[inds]

        # go through and pick most frequent label for each segment
        # get time step from fraction specified, rounding to nearest sample
        step = (times[-1] - times[0]) * step_f
        step = step - step%TIME_STEP

        for seg_start in np.arange(times[0], times[-1], step):
            # get most common cluster
            most_common_cluster = mode(clusters[
                np.logical_and(times > seg_start, times < seg_start + step)
            ]).mode
            
            if len(most_common_cluster) > 0:
                most_common_cluster = most_common_cluster[0]

                # plot colored region
                plt.axvspan(
                    seg_start, 
                    seg_start + step,
                    color = f'C{most_common_cluster}',
                    alpha = 0.4,
                    label = f'Cluster \\#{most_common_cluster}'
                )

        # no ticks for cluster time series
        plt.tick_params(
            axis='y', which='both', left=False, right=False, labelleft=False
        )

        # trick to invert connector lines in matplotlib indicate_inset_zoom 
        plt.ylim([1, 0] if zoomed else [0, 1])

        if zoomed:
            # zoomed time axis
            axes[zoomed].tick_params(labelbottom=False) 
        else:
            # coarse time axis
            plt.xlabel('Time')
            axes[zoomed].set_xticks(
                [times[0], times[-1]],
                labels=['April 2019', 'March 2020']
            )
            axes[zoomed].xaxis.tick_top()
            axes[zoomed].xaxis.set_label_position('top') 

    # draw connector lines between full and zoomed time series
    connector_lw = 1.5
    _, lines = axes[0].indicate_inset_zoom(
        axes[1], lw=connector_lw, alpha=1, edgecolor='k'
    )
    for line in lines:
        line.set_linewidth(connector_lw)

    # draw time series of channels
    ts_ax = plt.subplot(4, 1, 3)
    for c in sub_config['zoom_channels']:
        channel_values = full_model.detrended_data.training_features[inds][c]

        # remove outliers (removes glitches when e.g. relocking)
        channel_values = remove_outliers(channel_values)

        plt.plot(fill_nans(channel_values), label=c, alpha=0.75)
    
    # time series plotting style
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.9))
    plt.xlabel('GPS Time [s]')
    plt.ylabel('Channel values [a.u.]')

    # move time series plot up against zoomed cluster time series
    pos = ts_ax.get_position()
    zm_pos = axes[1].get_position()
    ts_ax.set_position([pos.x0, pos.y0, pos.x1 - pos.x0, zm_pos.y0 - pos.y0])

    plt.savefig(figure_path)
    plt.close()

def gradient(plot_sign=False, **config):
    config = config.copy()

    # grab config data
    sub_config = config['plotting']['gradient']
    cluster_config = config['plotting']['cluster']
    figure_path = make_path(sub_config['figure_path'])

    config['cut_channels'] = config['plotting']['cut_channels']

    # which cluster ordering to sort by
    sort_by = sub_config['sort_by_cluster']
    
    # train models
    models = [None]*len(sub_config['sub_start_gps'])
    for i, start in enumerate(sub_config['sub_start_gps']):
        models[i] = SQZModel(
            Path(sub_config['model_path']) / str(start),
            start,
            start + sub_config['sub_duration'],
            **config
        )

    # dummy model to get cluster centroids
    config['cluster_count'] = cluster_config['cluster_count']
    full_model = SQZModel(
        Path(cluster_config['model_path']),
        config['start_gps'],
        config['end_gps'],
        val_fraction=4e-6,
        **config
    )

    # plot colors
    cmap = plt.get_cmap('tab20')

    # average over different time intervals
    for i, model in enumerate(models):
        # count number of samples in each cluster for this time interval
        tr_clusters = full_model.kmeans.predict(
                model.detrended_data.training_features
        )
        cluster_counts = np.bincount(tr_clusters)
        logging.warning(
            f'Model {i} ({sub_config["sub_start_gps"][i]}) cluster counts: {cluster_counts}'
        )

        # plot time series of validation labels for this time interval
        plt.figure()
        val_labels = model.raw_data.validation_labels
        val_features = model.raw_data.validation_features
        plt.subplot(211)
        plt.plot(val_labels)
        plt.plot(val_labels.index, model.estimate_sqz(val_features))
        plt.subplot(212)
        plt.plot(model.detrended_data.validation_features['uSeism.-X 300M-1'])
        plt.plot(model.detrended_data.validation_features['uSeism.-Y 100M-300M'])
        plt.plot(model.detrended_data.validation_features['ZM1 P (RMS)'])
        plt.plot(model.detrended_data.validation_features['INP2 P (RMS)'])
        plt.savefig(sub_config['val_figure_path'].format(
            sub_config['sub_start_gps'][i]
        ))
        plt.close()

        # compute gradients at each cluster
        dfs = [None]*full_model.clusters.shape[0]
        for j, cluster in full_model.clusters.iterrows():
            # convert cluster to raw data coordinates
            raw_cluster = full_model.features_detrender.retrend(cluster)

            # compute gradient
            gradients_df = model.gradient(point=raw_cluster, sort=sort_by is None)[0]

            # differentiate between positive and negative gradients for plotting
            grads = gradients_df['Gradient']
            gradients_df['abs'] = np.abs(grads)
            gradients_df['$+$'] = grads * (grads>0)
            gradients_df['$-$'] = -1 * grads * (grads<0)

            dfs[j] = gradients_df
        
        if sort_by is not None:
            sort_inds = np.argsort(dfs[sort_by]['abs'])[::-1]
        
        for j, gradients_df in enumerate(dfs):
            if sort_by is not None:
                gradients_df = gradients_df.iloc[sort_inds]

            plt.figure(figsize=(3.75, 3.25))

            # bar plot of gradients
            if plot_sign:
                cols = [cmap(1), cmap(3)]
                labels = ['$+$', '$-$']
            else:
                cols = [cmap(1)]
                labels = ['abs']
            for color, label in zip(cols, labels):
                barplot = sns.barplot(
                    x = 'Channel',
                    y = label,
                    color = color,
                    data = gradients_df,
                    label = label
                )
            
            # plot styling
            if plot_sign:
                plt.legend()
            barplot.set_xticklabels(
                barplot.get_xticklabels(),
                rotation=90,
                fontsize=SMALL_FONT_SIZE
            )
            barplot.set(
                ylabel=r'$\Big| \frac{\mathrm{dSQZ}}{\mathrm{d} x} \sigma_x \Big|$ [dB]'
            )
            plt.tight_layout()

            plt.savefig(str(figure_path).format(
                sub_config['sub_start_gps'][i], j
            ))
            plt.close()

def sobol(**config):    
    config = config.copy()

    # grab config data
    sub_config = config['plotting']['sobol']
    figure_path = make_path(sub_config['figure_path'])

    config['cut_channels'] = config['plotting']['cut_channels']

    plot_errors = sub_config['plot_errors']
    
    # train models
    models = [None]*len(sub_config['sub_start_gps'])
    for i, start in enumerate(sub_config['sub_start_gps']):
        models[i] = SQZModel(
            Path(sub_config['model_path']) / str(start),
            start,
            start + sub_config['sub_duration'],
            **config
        )

    # compute/load Sobol indices
    Sis = [None]*len(sub_config['sub_start_gps'])
    for i, model in enumerate(models):
        Sis[i] = model.sobol(N=sub_config['samples'], save=True)
    
    # average over Sobol indices from each time interval
    Si = {}
    for key in Sis[0].keys():
        # new matrix for particular Sobol index
        Si[key] = np.zeros(Sis[0][key].shape)

        # sum over different time intervals
        for i, sub_Si in enumerate(Sis):
            Si[key] += sub_Si[key]
        
        # divide by number of time intervals
        Si[key] = Si[key]/len(Sis)

    # sort columns by the total Sobol indices
    columns = models[0].feature_columns
    n_columns = len(columns)
    inds = np.argsort(Si['ST'])[::-1]

    sorted_columns = columns[inds]

    # reflect S2 matrix about diagonal
    Si2s = {}
    for i, c in enumerate(columns):
        for j, d in enumerate(columns):
            Si2s[(c,d)] = Si['S2'][i,j]

    # construct new S2 matrix with columns sorted by ST
    sorted_Si2 = np.zeros((n_columns, n_columns))
    for i, c in enumerate(model.feature_columns[inds]):
        for j, d in enumerate(model.feature_columns[inds]):
            if np.isnan(Si2s[(c,d)]):
                sorted_Si2[i,j] = Si2s[(d,c)]
            else:
                sorted_Si2[i,j] = Si2s[(c,d)]

    # convert back to triangular matrix
    to_nan_matrix = np.tri(len(model.feature_columns), len(model.feature_columns))
    to_nan_matrix[to_nan_matrix==0] = np.nan
    sorted_Si2 = sorted_Si2 * to_nan_matrix

    # make plot
    _, ((dummy_ax, cbar_ax), (bar_ax, heat_ax)) = plt.subplots(
        2, 2, figsize=(7, 4.5), gridspec_kw={'height_ratios': (1, 12)}
    )

    # colors for bar plot
    cmap = plt.get_cmap('tab20')

    # cap S1 values by ST
    Si['S1'] = np.clip(Si['S1'], None, Si['ST'])

    # bar plot of ST and S1
    for col, err_col, index in zip([cmap(1), cmap(3)], [cmap(0), cmap(2)], ['ST', 'S1']):
        barplot = sns.barplot(
            x = 'Channel',
            y = 'Sobol index',
            color = col,
            data = pd.DataFrame({
                'Channel': sorted_columns,
                'Sobol index': Si[index][inds]
            }),
            yerr = Si[f'{index}_conf'][inds]/2 if plot_errors else None,
            ecolor = err_col,
            ax = bar_ax
        )

    # bar plot style
    barplot.set_yscale("log")
    barplot.set_xticklabels(
        barplot.get_xticklabels(),
        rotation=90,
        fontsize=SMALL_FONT_SIZE
    )
    bar_ax.legend(
        handles = [
            mpl.patches.Patch(facecolor=cmap(1), label='$S_T$'),
            mpl.patches.Patch(facecolor=cmap(3), label='$S_1$')
        ],
        loc='best'
    )

    # heatmap of S2
    heatmap = sns.heatmap(
        np.abs(sorted_Si2),
        square=True,
        norm=mpl.colors.LogNorm(),
        xticklabels=model.feature_columns[inds],
        yticklabels=model.feature_columns[inds],
        cmap='flare',
        cbar_kws = {
            'orientation': 'horizontal',
            'label': '$S_2$'
        },
        cbar_ax = cbar_ax,
        ax = heat_ax
    )

    # heatmap style
    heatmap.xaxis.set_tick_params(labelsize=SMALL_FONT_SIZE)
    heatmap.yaxis.set_tick_params(labelsize=SMALL_FONT_SIZE)
    heatmap.set(xlabel='Channel')

    dummy_ax.axis('off')

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

FUNC_MAP = {
    'summary': summary,
    'timeseries': timeseries,
    'genetic': genetic,
    'cluster': cluster,
    'gradient': gradient,
    'sobol': sobol
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    # load config file given in command line
    with open(sys.argv[1]) as config_file:
        config = yaml.load(config_file, yaml.FullLoader)
    
    if len(sys.argv) == 2:
        for func in FUNC_MAP.values():
            func(**config)
    else:
        FUNC_MAP[sys.argv[2]](**config)