import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import pandas as pd
from tqdm.keras import TqdmCallback

from SALib.sample import saltelli
from SALib.analyze import sobol

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental import RandomFourierFeatures

# suppress tensorflow optimization messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# https://gist.github.com/qin-yu/b3da088669db84f87a2541578cf7fa60
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(16)

from sklearn.cluster import KMeans

RNN_TYPES = {
    'lstm': tf.keras.layers.LSTM
}

def load_data(processed_path, nominal_blrms_lims, channels,
                cut_channels, **kwargs):
    # load in data
    data = pd.read_csv(processed_path, index_col='gps_time')

    # remove other SQZ columns
    sqz_column = f'SQZ_dB {nominal_blrms_lims[0]}-{nominal_blrms_lims[1]}Hz'
    other_sqz = data.columns[data.columns.str.startswith('SQZ')]
    other_sqz = other_sqz.drop(sqz_column)
    data.drop(other_sqz, axis=1, inplace=True)
    data.rename(columns={sqz_column: 'SQZ'}, inplace=True)

    # remove cut channels by first generating list of channels in dataframe
    # to be removed
    readable_cut = []
    for c in cut_channels:
        if channels[c] in data.columns and channels[c] != 'SQZ':
            readable_cut += [channels[c]]
    data.drop(readable_cut, axis=1, inplace=True)

    return data

@dataclass
class SQZData:
    training_features: pd.DataFrame
    training_labels: pd.DataFrame
    validation_features: pd.DataFrame
    validation_labels: pd.DataFrame

class Detrender:
    def __init__(self, data, save_path=None, loading_model=True):
        # try to load detrend values from file
        if (save_path is not None and save_path.is_file() and loading_model):
            detrend_values = np.loadtxt(save_path, ndmin=2)
            self.mean = detrend_values[0,:]
            self.std = detrend_values[1,:]
        # otherwise compute detrend values
        else:
            self.mean = data.mean()
            self.std = data.std()

            # optionally save to file
            if save_path is not None:
                np.savetxt(save_path, np.vstack((self.mean, self.std)))


    def detrend(self, data):
        return (data - self.mean) / self.std
    
    def retrend(self, data):
        return data * self.std + self.mean

class SQZModel:
    # filename constants
    LOSS_HISTORY_FILENAME = 'loss.txt'
    DETREND_FEATURES_FILENAME = 'detrend_features.txt'
    DETREND_LABELS_FILENAME = 'detrend_labels.txt'
    CLUSTERS_FILENAME = 'clusters.csv'
    CLUSTERS_ABS_FILENAME = 'clusters_abs.csv'

    TIME_STEP = 60

    ################################################
    #### helper function for saving training history
    def save_avg_loss(self, save_path):
        np.savetxt(
            save_path,
            self.loss_history_avg,
            header='loss val_loss'
        )

    ################################################
    #### helper function for building datasets for
    #### RNN input
    def build_sequence_dataset(self, features, labels=None, times=False):
        # handle gaps in data to avoid windows spanning more than nominal time
        # step
        if (isinstance(features, pd.DataFrame) and 
            (labels is None or isinstance(labels, pd.DataFrame))):
            # build contiguous time series
            full_times = np.arange(
                features.index[0], features.index[-1], self.TIME_STEP
            )

            # add NaNs where there are gaps in data
            contiguous_features = features.reindex(full_times, fill_value=np.nan)

            # produce sliding window with [window, window_size, features]
            sliding = np.lib.stride_tricks.sliding_window_view(
                contiguous_features, self.lookback, axis=0
            )
            sliding = np.swapaxes(sliding, 1, 2)

            # generate boolean mask for windows that can be kept (i.e. no NaNs)
            nan_mask = ~np.isnan(sliding).any(axis=(1,2))

            # if caller only cares about time series, mask into times and return
            if times:
                return full_times[self.lookback-1:][nan_mask]

            # mask features to be returned
            features_ds = sliding[nan_mask, :, :]
            
            # mask labels dataframe if given
            if labels is not None:
                contiguous_labels = labels.reindex(full_times, fill_value=np.nan)
                sliding_labels = contiguous_labels.iloc[self.lookback-1:]
                labels_ds = sliding_labels[nan_mask]
            
                return (features_ds, labels_ds)
            else:
                return (features_ds,)
        # otherwise just use tensorflow dataset generator
        else:
            # convert to numpy arrays so tf timeseries generation does correct
            # indexing
            features = np.array(features)
            labels = np.array(features)

            return tf.keras.utils.timeseries_dataset_from_array(
                features,
                labels[self.lookback-1:],
                self.lookback,
                sequence_stride=1,
                batch_size=self.batch_size
            )

    ################################################
    #### cluster computation
    def compute_clusters(self, cluster_count, save_path=None,
                        output_file_path=None, force_overwrite=False):
        loading_model = (save_path is not None and not force_overwrite)
        
        # try to load cluster centroids
        if (loading_model and output_file_path is not None
            and (output_file_path / self.CLUSTERS_FILENAME).is_file()
        ):
            # load cluster centers from file (use first column as cluster IDs)
            self.clusters = pd.read_csv(
                output_file_path / self.CLUSTERS_FILENAME,
                index_col=0
            )

            # initialize kmeans object for labeling, but instead manually
            # set cluster centers
            self.kmeans = KMeans(
                n_clusters=cluster_count, random_state=0, max_iter=1
            ).fit(self.clusters)
            self.kmeans.cluster_centers_ = np.ascontiguousarray(
                self.clusters, dtype=float
            )

            # label training data based on clusters
            training_clusters = self.kmeans.predict(
                self.detrended_data.training_features
            )
        else:
            # do k-means clustering
            self.kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(
                self.detrended_data.training_features
            )
            
            self.clusters = pd.DataFrame(
                self.kmeans.cluster_centers_,
                columns=self.feature_columns
            )
            if save_path is not None:
                # save cluster centers
                self.clusters.to_csv(output_file_path / self.CLUSTERS_FILENAME)

                # save de-normalized cluster centers
                retrended_clusters = self.features_detrender.retrend(self.clusters)
                retrended_clusters.to_csv(
                    output_file_path / self.CLUSTERS_ABS_FILENAME
                )

            # use labels allocated during k-means for the training labels
            training_clusters = self.kmeans.labels_

        # compute labels for the validation data
        validation_clusters = self.kmeans.predict(
            self.detrended_data.validation_features
        )

        return self.clusters, training_clusters, validation_clusters

    ################################################
    #### initialization
    def __init__(self, save_path, sub_start_gps, sub_end_gps, processed_path,
                nominal_blrms_lims, neural_network, cut_channels, channels,
                val_fraction=0.2, val_start_gps=None, val_end_gps=None,
                save_period=10, batch_size=512, cluster_count=1,
                interpolate=True, show_progress=False, force_overwrite=False,
                **kwargs):
        # save_path = None => nothing saved


        # throw error if clustered neural network and RNN
        if cluster_count > 1 and neural_network['rnn']['type'] in RNN_TYPES:
            raise RuntimeError(
                'Cannot use clustered neural networks with RNN.'
            )

        ###################################################
        #### prepare data for training the neural network
        data = load_data(processed_path,
                        nominal_blrms_lims,
                        channels,
                        cut_channels)

        ###################################################
        #### prepare data for model training
        
        # create output file path if it does not exist
        if save_path is not None:
            output_file_path = Path(save_path)
            output_file_path.mkdir(exist_ok=True, parents=True)

        # divide data into training and validation sets
        sub_start_gps = int(sub_start_gps)
        sub_end_gps = int(sub_end_gps)
        if val_start_gps is None or val_end_gps is None:
            # trim data to within given GPS times
            data.drop(data[(data.index < sub_start_gps) 
                            | (data.index > sub_end_gps)].index, inplace=True)

            # use given validation fraction to separate training and validation
            # features
            split_ind = int((1-val_fraction) * len(data))
            training_features = data.iloc[:split_ind]
            validation_features = data.iloc[split_ind:]
        else:
            # if validation GPS start and end times are given, training features
            # will use complete specified range and validation features will be
            # the separately specified GPS timestamps
            training_features = data.drop(data[(data.index < sub_start_gps) 
                            |               (data.index > sub_end_gps)].index)
            validation_features = data.drop(data[(data.index < val_start_gps) 
                            |               (data.index > val_end_gps)].index)

        training_labels = training_features.pop('SQZ')
        validation_labels = validation_features.pop('SQZ')
        
        self.feature_columns = training_features.columns

        # save training/validation labels/features in object
        self.raw_data = SQZData(
            training_features,
            training_labels,
            validation_features,
            validation_labels
        )

        ###################################################
        #### compute means and stds for detrending

        # boolean whether to attempt to load existing model
        loading_model = (save_path is not None and not force_overwrite)

        # create detrender objects, each of which will try to load
        # existing detrending values (if they exist), else will compute
        # them and (optionally) save
        self.features_detrender = Detrender(
            self.raw_data.training_features,
            output_file_path / self.DETREND_FEATURES_FILENAME,
            loading_model
        )

        self.labels_detrender = Detrender(
            self.raw_data.training_labels,
            output_file_path / self.DETREND_LABELS_FILENAME,
            loading_model
        )

        # use detrenders to compute detrended data
        self.detrended_data = SQZData(
            self.features_detrender.detrend(self.raw_data.training_features),
            self.labels_detrender.detrend(self.raw_data.training_labels),
            self.features_detrender.detrend(self.raw_data.validation_features),
            self.labels_detrender.detrend(self.raw_data.validation_labels)
        )

        ###################################################
        #### produce shifted datasets for RNNs

        self.batch_size = batch_size

        # shape training data for LSTM
        if neural_network['rnn']['type'] in RNN_TYPES:
            self.lookback = neural_network['rnn']['lookback']
            self.training_rnn_ds = self.build_sequence_dataset(
                self.detrended_data.training_features,
                self.detrended_data.training_labels
            )
            
            self.validation_rnn_ds = self.build_sequence_dataset(
                self.detrended_data.validation_features,
                self.detrended_data.validation_labels
            )
        else:
            self.lookback = 0

        ###################################################
        #### compute clusters

        ( clusters, training_clusters,
            validation_clusters ) = self.compute_clusters(
                cluster_count, save_path, output_file_path
        )
        
        ###################################################
        #### training the neural networks

        # initialize class properties to store models/losses
        self.cluster_count = cluster_count
        self.interpolate = interpolate
        self.models = [None] * cluster_count
        self.loss_histories = [None] * cluster_count
        
        # save cluster data in object
        self.clusters = clusters
        self.training_clusters = training_clusters
        self.validation_clusters = validation_clusters

        for i in range(cluster_count):
            ####################
            # initialize model

            # select training features/labels associated with current cluster
            inds = (training_clusters==i)
            sub_training_features = self.detrended_data.training_features[inds]
            sub_training_labels = self.detrended_data.training_labels[inds]

            # boolean whether there are any validation data points in this
            # cluster
            sub_validation_exists = (validation_clusters==i).sum() > 0

            ####################
            # train/load model

            if save_path is not None:
                # create subfolder for this sub-network if it doesn't exist
                sub_output_path = output_file_path / f'cluster_{i}'
                sub_output_path.mkdir(exist_ok=True, parents=True)

                # check if loss history exists
                loss_path = sub_output_path / self.LOSS_HISTORY_FILENAME

            # if loss file exists (and model was already trained)
            if loading_model and loss_path.is_file():
                # load loss history from file
                loss_history = np.loadtxt(loss_path, skiprows=1)
                
                # load model weights from file
                available_epochs = []
                for checkpoint_file in sub_output_path.glob('*.hdf5'):
                    available_epochs += [
                        int(str(checkpoint_file).split('-')[1].replace('.hdf5',''))
                    ]

                # if no checkpoints saved 
                if len(available_epochs) == 0:
                    model = tf.keras.models.load_model(str(sub_output_path))
                else:
                    # use the lowest loss checkpoint
                    ckpt = available_epochs[np.argmin(
                        loss_history[np.minimum(
                                available_epochs, neural_network['epochs']-1
                            ), 1]
                        )]
                    
                    model = tf.keras.models.load_model(
                        sub_output_path / f'checkpoint-{ckpt:04d}.hdf5'
                    )
            # if loss file doesn't exist and model needs to be trained
            else:
                # define internal layers in neural network
                model = tf.keras.Sequential(
                    # pre-dense layers
                    [layers.Dense(neural_network['dense_dim'],
                                    activation=neural_network['activation'])
                        for _ in range(neural_network['pre_dense_layers'])]
                    # RFF layer
                    # include if non-zero dimension specified or if RNN
                    # (in which case neural network accepts time series)
                    + (
                        [] if (neural_network['rff_dim'] == 0 or
                        neural_network['rnn']['type'] in RNN_TYPES)
                        else [RandomFourierFeatures(
                            output_dim=neural_network['rff_dim']
                        )]
                    )
                    # RNN layers
                    + (
                        [
                            RNN_TYPES[neural_network['rnn']['type']](
                                units=neural_network['rnn']['dim']
                            )
                        ]
                        if neural_network['rnn']['type'] in RNN_TYPES
                        else []
                    )
                    # dense layers
                    + [layers.Dense(neural_network['dense_dim'],
                                    activation=neural_network['activation'])
                        for _ in range(neural_network['dense_layers'])]
                    # final dense layer to single number for squeezing level
                    # estimate
                    + [layers.Dense(1)]
                )

                model.compile(loss='mean_absolute_error',
                            optimizer=tf.keras.optimizers.Adam(0.001))

                # set verbosity level for training
                is_verbose = (logging.root.level <= logging.DEBUG) or show_progress

                # set up callbacks for saving model checkpoints
                if save_path is None:
                    callbacks_list = []
                else:
                    steps_per_epoch = max(
                        (sub_training_labels.size - (self.lookback-1)) 
                        / batch_size, 1
                    )
                    checkpoint_path = sub_output_path / 'checkpoint-{epoch:04d}.ckpt'
                    
                    callbacks_list = ([
                        callbacks.ModelCheckpoint(checkpoint_path,
                                                monitor='val_loss', verbose=is_verbose,
                                                save_freq=int(save_period * steps_per_epoch))
                        ] + ([TqdmCallback(verbose=0)] if is_verbose else []))
                
                # train model
                fit_args = {
                    'verbose': is_verbose,
                    'epochs': neural_network['epochs'],
                    'callbacks':callbacks_list,
                    'batch_size': batch_size
                }

                if neural_network['rnn']['type'] in RNN_TYPES:
                    # if RNN, check if generate a tensorflow dataset:
                    # if so, just use dataset as argument
                    if isinstance(self.training_rnn_ds, tf.data.Dataset):
                        model.fit(
                            self.training_rnn_ds,
                            validation_data=self.validation_rnn_ds,
                            **fit_args
                        )
                    # otherwise grab elements from tuple
                    else:
                        model.fit(
                            self.training_rnn_ds[0],
                            self.training_rnn_ds[1],
                            validation_data=self.validation_rnn_ds,
                            **fit_args
                        )

                else:
                    # just input features and labels if non-RNN
                    if sub_validation_exists:
                        inds = (validation_clusters==i)
                        fit_args['validation_data'] = (
                            self.detrended_data.validation_features[inds],
                            self.detrended_data.validation_labels[inds]
                        )

                    model.fit(
                        sub_training_features,
                        sub_training_labels,
                        **fit_args
                    )

                # save model
                if save_path is not None:
                    model.save(sub_output_path)

                # save loss history
                cur_loss = model.history.history['loss']
                loss_history = np.vstack((
                                    cur_loss,
                                    model.history.history['val_loss']
                                    if sub_validation_exists else
                                    np.zeros( len(cur_loss) )
                                    )).T

                # re-scale loss history by label std
                loss_history *= self.labels_detrender.std

                if save_path is not None:
                    np.savetxt(loss_path, loss_history,
                            header='loss val_loss')

            self.models[i] = model
            self.loss_histories[i] = loss_history
        
        # save averaged losses to file:
        if save_path is not None:
            avg_loss_filepath = output_file_path / self.LOSS_HISTORY_FILENAME
        
        if save_path is None or not avg_loss_filepath.is_file():
            self.loss_history_avg = np.zeros(self.loss_histories[0].shape)
            # 1) training loss is just the average of individual losses weighted by
            # number of data points associated with each cluster
            # 2) can also compute a validation loss using the same as above
            for i, clusters in enumerate([training_clusters, validation_clusters]):
                # count number of data points in each cluster to use as weights
                cluster_weights = [
                    (clusters==i).sum() for i in range(self.cluster_count)
                ]

                # take average of losses weighted by cluster counts
                self.loss_history_avg[:,i] = np.average(
                    np.array(self.loss_histories)[:,:,i],
                    axis=0, weights=cluster_weights
                )
            
            # 3) and can compute a validation loss using interpolation between
            # models (as in the estimate_sqz function), but this would require
            # reloading models from each epoch and performing interpolation.

            if save_path is not None:
                self.save_avg_loss(avg_loss_filepath)

    ################################################
    #### predict squeezing level using trained model
    def estimate_sqz(self, features):
        # transpose if given a single point
        if features.ndim == 1:
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            features = features[np.newaxis, :]
        
        # convert feature values to DataFrame if not already
        features = pd.DataFrame(features, columns=self.feature_columns)

        # detrend provided data
        detrended_features = self.features_detrender.detrend(features)
        
        # handle RNN
        # assumes contiguous (i.e. uniformly spaced data)
        if self.lookback > 0:
            # if RNN and datapoints given is less than lookback, throw error
            if features.shape[0] < self.lookback:
                raise RuntimeError(
                    f'Insufficient datapoints given for RNN lookback length {self.lookback}'
                )
            elif self.cluster_count > 1:
                raise RuntimeError(
                    'Cannot use clustered neural networks with RNN.'
                )
            # build timeseries of input features for RNN and compute sqz
            # estimate
            else:
                rnn_ds = self.build_sequence_dataset(detrended_features)
                sqz_est = self.models[0].predict(rnn_ds).flatten()

        else:
            # for non-RNN networks, interpolate between cluster centers to
            # produce final estimate
            if self.interpolate:
                # compute sqz estimate from each model
                sqz_ests = [
                    model.predict(detrended_features).flatten()
                    for model in self.models
                ]

                # compute distances for each data point from each cluster
                # (number of clusters, number of data points)
                distances = np.zeros( (self.cluster_count, features.shape[0]) )
                for i in range(self.cluster_count):
                    distances[i,:] = np.sqrt(
                        np.square(detrended_features - self.clusters.loc[i]).sum(axis=1)
                    )
                # use inverse distance as weighting for model
                weights = 1 / distances

                sqz_est = (sqz_ests * weights).sum(axis=0) / weights.sum(axis=0)
            # label each data point based on cluster membership and use
            # a single model to predict
            else:
                # calculate clusters for each data point
                labels = self.kmeans.predict(detrended_features)
                
                # initialize array for sqz estimate
                sqz_est = np.zeros(labels.shape)

                for i in range(self.cluster_count):
                    if (labels==i).sum() > 0:
                        sqz_est[labels==i] = (
                            self.models[i].predict(detrended_features[labels==i])
                            .flatten()
                        )

        return self.labels_detrender.retrend(sqz_est)
    
    ################################################
    #### sensitivity analysis functions
    def sobol(self, N=1000):
        '''
        Computes Sobol indices for current model.

        N = number of samples will be computed as (N x number of features)

        Returns dictionary with keys:
        'S1' = list of first-order Sobol indices
        'S2' = matrix of second-order Sobol indices
        'ST' = list of total-order indices (ST~=S1 indices mostly first-order
        interactions, while ST-S1>0 implies higher-order interactions)

        Also '_conf' suffixes to get 95% confidence intervals
        '''
        if self.lookback > 0:
            raise RuntimeError('Cannot compute Sobol indices for RNN.')

        # define number of parameters and bounds
        sobol_problem = {
            'num_vars': self.feature_columns.shape,
            'names': self.feature_columns,
            'bounds': list(zip(
                self.detrended_data.training_features.min().to_list(),
                self.detrended_data.training_features.max().to_list()
            ))
        }

        # generate random samples
        feature_values = saltelli.sample(sobol_problem, N)

        # run samples through model
        est_sqz = self.estimate_sqz(feature_values)

        # compute Sobol indices
        Si = sobol.analyze(sobol_problem, est_sqz)

        return Si
    
    def __gradient_tape(self, model, point, depth=1):
        if depth == 0:
            return model(point)

        with tf.GradientTape() as tape:
            evaluate = self.__gradient_tape(model, point, depth-1)
        return tape.gradient(evaluate, point)

    def gradient(self, normalize=True, sort=True, depth=1, point=None,
                numerical=False):
        if self.lookback > 0:
            raise RuntimeError('Cannot compute gradient for RNN.')

        # compute gradient of model at the specified point or, if undefined,
        # the median of the training data (if a single model) or the cluster
        # centroids (if multiple clusters).
        # can optionally normalize by the standard deviation

        NUMERICAL_STD_FRACTION = 20

        # use median of training features if no point given and no clustering
        if point is None and self.cluster_count == 1:
            point = [self.detrended_data.training_features.median().to_frame().T]
        # use cluster medians if clustering
        elif point is None:
            point = [self.clusters.iloc[i].to_frame().T 
                        for i in range(self.cluster_count)]
        else:
            # transpose if given 1D array
            if point.ndim == 1:
                if not isinstance(point, np.ndarray):
                    point = np.array(point)
                point = point[np.newaxis, :]
            
            point = self.features_detrender.detrend(point)

            # duplicate given point by number of clusters
            point = [point]*self.cluster_count
        
        gradients = [None]*self.cluster_count
        for i in range(self.cluster_count):
            # compute gradient at given point
            if numerical:
                # set up gradient array for results
                gradient = np.zeros(
                    (1, self.feature_columns.size)
                )

                # compute sqz estimate at the point of interest
                est_mid = self.models[i](point[i])

                # iterate over all channels
                for j, c in enumerate(self.feature_columns):
                    # use some fraction of the channel std for the finite
                    # difference computation
                    h = (
                        self.detrended_data.training_features
                        [self.training_clusters==i][c]
                        .std() / NUMERICAL_STD_FRACTION
                    )

                    # compute the sqz estimate at offsets from the point of
                    # interest
                    sign = [-1, 1]
                    ests = [0, 0]
                    for k, s in enumerate(sign):
                        p = point[i].copy()
                        p[c] = p[c] + h * s
                        ests[k] = self.models[i](p)

                    # depending on required differentiation order, use appropriate
                    # finite difference coefficients
                    # https://en.wikipedia.org/wiki/Finite_difference_coefficient
                    if depth == 1:
                        gradient[0,j] = (-ests[0] + ests[1]) / 2 / h
                    elif depth == 2:
                        gradient[0,j] = (ests[0] - 2*est_mid + ests[1]) / h**2
                    else:
                        raise RuntimeError(
                            "Can only numerically compute first and second derivatives"
                        )
            else:
                # convert to tensorflow variable
                tf_point = [tf.Variable(p, dtype=tf.float32) for p in point]

                gradient = self.__gradient_tape(self.models[i], tf_point[i], depth).numpy()

            # retrend to convert back to gradient in units of SQZ dB
            # and original channels
            gradient = gradient * self.labels_detrender.std[0]
            gradient = gradient / (self.features_detrender.std)**depth

            # optionally normalize by standard deviation of channels
            # within this cluster
            if normalize:
                # use standard deviation within the specific cluster
                gradient = gradient * (
                    self.raw_data.training_features[self.training_clusters==i]
                    .std().to_numpy()
                )**depth
        
            # convert to dataframe
            gradient_df = pd.DataFrame(
                gradient, columns=self.feature_columns
            ).T.reset_index()
            gradient_df = gradient_df.rename(columns={'index': 'Channel', 0: f'Gradient'})

            # optionally sort by absolute value of the gradient
            if sort:
                # create temporary third column with the absolute value of the
                # gradient for sorting
                gradient_df['abs_gradient'] = gradient_df['Gradient'].abs()
                gradient_df.sort_values(by='abs_gradient', inplace=True)
                gradient_df.drop('abs_gradient', axis=1, inplace=True)

            gradients[i] = gradient_df
        
        return gradients