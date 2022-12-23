import logging
from pathlib import Path
import numpy as np

import pandas as pd
from tqdm.keras import TqdmCallback

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import callbacks
from keras import preprocessing

# https://gist.github.com/qin-yu/b3da088669db84f87a2541578cf7fa60
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(16)

from sklearn.cluster import KMeans

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

class SQZModel:
    # filename constants
    LOSS_HISTORY_FILENAME = 'loss.txt'
    DETREND_VALUES_FILENAME = 'detrend.txt'
    CLUSTERS_FILENAME = 'clusters.csv'
    CLUSTERS_ABS_FILENAME = 'clusters_abs.csv'

    def save_avg_loss(self, save_path):
        np.savetxt(
            save_path,
            self.loss_history_avg,
            header='loss val_loss'
        )

    def compute_clusters(self, cluster_count, save_path=None,
                        output_file_path=None):
        # try to load detrending properties
        if (save_path is not None and output_file_path is not None and
            (output_file_path / self.DETREND_VALUES_FILENAME).is_file()
        ):
            aux_values = np.loadtxt(
                output_file_path / self.DETREND_VALUES_FILENAME
            )

            aux_mean = aux_values[0,:]
            aux_std = aux_values[1,:]
        # compute detrending properties
        else:
            aux_mean = self.training_features.mean()
            aux_std = self.training_features.std()

            # save detrending values for loading later on
            if save_path is not None:
                np.savetxt(
                    output_file_path / self.DETREND_VALUES_FILENAME,
                    np.vstack( (aux_mean, aux_std) )
                )

        # define detrending functions
        self.detrend = lambda d: (d - aux_mean) / aux_std
        self.retrend = lambda r: (r * aux_std) + aux_mean
        
        # detrend data to get normed data for labeling by cluster
        norm_data = self.detrend(self.training_features)

        # try to load cluster centroids
        if (save_path is not None and output_file_path is not None
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
                self.clusters, dtype=np.float
            )

            # label training data based on clusters
            training_clusters = self.kmeans.predict(self.detrend(self.training_features))
        else:
            # do k-means clustering
            self.kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(
                norm_data
            )
            
            self.clusters = pd.DataFrame(self.kmeans.cluster_centers_, columns=norm_data.columns)
            if save_path is not None:
                # save cluster centers
                self.clusters.to_csv(output_file_path / self.CLUSTERS_FILENAME)

                # save de-normalized cluster centers
                self.retrend(self.clusters).to_csv(output_file_path / self.CLUSTERS_ABS_FILENAME)

            # use labels allocated during k-means for the training labels
            training_clusters = self.kmeans.labels_

        # compute labels for the validation data
        validation_clusters = self.kmeans.predict(self.detrend(self.validation_features))

        return self.clusters, training_clusters, validation_clusters

    def __init__(self, save_path, sub_start_gps, sub_end_gps, processed_path,
                nominal_blrms_lims, neural_network, cut_channels, channels,
                val_fraction=0.2, val_start_gps=None, val_end_gps=None,
                save_period=10, batch_size=512, cluster_count=1,
                interpolate=True, show_progress=False, **kwargs):
        # save_path = None => nothing saved
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

        # shape training data for LSTM
        if neural_network['lstm_dim'] > 0:
            training_features = training_features.reset_index().drop(
                columns='gps_time')
            training_labels = training_labels.reset_index().drop(
                columns='gps_time')

            training_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=training_features, targets=training_labels,
                sequence_length=neural_network['lstm_lookback']
            )

            validation_features = validation_features.reset_index().drop(
                columns='gps_time')
            validation_labels = validation_labels.reset_index().drop(
                columns='gps_time')

            val_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=validation_features, targets=validation_labels,
                sequence_length=neural_network['lstm_lookback']
            )

        # save training/validation labels/features in object
        self.training_labels = training_labels
        self.training_features = training_features
        self.validation_labels = validation_labels
        self.validation_features = validation_features

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
            sub_training_features = training_features[training_clusters==i]
            sub_training_labels = training_labels[training_clusters==i]

            # boolean whether there are any validation data points in this
            # cluster
            sub_validation_exists = (validation_clusters==i).sum() > 0

            # set up normalizer layer
            normalizer = preprocessing.Normalization()
            normalizer.adapt(np.array(sub_training_features))

            # define internal layers in neural network
            model = keras.Sequential(
                # input layer for LSTM
                (
                    [] if neural_network['lstm_dim'] == 0
                    # else [layers.Input(
                    #     shape=()
                    # )]
                    else [] #########
                )
                # normalizer
                + [normalizer]
                # RFF layer
                + (
                    [] if neural_network['rff_dim'] == 0
                    else [layers.experimental.RandomFourierFeatures(
                        output_dim=neural_network['rff_dim']
                    )]
                )
                # LSTM layers
                + (
                    [] if neural_network['lstm_dim'] == 0
                    else [layers.LSTM(
                        units=neural_network['lstm_dim'],
                        stateful=True
                    )]
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

            ####################
            # train/load model

            if save_path is not None:
                # create subfolder for this sub-network if it doesn't exist
                sub_output_path = output_file_path / f'cluster_{i}'
                sub_output_path.mkdir(exist_ok=True, parents=True)

                # check if loss history exists
                loss_path = sub_output_path / self.LOSS_HISTORY_FILENAME

            # if loss file exists (and model was already trained)
            if save_path is not None and loss_path.is_file():
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
                    model = tf.saved_model.load(str(sub_output_path))
                else:
                    # use the lowest loss checkpoint
                    ckpt = available_epochs[np.argmin(
                        loss_history[np.minimum(
                                available_epochs, neural_network['epochs']-1
                            ), 1]
                        )]
                    
                    model.load_weights(sub_output_path / f'checkpoint-{ckpt:04d}.hdf5')
            # if loss file doesn't exist and model needs to be trained
            else:
                # set verbosity level for training
                is_verbose = (logging.root.level <= logging.DEBUG) or show_progress

                # set up callbacks for saving model checkpoints
                if save_path is None:
                    callbacks_list = []
                else:
                    steps_per_epoch = max(
                        sub_training_labels.size / batch_size, 1
                    )
                    checkpoint_path = sub_output_path / 'checkpoint-{epoch:04d}.hdf5'
                    
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
                if neural_network['lstm_dim'] > 0:
                    model.fit(training_dataset, validation_data=val_dataset, **fit_args)
                else:
                    if sub_validation_exists:
                        fit_args['validation_data'] = (
                            validation_features[validation_clusters==i],
                            validation_labels[validation_clusters==i]
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

    def estimate_sqz(self, features):
        # detrend data
        norm_features = self.detrend(features)

        # interpolate between cluster centers to produce final estimate
        if self.interpolate:
            # compute sqz estimate from each model
            sqz_ests = [
                -model.predict(features).flatten() for model in self.models
            ]

            # compute distances for each data point from each cluster
            # (number of clusters, number of data points)
            distances = np.zeros( (self.cluster_count, features.shape[0]) )
            for i in range(self.cluster_count):
                distances[i,:] = np.sqrt(
                    np.square(norm_features - self.clusters.loc[i]).sum(axis=1)
                )
            # use inverse distance as weighting for model
            weights = 1 / distances

            sqz_est = (sqz_ests * weights).sum(axis=0) / weights.sum(axis=0)
        # label each data point based on cluster membership and use
        # a single model to predict
        else:
            # calculate clusters for each data point
            labels = self.kmeans.predict(norm_features)
            
            # initialize array for sqz estimate
            sqz_est = np.zeros(labels.shape)

            for i in range(self.cluster_count):
                if (labels==i).sum() > 0:
                    sqz_est[labels==i] = -(
                        self.models[i].predict(features[labels==i]).flatten()
                    )

        return sqz_est
    
    def __gradient_tape(self, model, point, depth=1):
        if depth == 0:
            return model(point)

        with tf.GradientTape() as tape:
            evaluate = self.__gradient_tape(model, point, depth-1)
        return tape.gradient(evaluate, point)

    def gradient(self, normalize=True, sort=True, depth=1, point=None,
                numerical=False):
        # compute gradient of model at the specified point or, if undefined,
        # the median of the training data (if a single model) or the cluster
        # centroids (if multiple clusters).
        # can optionally normalize by the standard deviation

        NUMERICAL_STD_FRACTION = 20

        # use median of training features if no point given and no clustering
        if point is None and self.cluster_count == 1:
            point = [self.training_features.median().to_frame().T]
        # use cluster medians if clustering
        elif point is None:
            point = [self.clusters.iloc[i].to_frame().T 
                        for i in range(self.cluster_count)]
        # duplicate given point by number of clusters
        else:
            point = [point]*self.cluster_count
        
        gradients = [None]*self.cluster_count
        for i in range(self.cluster_count):
            # compute gradient at given point
            if numerical:
                # set up gradient array for results
                gradient = np.zeros((1, self.training_features.columns.size))

                # compute sqz estimate at the point of interest
                est_mid = self.models[i](point[i])

                # iterate over all channels
                for j, c in enumerate(self.training_features.columns):
                    # use some fraction of the channel std for the finite
                    # difference computation
                    h = (self.training_features[self.training_clusters==i][c]
                            .std() / NUMERICAL_STD_FRACTION)

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

            # optionally normalize by standard deviation of channels
            if normalize:
                # use standard deviation within the specific cluster
                gradient = gradient * (
                    self.training_features[self.training_clusters==i]
                    .std().to_numpy()
                )**depth
        
            # convert to dataframe
            # negative gradient since internal model maps to negative sqz
            gradient_df = pd.DataFrame(
                - gradient, columns=self.training_features.columns
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