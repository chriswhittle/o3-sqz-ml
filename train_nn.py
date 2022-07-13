import sys
import logging
from pathlib import Path
import numpy as np

import yaml
import pandas as pd
from tqdm.keras import TqdmCallback

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.layers.experimental import preprocessing

LOSS_HISTORY_FILENAME = 'loss.txt'

def train_model(save_path, sub_start_gps, sub_end_gps, processed_path,
                nominal_blrms_lims, neural_network, cut_channels, channels,
                val_fraction=0.2, val_start_gps=None, val_end_gps=None,
                save_period=10, batch_size=512, **kwargs):
    #### prepare data for training the neural network

    # load in data
    data = pd.read_csv(processed_path, index_col='gps_time')
    
    # create output file path if it does not exist
    output_file_path = Path(save_path)
    output_file_path.mkdir(exist_ok=True, parents=True)

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

    # divide data into training and validation sets
    if val_start_gps is None or val_end_gps is None:
        # trim data to within given GPS times
        data.drop(data[(data.index < sub_start_gps) 
                        | (data.index > sub_end_gps)].index, inplace=True)

        split_ind = int((1-val_fraction) * len(data))
        training_features = data.iloc[:split_ind]
        validation_features = data.iloc[split_ind:]
    else:
        training_features = data.drop(data[(data.index < sub_start_gps) 
                        |               (data.index > sub_end_gps)].index)
        validation_features = data.drop(data[(data.index < val_start_gps) 
                        |               (data.index > val_end_gps)].index)

    training_labels = training_features.pop('SQZ')
    validation_labels = validation_features.pop('SQZ')

    # shape training data for LSTM
    if neural_network['lstm_dim'] > 0:
        training_features = training_features.reset_index().drop(
            columns='gps_times')
        training_labels = training_labels.reset_index().drop(
            columns='gps_times')

        training_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=training_features, targets=training_labels,
            sequence_length=neural_network['lstm_lookback']
        )

        validation_features = validation_features.reset_index().drop(
            columns='gps_times')
        validation_labels = validation_labels.reset_index().drop(
            columns='gps_times')

        val_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=validation_features, targets=validation_labels,
            sequence_length=neural_network['lstm_lookback']
        )

    #### training the neural network

    # initialize model
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(training_features))

    model = keras.Sequential(
        (
            [] if neural_network['lstm_dim'] == 0
            # else [layers.Input(
            #     shape=()
            # )]
            else [] #########
        )
        + [normalizer]
        + (
            [] if neural_network['rff_dim'] == 0
            else [layers.experimental.RandomFourierFeatures(
                output_dim=neural_network['rff_dim']
            )]
        )
        + (
            [] if neural_network['lstm_dim'] == 0
            else [layers.LSTM(
                units=neural_network['lstm_dim'],
                stateful=True
            )]
        )
        + [layers.Dense(neural_network['dense_dim'],
                        activation=neural_network['activation'])
            for _ in range(neural_network['dense_layers'])]
        + [layers.Dense(1)]
    )

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))

    loss_path = output_file_path / LOSS_HISTORY_FILENAME
    if loss_path.is_file():
        # load loss history from file
        loss_history = np.loadtxt(loss_path, skiprows=1)
        
        # load model weights from file
        available_epochs = []
        for checkpoint_file in output_file_path.glob('*.hdf5'):
            available_epochs += [
                int(str(checkpoint_file).split('-')[1].replace('.hdf5',''))
            ]

        # if no checkpoints saved 
        if len(available_epochs) == 0:
            model = tf.saved_model.load(str(output_file_path))
        else:
            # use the lowest loss checkpoint
            ckpt = available_epochs[np.argmin(
                loss_history[np.minimum(
                        available_epochs, neural_network['epochs']-1
                    ), 1]
                )]
            
            model.load_weights(output_file_path / f'checkpoint-{ckpt:04d}.hdf5')
    else:
        # set up callbacks for saving model checkpoints
        steps_per_epoch = training_labels.size / batch_size
        checkpoint_path = output_file_path / 'checkpoint-{epoch:04d}.hdf5'
        is_verbose = logging.root.level <= logging.DEBUG
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
            model.fit(
                training_features, training_labels,
                validation_data=(validation_features, validation_labels),
                **fit_args
            )

        # save model
        model.save(output_file_path)

        # save loss history
        loss_history = np.vstack((
                            model.history.history['loss'],
                            model.history.history['val_loss']
                            )).T
        np.savetxt(loss_path, loss_history,
                   header='loss val_loss')

    return model, loss_history