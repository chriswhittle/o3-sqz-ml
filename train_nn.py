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
                nominal_blrms_lims, neural_network, val_fraction=0.2,
                save_period=10, batch_size=512, **kwargs):
    #### prepare data for training the neural network

    # load in data
    data = pd.read_csv(processed_path, index_col='gps_time')
    
    # create output file path if it does not exist
    output_file_path = Path(save_path)
    output_file_path.mkdir(exist_ok=True, parents=True)

    # trim data to within given GPS times
    data.drop(data[(data.index < sub_start_gps) 
                    | (data.index > sub_end_gps)].index, inplace=True)

    # remove other SQZ columns
    sqz_column = f'SQZ_dB {nominal_blrms_lims[0]}-{nominal_blrms_lims[1]}Hz'
    other_sqz = data.columns[data.columns.str.startswith('SQZ')]
    other_sqz = other_sqz.drop(sqz_column)
    data.drop(other_sqz, axis=1, inplace=True)
    data.rename(columns={sqz_column: 'SQZ'}, inplace=True)

    # divide data into training and validation sets
    split_ind = int((1-val_fraction) * len(data))
    training_features = data.iloc[:split_ind]
    validation_features = data.iloc[split_ind:]

    training_labels = training_features.pop('SQZ')
    validation_labels = validation_features.pop('SQZ')

    #### training the neural network

    # initialize model
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(training_features))

    model = keras.Sequential([normalizer]
        + (
            [] if neural_network['rff_dim'] == 0
            else [layers.experimental.RandomFourierFeatures(
                output_dim=neural_network['rff_dim']
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
        model.fit(
            training_features, training_labels,
            validation_data=(validation_features, validation_labels),
            verbose=is_verbose, epochs=neural_network['epochs'],
            callbacks=callbacks_list, batch_size=batch_size
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