import sys
from pathlib import Path
import logging

import yaml
import numpy as np
import pandas as pd

from fetch_aux import TIME_COL

def hampel(data, window_size=50, threshold=6, fast=True):
    '''
    Hampel filter to remove glitches by comparing deviation from the median
    against the median deviation. Computed quickly using a typical stretch of
    data for the baseline deviation.
    '''
    data = data.copy()
    
    L = 1.4826
    rolling_median = data.rolling(window=window_size, center=True).median()
    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    if fast:
        MAD = (np.median(np.abs(data.iloc[:window_size*2] 
        - np.median(data.iloc[:window_size*2]))))
    else:
        MAD = data.rolling(window=window_size, center=True).apply(MAD)
    difference = np.abs(data - rolling_median)
    
    outlier_idx = data.index[difference > (threshold * L * MAD)]
    return outlier_idx

def process_data(sqz_path, aux_path, processed_path, ifo_lock_channels,
                 ifo, channels, cut_channels, min_lock=600, **kwargs):
    # load data from files
    aux_data = pd.read_csv(aux_path, index_col=TIME_COL)
    sqz_data = pd.read_csv(sqz_path, index_col=TIME_COL)
    
    # get sqz column names
    sqz_columns = list(sqz_data.columns)
    ifo_lock_channels = dict((f'{ifo}:{c}', value) for 
                        (c, value) in ifo_lock_channels.items())
    ifo_lock_channel_names = list(ifo_lock_channels.keys())

    # Hampel filter to remove glitches
    for c in sqz_columns:
        glitch_inds = hampel(sqz_data[c])
        sqz_data.drop(glitch_inds, inplace=True)

    # remove any remaining rows where any squeeze BLRMS exceeds 0 dB
    non_sqz_indices = np.where(np.logical_and.reduce(
        [sqz_data[c] > 0 for c in sqz_columns]
    ))
    non_sqz_times = sqz_data.index[non_sqz_indices]
    sqz_data.drop(non_sqz_times, inplace=True)


    # compute locked state according to config specifications
    aux_data['locked'] = np.logical_and.reduce(
        [aux_data[c]==ifo_lock_channels[c] for c in ifo_lock_channel_names]
    )
    # compute changes between lock state
    blocks = (aux_data['locked'] != aux_data['locked'].shift()).cumsum()
    blocks = blocks[aux_data['locked']]
    # compute lock stretch starts and ends
    locks = blocks.groupby(blocks).apply(lambda x: (x.index[0], x.index[-1]))
    # remove lockstretches shorter than minimum allowed
    for lock_num, (start, end) in locks.items():
        if end-start < min_lock:
            aux_data.drop(blocks.index[blocks==lock_num], inplace=True)

    # merge squeeze level and auxiliary data
    data = pd.concat([sqz_data, aux_data], axis=1, join='inner')

    # remove unneeded columns
    for c in cut_channels:
        data.pop(f'{ifo}:{c}')
    data.pop('locked')

    # rename columns to friendly names
    data.rename(columns=dict(
        (f'{ifo}:{c}', cf) for (c, cf) in channels.items()
    ), inplace=True)

    # create output file path if it does not exist
    output_file_path = Path(processed_path)
    output_file_path.parent.mkdir(exist_ok=True, parents=True)
    data.to_csv(processed_path)
    logging.info(f'Wrote combined data to {processed_path}.')

if __name__ == "__main__":
    # show info messages
    logging.basicConfig(level=logging.INFO)

    # load config file given in command line
    with open(sys.argv[1]) as config_file:
        config = yaml.load(config_file, yaml.FullLoader)

    # process data
    process_data(**config)