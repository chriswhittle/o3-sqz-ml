import os
import sys
import tqdm
import logging
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from gwpy.timeseries import TimeSeriesDict
from subprocess import CalledProcessError

TIME_COL = 'gps_time'
SEGMENT_SIZE = 60

def fetch_timeseries(channels, gps_start, gps_end, max_attempts=6):
    '''
    Use gwpy to fetch channel data, allowing for a number of re-attempts up to
    some maximum.
    '''
    for _ in range(max_attempts):
        try:
            data = TimeSeriesDict.fetch(channels,
                                        gps_start, gps_end,
                                        allow_tape=True)
            return data
        except (KeyError, RuntimeError, CalledProcessError):
            pass
    logging.debug(f'Failed to fetch {channels} in {gps_start}-{gps_start} '
                    + f'after {max_attempts} attempts.')
    return None


def times_from_gwpy_timeseries(tdict):
    '''
    Returns an array of GPS times given a timeseries
    element of a dictionary returned by gwpy.
    '''
    return np.arange(tdict.t0.value,
                     tdict.t0.value + len(tdict.value)*tdict.dt.value,
                     tdict.dt.value)

def round_gps_time(n, floor=False, segment_length=60):
    '''
    Round to the nearest segment chunk (default 60).
    '''
    return int((np.floor if floor else np.ceil)(n / segment_length) * segment_length)

def fetch_aux(start_gps, end_gps, channels, ifo, aux_path,
              batch_size=1800, **kwargs):
    channel_names = [f'{ifo}:{c}' for c in list(channels.keys())]

    # round start and end times to nearest 60s segment
    start_gps = round_gps_time(start_gps)
    end_gps = round_gps_time(end_gps, floor=True)

    # create output file path if it does not exist
    output_file_path = Path(aux_path)
    output_file_path.parent.mkdir(exist_ok=True, parents=True)

    # load progress if it exists and contains some data
    current_gps = start_gps
    if (output_file_path.is_file()
        and len(aux_data := 
                pd.read_csv(output_file_path, index_col=TIME_COL)
            ) > 0):
        current_gps = aux_data.index[-1]
        percent_progress = 100 * (current_gps-start_gps)/(end_gps-start_gps)

        logging.info(f'Loaded {percent_progress:.0f}% progress.')

        current_gps += SEGMENT_SIZE
    # build empty dataframe if no progress
    else:
        aux_data = pd.DataFrame(dict(
            {TIME_COL: []},
            **{ c: [] for c in channel_names }))
        aux_data.set_index(TIME_COL, inplace=True)

        aux_data.to_csv(output_file_path)
    
    logging.info('Fetching auxiliary channel data...')
    with tqdm.tqdm(total=end_gps-start_gps,
                   initial=current_gps-start_gps) as pbar:
        while current_gps <= end_gps:

            batch_data = fetch_timeseries(channel_names,
                                          current_gps,
                                          current_gps + batch_size)

            if batch_data is not None:
                new_aux_data = pd.DataFrame(batch_data)
                new_aux_data[TIME_COL] = times_from_gwpy_timeseries(
                    batch_data[channel_names[0]]
                )
                new_aux_data.set_index(TIME_COL, inplace=True)
                aux_data = pd.concat((aux_data, new_aux_data))

                # save checkpoint of data
                aux_data.to_csv(output_file_path)

            current_gps += batch_size
            pbar.update(batch_size)

    logging.info('Fetched all auxiliary data.')

if __name__ == "__main__":
    # show info messages
    logging.basicConfig(level=logging.INFO)

    # load config file given in command line
    with open(sys.argv[1]) as config_file:
        config = yaml.load(config_file, yaml.FullLoader)

    # set keytab path for gwpy data fetching
    os.environ['KRB5_KTNAME'] = config['keytab_path']

    # pull data
    fetch_aux(**config)
