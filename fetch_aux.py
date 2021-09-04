#!/cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/envs/igwn-py37/bin/python
import os
import sys
import yaml
import numpy as np
import pandas as pd
import tqdm

from gwpy.timeseries import TimeSeriesDict

def fetch_timeseries(channels, gps_start, gps_end, max_attempts=3):
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
        except (KeyError, RuntimeError):
            pass
    return None


def times_from_gwpy_timeseries(tdict):
    '''
    Returns an array of GPS times given a timeseries
    element of a dictionary returned by gwpy.
    '''
    return np.arange(tdict.t0.value,
                     tdict.t0.value + len(tdict.value)*tdict.dt.value,
                     tdict.dt.value, dtype=np.int32)

def round_gps_time(n, floor=False, segment_length=60):
    '''
    Round to the nearest segment chunk (default 60).
    '''
    return int((np.floor if floor else np.ceil)(n / segment_length) * segment_length)

def fetch(channels, gps_start, gps_end, path, scale='m',
              chunk_size=5*60, channel_names=None, verbose=False):
    pass

if __name__ == "__main__":
    # load config file given in command line
    with open(sys.argv[1]) as config_file:
        config = yaml.load(config_file, yaml.FullLoader)

    # set keytab path for gwpy data fetching
    os.environ['KRB5_KTNAME'] = config['keytab_path']

