import enum
import os
import sys
import tqdm
import logging
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import scipy.signal

from fetch_aux import (round_gps_time, 
                       times_from_gwpy_timeseries,
                       fetch_timeseries,
                       TIME_COL, SEGMENT_SIZE)

def mag2db(mag):
    '''
    Convert magnitude to decibels.
    '''
    return 20*np.log10(mag)

def fetch_sqz(start_gps, end_gps, no_sqz_start_gps, no_sqz_end_gps,
              ifo, sqz_path, blrms_lims, veto_channels,
              overlap_fraction=0.7, batch_size=1800,
              average='median', **kwargs):
    OMC_CHANNELS = [f'{ifo}:OMC-DCPD_{c}_OUT_DQ' for c in 'AB']

    # channels used to veto stretches without squeezing
    veto_channel_names = [f'{ifo}:{c}' for c in list(veto_channels.keys())]

    # column names for each squeeze level in the dataframe
    sqz_column_names = [f'SQZ_dB {f_low}-{f_high}Hz' 
                        for f_low, f_high in blrms_lims]

    # round start and end times to nearest 60s segment
    start_gps = round_gps_time(start_gps)
    end_gps = round_gps_time(end_gps, floor=True)

    # create output file path if it does not exist
    output_file_path = Path(sqz_path)
    output_file_path.parent.mkdir(exist_ok=True, parents=True)

    # fetch cross-correlation from no-squeeze segment
    logging.info('Fetching OMC PD data from no-squeeze segment...')
    no_sqz_timeseries = fetch_timeseries(OMC_CHANNELS,
                                         no_sqz_start_gps,
                                         no_sqz_end_gps)

    fs = int(1 / no_sqz_timeseries[OMC_CHANNELS[0]].dt.value)
    hann_window = scipy.signal.hann(fs)
    noverlap = np.round(overlap_fraction * fs)
    csd_settings = {
        'window': hann_window,
        'noverlap': noverlap,
        'fs': fs,
        'nfft': fs,
        'average': average
    }
    average_fn = np.median if average == 'median' else np.mean

    # use mean for the cross-correlation
    _, xcorr = scipy.signal.csd(no_sqz_timeseries[OMC_CHANNELS[0]],
                                no_sqz_timeseries[OMC_CHANNELS[1]],
                                **dict(csd_settings, **{'average': 'mean'}))

    logging.info('Computed non-squeeze cross-correlation.')

    # load progress if it exists and contains some data
    current_gps = start_gps
    if (output_file_path.is_file()
        and len(sqz_data := 
                pd.read_csv(output_file_path, index_col=TIME_COL)
            ) > 0):
        current_gps = sqz_data.index[-1]
        percent_progress = 100 * (current_gps-start_gps)/(end_gps-start_gps)

        logging.info(f'Loaded {percent_progress:.0f}% progress.')

        current_gps += SEGMENT_SIZE
    # build empty dataframe if no progress
    else:
        sqz_data = pd.DataFrame(dict(
            {TIME_COL: []},
            **{ c: [] for c in sqz_column_names }))
        sqz_data.set_index(TIME_COL, inplace=True)

        sqz_data.to_csv(output_file_path)

    # grab full squeeze data
    logging.info('Fetching full squeezer data...')
    with tqdm.tqdm(total=end_gps-start_gps,
                   initial=current_gps-start_gps) as pbar:
        while current_gps <= end_gps:
            veto_timeseries = fetch_timeseries(veto_channel_names,
                                               current_gps,
                                               current_gps + batch_size)

            # grab OMC data for current batch
            omc_timeseries = fetch_timeseries(OMC_CHANNELS,
                                              current_gps,
                                              current_gps + batch_size)

            # only analyze if data was fetched from server
            if veto_timeseries is not None and omc_timeseries is not None:
                # get gps times corresponding to each segment
                times = times_from_gwpy_timeseries(
                    veto_timeseries[veto_channel_names[0]]
                )
                # store squeezing levels from this batch in a list to avoid
                # dataframe appending overhead for every loop
                new_sqz_data_list = []
                for i, t in enumerate(times):
                    a, b = i*fs*SEGMENT_SIZE, (i+1)*fs*SEGMENT_SIZE
                    # check that the IFO is in the desired state
                    if (all((int(veto_timeseries[f'{ifo}:{c}'][i].value)
                        == veto_channels[c]) for c in veto_channels.keys())
                        and a < len(omc_timeseries[OMC_CHANNELS[0]])):
                        # compute sum and difference of OMC PDs in time
                        omc_null = (omc_timeseries[OMC_CHANNELS[0]].value[a:b]
                                - omc_timeseries[OMC_CHANNELS[1]].value[a:b])
                        omc_sum = (omc_timeseries[OMC_CHANNELS[0]].value[a:b]
                                + omc_timeseries[OMC_CHANNELS[1]].value[a:b])

                        # compute ASDs of sum and difference
                        f, psd_null = scipy.signal.welch(omc_null,
                                                        **csd_settings)
                        asd_null = np.sqrt(psd_null)
                        _, psd_sum = scipy.signal.welch(omc_sum,
                                                        **csd_settings)

                        # set negative values to nan to avoid numpy warnings
                        psd_quantum = psd_sum - 2*np.abs(xcorr)
                        psd_quantum[np.argwhere(psd_quantum < 0)] = np.nan
                        asd_sum = np.sqrt(psd_quantum)

                        # compute median squeezing level within each frequency
                        # band
                        cur_sqz_levels = [0]*len(blrms_lims)
                        for i, (f_min, f_max) in enumerate(blrms_lims):
                            # get indices corresponding to each BLRMS bin
                            f_inds = np.argwhere(np.logical_and.reduce((
                                f>=f_min, f<=f_max, ~np.isnan(asd_sum)
                            )))
                            # compute sum/null ratio and convert to dBs to get
                            # squeezing level
                            cur_sqz_levels[i] = mag2db(
                                average_fn(asd_sum[f_inds])
                                / average_fn(asd_null[f_inds])
                            )
                        new_sqz_data_list += [[t] + cur_sqz_levels]

                # append new data to the existing dataframe
                if len(new_sqz_data_list) > 0:
                    new_sqz_data = pd.DataFrame.from_dict(new_sqz_data_list)
                    new_sqz_data.columns = [TIME_COL] + sqz_column_names
                    new_sqz_data.set_index(TIME_COL, inplace=True)
                    sqz_data = pd.concat((sqz_data, new_sqz_data))

                # save checkpoint of data
                sqz_data.to_csv(output_file_path)

            current_gps += batch_size
            pbar.update(batch_size)
    
    logging.info('Fetched all data.')

if __name__ == "__main__":
    # show info messages
    logging.basicConfig(level=logging.INFO)

    # load config file given in command line
    with open(sys.argv[1]) as config_file:
        config = yaml.load(config_file, yaml.FullLoader)

    # set keytab path for gwpy data fetching
    os.environ['KRB5_KTNAME'] = config['keytab_path']

    # pull data and estimate squeezing level
    fetch_sqz(**config)