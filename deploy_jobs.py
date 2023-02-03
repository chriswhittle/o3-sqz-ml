from cmath import e
import sys
import os
from pathlib import Path

import yaml
import logging

import numpy as np
import pandas as pd

from slurm import submit_jobs
from train_nn import SQZModel

'''
Function for deploying training jobs that span some specified range of
parameter space.

Run by calling deploy function or from commandline with:
`python deploy_jobs.py main config.yaml savepath parameter1 start1 end1 count1 parameter2 start2 end2 count2 ...`
for making numerical ranges for given parameters, or
`python deploy_jobs.py main config.yaml savepath parameter1 value1/value2/...`
for setting a list of parameter values. This can be used to set a parameter to a single value with a trailing slash: `parameter_name 1/`.

Parameters that live in nested dictionaries can be addressed as, e.g. `nested/dict/keys`.

List values can be set as e.g. `parameter_name 1,2/1,2,3,4/5,6,7`.

Run
`python deploy_jobs.py $jobId config.yaml savepath`
for the jobs

If the jobs should be lightweight (i.e. don't save the models), add a --light flag at the end.

If you only want to produce the parameter file (i.e. don't send jobs to slurm), add a --nodeploy flag at the end.

Can also collate data files with
`python deploy_jobs.py collate config.yaml savepath`
'''

# label for special 'duration' parameter
DURATION_LABEL = 'duration'

# threshold for which a segment is considered valid
DATA_THRESHOLD = 0.5
# time step for data
TIME_STEP = 60

# get path to source directory
source_directory = Path(os.path.dirname(os.path.abspath(__file__)))

# name for job results subdirectory
LOSS_FOLDER = 'losses'
MODEL_FOLDER = 'models'

# path for submit script
SUBMIT_FILENAME = 'batch.sh'
# filename for parameter able
PARAM_FILENAME = 'params.txt'

# commandline flags
LIGHTWEIGHT_FLAG = '--light'
NODEPLOY_FLAG = '--nodeploy'
NODELETE_FLAG = '--nodelete'

def build_span(span):
    '''
    Takes dictionary with values or start/end/count keys and returns list of
    parameter values.
    '''
    if 'values' in span:
        return span['values']
    else:
        return np.linspace(float(span['start']),
                            float(span['end']),
                            int(span['count']))

def deploy_aux(job_params, config_spans):
    '''
    Helper function to recursively iterate each span of parameter values and
    build up list of jobs to be run.

    job_params = list of jobs with which to perform outer product
    '''

    # if nothing in config_spans, just return given list of job_params
    if len(config_spans) == 0:
        return job_params

    # build current span
    cur_span = config_spans[0]
    span = build_span(cur_span)

    # outer product between passed list of job params and new span
    new_job_params = []
    for job in job_params:
        for value in span:
            new_job_params += [
                {cur_span['param']: value, **job}
            ]
    
    # do next layer of iteration
    return deploy_aux(new_job_params, config_spans[1:])

def deploy(save_path, lightweight, nodeploy, config_spans, config, check_data=True):
    '''
    Function to create points in specified parameter range and submit jobs
    with SLURM.

    save_path = name of directory for saving models or loss values
    config_limits = list of config parameters to be spanned. Each element
    should be a dictionary with the following values:
     - param = name of parameter
     - values = list of parameter values to span
     OR
     - start = initial value for parameter span
     - end = final value for parameter span
     - count = number of points with which to span this range
    config = standard dictionary of model parameters
    check_data = whether to load in data and check that a time segment
    generated from a duration fraction actually has sufficient data
    '''
    
    # ensure save path exists
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    # make subdirectory for results (either for models or losses)
    if lightweight:
        (save_path / LOSS_FOLDER).mkdir(exist_ok=True, parents=True)
    else:
        (save_path / MODEL_FOLDER).mkdir(exist_ok=True, parents=True)

    # initialize list of individual job parameters
    init_config_spans = [{}]

    # if doing data check, load in dataframe
    if check_data:
        logging.info('Loading data for interval checking...')
        data = pd.read_csv(config['processed_path'], index_col='gps_time')
        data_times = data.index

    #### handle custom span for `duration`
    # duration = fraction of full time segment to use for training
    duration_check = [c['param'] == DURATION_LABEL for c in config_spans]
    if any(duration_check):
        # build span of duration values
        duration_ind = np.argmax(duration_check)
        duration_values = [
            float(f) for f in build_span(config_spans[duration_ind])
        ]

        # full segment length from which we take sub segments
        full_segment_length = config['end_gps'] - config['start_gps']

        # iterate over duration fractions and build segment start/end GPS times
        init_config_spans = []
        for f in duration_values:
            # for data checking: maximum possible number of samples in 
            # this interval length
            samples_threshold = int(
                DATA_THRESHOLD * full_segment_length * f / TIME_STEP
            )

            # iterate over all fractions of full segment
            for start_f in np.arange(0, 1, f):
                # starting GPS timestamp
                sub_start_gps = int(
                    config['start_gps'] + full_segment_length * start_f
                )
                sub_end_gps = sub_start_gps + int(full_segment_length * f)

                # if doing data check, check how many samples are actually in
                # interval and compare to maximum possible
                run_interval = True
                if check_data:
                    samples_count = np.logical_and(
                        data_times >= sub_start_gps,
                        data_times <= sub_end_gps
                    ).sum()
                    if samples_count < samples_threshold:
                        run_interval = False
                        logging.info(f'Interval ({sub_start_gps}, {sub_end_gps}) rejected ({samples_count} < {samples_threshold} samples)')

                # save GPS start and end timestamps
                if run_interval:
                    init_config_spans += [{
                        DURATION_LABEL: f,
                        'sub_start_gps': sub_start_gps,
                        'sub_end_gps': sub_end_gps
                    }]

        # remove duration config span
        config_spans.pop(duration_ind)
        logging.info(f'Producing jobs with duration fractions = {duration_values}')

    #### build parameter spans
    job_params = deploy_aux(init_config_spans, config_spans)
    final_job_params = []

    # fix parameter values to ints according to config values
    for job in job_params:
        for p in job:
            if p in config and isinstance(config[p], int):
                job[p] = int(job[p])
        

        # post-processing checks on job parameters:
        job_ok = True

        # ensure sequential 1D convolution layers reduce to output size 1
        # before dense layers
        if ('neural_network/lookback' in job and
            'neural_network/cnn/kernel_sizes' in job):
            kernel_sizes = job['neural_network/cnn/kernel_sizes']
            kernel_sizes = np.array([int(s) for s in kernel_sizes.split(',')])

            if int(job['neural_network/lookback'])-1 != (kernel_sizes-1).sum():
                job_ok = False
        
        # if passed all checks, add to final list of jobs
        if job_ok:
            final_job_params += [job]

    # count final number of jobs
    num_jobs = len(final_job_params)

    logging.info(f'Built parameters for {num_jobs} jobs')

    # write jobs and parameters to file
    parameter_table_path = save_path / PARAM_FILENAME
    with open(parameter_table_path, 'w') as parameter_table:
        for i, job in enumerate(final_job_params):
            parameter_table.write(
                f"{i} {' '.join([f'{k} {v}' for k, v in job.items()])}\n"
            )

    #### get number of runs per job from config file
    if 'batch_serial_runs' in config['computation']:
        serial_runs = config['computation']['batch_serial_runs']
    else:
        serial_runs = 1

    #### submit jobs
    submit_path = Path(save_path) / SUBMIT_FILENAME
    logging.debug(f'Building submit file at {submit_path}...')

    logging.debug('Submitting...')
    submit_jobs(num_jobs, __file__, 'batch', 
                submit_path,
                str(save_path) + (' ' + LIGHTWEIGHT_FLAG) if lightweight else '',
                config, serial_runs, nodeploy)

def sub_job(save_path, lightweight, job_num, config):
    '''
    Individual job to train model for specific parameter values.
    '''
    
    # load parameters for this job from the parameter file
    save_path = Path(save_path)
    parameter_table_path = save_path / PARAM_FILENAME
    with open(parameter_table_path) as parameter_table:
        for line in parameter_table:
            line_params = line.strip().split(' ')
            # if current line specifies parameter for this job number
            if int(line_params[0]) == job_num:
                # build dictionary of parameter values
                job_params = dict(zip(
                    line_params[1::2], line_params[2::2]
                ))
    

    # update config file with new values
    logging.info(f'Updating config with: {job_params}')

    # iterate and set each job parameter
    for p in job_params:
        # get nested keys
        keys = p.split('/')
        parent = config

        # recurse into dictionaries to set value and also check type
        for i, k in enumerate(keys):
            if i == len(keys)-1:
                # fix type based on type of value written in config file
                # if list, split into values (assume int elements)
                if (k in parent and isinstance(parent[k], list)) or ',' in job_params[p]:
                    job_params[p] = [int(val) for val in job_params[p].split(',')]
                # if int/float, perform cast
                else:
                    for type in [float, int]:
                        if k in parent and isinstance(parent[k], type):
                            job_params[p] = type(job_params[p])

                # set value in config dictionary
                parent[k] = job_params[p]
            else:
                if k not in parent:
                    parent[k] = {}
                
                parent = parent[k]
    
    # if sub GPS start/end is not specified, set to full segment
    for l in ['start_gps', 'end_gps']:
        if f'sub_{l}' not in config:
            logging.warning(f'No sub_{l} given in job specification.')
            config[f'sub_{l}'] = config[l]
        
    # training parameters for batch deployment
    config['save_period'] = 200
    config['batch_size'] = 4096

    # train model
    logging.info('Training model...')
    model = SQZModel(
        None if lightweight else 
        save_path / MODEL_FOLDER / f'model_{job_num}', **config
    )

    # if not saving model, save loss values
    if lightweight:
        loss_path = save_path / LOSS_FOLDER / f'loss_{job_num}.txt'
        logging.info(f'Loss history saved to {loss_path}.')
        model.save_avg_loss(loss_path)

def collate_results(save_path, filename='results.txt', delete=True, min_count=8):
    save_path = Path(save_path)
    fetch_path = save_path / LOSS_FOLDER

    # throw error if folder with losses doesn't exist
    if not fetch_path.exists():
        raise FileNotFoundError('{fetch_path} not found.')
    
    # fetch results saved in files and put in dictionary
    job_results = {}
    min_epochs = {}
    job_files = []
    for file in fetch_path.iterdir():
        if file.stem.startswith('loss_'):
            job_number = int(file.stem.replace('loss_', ''))
            loss_history = np.loadtxt(file)

            # take median of lowest min_count losses
            job_results[job_number] = np.median(
                sorted(loss_history[:,1])[
                    :min(min_count, loss_history.shape[0])
                ]
            )

            min_epochs[job_number] = np.argmin(loss_history[:,1])
        
        job_files += [file]
    
    # write results to single file
    output_path = save_path / filename
    with open(output_path, 'w') as stream:
        stream.write('\n'.join(
            [f'{k} {job_results[k]} {min_epochs[k]}' for k in sorted(job_results.keys())]
        ))
    
    # after successfully fetching results, delete individual result files
    if delete:
        for file in job_files:
            file.unlink()

    logging.info(f'Collated {len(job_files)} files into {output_path}.')

if __name__ == "__main__":
    # show info messages
    logging.basicConfig(level=logging.INFO)

    # define save path for parameter table and models/loss data
    args = sys.argv
    save_path = args[3]

    # check for lightweight flag
    lightweight = LIGHTWEIGHT_FLAG in args
    if lightweight:
        args.remove(LIGHTWEIGHT_FLAG)

    # check for nodeploy flag
    nodeploy = NODEPLOY_FLAG in args
    if nodeploy:
        args.remove(NODEPLOY_FLAG)

    # load config file given in command line
    with open(args[2]) as config_file:
        config = yaml.load(config_file, yaml.FullLoader)

    if args[1] == 'main':
        # main job for submitting member jobs and analyzing results
        logging.info(f'Starting deployment...')

        # read commandline arguments after 'main' and config file name in
        # format param_name1 start1 end1 count1
        arg_ind = 4
        config_spans = []
        span_keys = ['param', 'start', 'end', 'count']
        while arg_ind < len(args):
            # handle case with list of values
            if '/' in args[arg_ind+1]:
                values = args[arg_ind+1].split('/')
                # remove blank options to allow for setting config parameter
                # to a single value
                if '' in values:
                    values.remove('')

                config_spans += [
                    {'param': args[arg_ind], 'values': values}
                ]
                arg_ind += 2
            # handle case with range specified
            else:
                config_spans += [
                    dict(zip(span_keys, args[arg_ind:arg_ind+len(span_keys)]))
                ]
                arg_ind += len(span_keys)

        deploy(save_path, lightweight, nodeploy, config_spans, config)
    elif args[1] == 'collate':
        # check for nodelete flag
        nodelete = NODELETE_FLAG in args

        collate_results(save_path, delete=not nodelete)
    else:
        # individual generation member job
        job_num = int(sys.argv[1])
        logging.info(f'Starting job number {job_num}...')

        sub_job(save_path, lightweight, job_num, config)