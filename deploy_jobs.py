from cmath import e
import sys
import os
from pathlib import Path

import yaml
import logging

import numpy as np

from slurm import submit_jobs
from train_nn import SQZModel

'''
Function for deploying training jobs that span some specified range of
parameter space.

Run by calling deploy function or from commandline with:
`python deploy_jobs.py main config.yaml savepath parameter1 start1 end1 count1 parameter2 start2 end2 count2 ...`
for making numerical ranges for given parameters, or
`python deploy_jobs.py main config.yaml savepath parameter1 value1/value2/...`
for setting a list of parameter values.

Parameters that live in nested dictionaries can be addressed as, e.g. `nested/dict/keys`.

List values can be set as e.g. `parameter_name 1,2/1,2,3,4/5,6,7`.

Run
`python deploy_jobs.py $jobId config.yaml savepath`
for the jobs

If the jobs should be lightweight (i.e. don't save the models), add a --light flag at the end.

If you only want to produce the parameter file (i.e. don't send jobs to slurm), add a --nodeploy flag at the end.
'''

# label for special 'duration' parameter
DURATION_LABEL = 'duration'

# get path to source directory
source_directory = Path(os.path.dirname(os.path.abspath(__file__)))

# path for submit script
SUBMIT_FILENAME = 'batch.sh'
# filename for parameter able
PARAM_FILENAME = 'params.txt'

# commandline flags
LIGHTWEIGHT_FLAG = '--light'
NODEPLOY_FLAG = '--nodeploy'

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

def deploy(save_path, lightweight, nodeploy, config_spans, config):
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
    '''
    
    # ensure save path exists
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    # initialize list of individual job parameters
    init_config_spans = [{}]

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
            # iterate over all fractions of full segment
            for start_f in np.arange(0, 1, f):
                # starting GPS timestamp
                sub_start_gps = int(
                    config['start_gps'] + full_segment_length * start_f
                )

                # save GPS start and end timestamps
                init_config_spans += [{
                    'duration': f,
                    'sub_start_gps': sub_start_gps,
                    'sub_end_gps': sub_start_gps + int(full_segment_length * f)
                }]

        # remove duration config span
        config_spans.pop(duration_ind)
        logging.info(f'Producing jobs with duration fractions = {duration_values}')

    #### build parameter spans
    job_params = deploy_aux(init_config_spans, config_spans)
    num_jobs = len(job_params)

    # fix parameter values to ints according to config values
    for job in job_params:
        for p in job:
            if p in config and isinstance(config[p], int):
                job[p] = int(job[p])

    logging.info(f'Built parameters for {num_jobs} jobs')

    # write jobs and parameters to file
    parameter_table_path = save_path / PARAM_FILENAME
    with open(parameter_table_path, 'w') as parameter_table:
        for i, job in enumerate(job_params):
            parameter_table.write(
                f"{i} {' '.join([f'{k} {v}' for k, v in job.items()])}\n"
            )

    #### submit jobs
    submit_path = Path(save_path) / SUBMIT_FILENAME
    logging.debug(f'Building submit file at {submit_path}...')

    logging.debug('Submitting...')
    submit_jobs(num_jobs, __file__, 'batch', 
                submit_path, str(save_path) + LIGHTWEIGHT_FLAG if lightweight else '',
                config, nodeploy)

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
                if isinstance(parent[k], list) or ',' in job_params[p]:
                    job_params[p] = [int(val) for val in job_params[p].split(',')]
                # if int/float, perform cast
                else:
                    for type in [float, int]:
                        if p in config and isinstance(config[p], type):
                            job_params[p] = type(job_params[p])

                # set value in config dictionary
                parent[k] = job_params[p]
            else:
                if k not in parent:
                    parent[k] = {}
                
                parent = parent[k]
    
    return
    config.update(job_params)

    # if sub GPS start/end is not specified, set to full segment
    for l in ['start_gps', 'end_gps']:
        if f'sub_{l}' not in config:
            logging.warning(f'No sub_{l} given in job specification.')
            config[f'sub_{l}'] = config[l]
        
    # training parameters for batch deployment
    config['save_period'] = 200
    config['batch_size'] = 4096

    # train model
    model = SQZModel(
        None if lightweight else save_path / f'model_{job_num}', **config
    )

    # if not saving model, save loss values
    if lightweight:
        model.save_avg_loss(save_path / f'loss_{job_num}.txt')

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
    else:
        # individual generation member job
        job_num = int(sys.argv[1])
        logging.info(f'Starting job number {job_num}...')

        sub_job(save_path, lightweight, job_num, config)