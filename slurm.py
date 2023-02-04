import os
from pathlib import Path
import numpy as np

# name of stub submit file for job submission
SUBMIT_STUB_NAME = 'submit_stub.txt'

# job args
NUM_CORES = 12

# get path to source directory
source_directory = Path(os.path.dirname(os.path.abspath(__file__)))

def submit_jobs(num_jobs, script_path, log_tag, submit_path, script_args = '',
                config = {}, serial_runs=1, nodeploy=False, id_logs=True):
    '''
    Generate new slurm submit script based on given parameters and then use it
    to submit jobs.

    num_jobs = number of jobs to submit (or list if submitting specified jobs)
    script_path = path to script each job should run
    log_tag = label to use for log file naming
    submit_path = path of new submit file
    script_args = additional commandline arguments to add after config file
    config = standard config dictionary, including nodes to be excluded
    serial_jobs = number of runs each job should do serially
    nodeploy = make submit file but don't submit
    '''
    # copy content of submit stub
    submit_stub_path = source_directory / Path(SUBMIT_STUB_NAME)
    with open(submit_stub_path) as file:
        submit_stub = file.read()
    
    # check if list of jobs being submitted or a job count
    if isinstance(num_jobs, list):
        array_value = ','.join(map(str, num_jobs))
    else:
        # number of jobs submitted will be total number of runs
        array_max = int(np.ceil(num_jobs/serial_runs)) - 1
        array_value = f'0-{array_max}'
    
    # uniquely identify logs by job ID if specified
    if id_logs:
        log_id_suffix = '-%A-%a'
    else:
        log_id_suffix = '-%a'

    # build batch options string
    # tuple becomes: #SBATCH -e[0] e[1]
    batch_options_list = [
        ('o', f'logs/{log_tag}.log{log_id_suffix}'),
        ('e', f'logs/{log_tag}.err{log_id_suffix}'),
        ('a', array_value),
        ('c', NUM_CORES)
    ]

    # if hosts to be excluded is included in the config dictionary,
    # add batch option to remove
    if ('computation' in config and 
        'excluded_hosts' in config['computation'] and 
        len(config['computation']['excluded_hosts']) > 0):
        batch_options_list += [
            ('x', ','.join(config['computation']['excluded_hosts']))
        ]

    batch_options = '\n'.join(
        [f'#SBATCH -{b[0]} {b[1]}' for b in batch_options_list]
    )

    # write modified submit script to new file
    with open(submit_path, 'w') as file:
        file.write(submit_stub.replace(
                '[BATCH_OPTIONS]', batch_options
            ).replace(
                '[SERIAL_RUNS]', str(serial_runs)
            ).replace(
                '[SCRIPT_PATH]', script_path
            ).replace(
                '[SCRIPT_ARGS]', script_args
            )
        )
    
    # submit jobs and extract job ID
    if nodeploy:
        return None
    else:
        submit_string = os.popen(f'LLsub {submit_path}').read()
        job_id = int(submit_string.split(' ')[-1].replace('\n', ''))

        return job_id