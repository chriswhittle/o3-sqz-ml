import os
from pathlib import Path

# name of stub submit file for job submission
SUBMIT_STUB_NAME = 'submit_stub.txt'

# get path to source directory
source_directory = Path(os.path.dirname(os.path.abspath(__file__)))

def submit_jobs(num_jobs, script_path, log_tag,
                submit_path, script_args = '', config = {}):
    '''
    Generate new slurm submit script based on given parameters and then use it
    to submit jobs.

    num_jobs = number of jobs to submit
    script_path = path to script each job should run
    log_tag = label to use for log file naming
    submit_path = path of new submit file
    script_args = additional commandline arguments to add after config file
    config = standard config dictionary, including nodes to be excluded
    '''
    # copy content of submit stub
    submit_stub_path = source_directory / Path(SUBMIT_STUB_NAME)
    with open(submit_stub_path) as file:
        submit_stub = file.read()

    # build batch options string
    # tuple becomes: #SBATCH -e[0] e[1]
    batch_options_list = [
        ('o', f'logs/{log_tag}.log-%A-%a'),
        ('a', f'0-{num_jobs}')
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
                '[JOB_IDS]', f'0-{num_jobs}'
            ).replace(
                '[SCRIPT_PATH]', script_path
            ).replace(
                '[BATCH_OPTIONS]', batch_options
            ).replace(
                '[SCRIPT_ARGS]', script_args
            )
        )
    
    # submit jobs and extract job ID
    submit_string = os.popen(f'LLsub {submit_path}').read()
    job_id = int(submit_string.split(' ')[-1].replace('\n', ''))

    return job_id