import sys
import os
from pathlib import Path

import yaml
import time
import logging

import numpy as np

from slurm import submit_jobs
from train_nn import SQZModel

'''
Functions for running genetic algorithm to feature select channels.

run with:
`python genetic.py main config.yaml 100` for the main loop (and 100 iterations)
`python genetic.py $jobId config.yaml` for the jobs
'''

# path for genetic bash submit file
submit_path = Path('genetic/genetic.sh')

# poll job statuses every X seconds
MAIN_POLL_PERIOD = 5
SUB_POLL_PERIOD = 10

# path for job loss results
JOB_LOSS_PATH = 'genetic/job_{}.txt'

# yet-to-be-computed loss value
NULL_LOSS = -1

def bitmask_str2np(s, fill=None):
    '''
    Convert a string bitmask to a numpy array of integers.
    '''
    binary = '{:b}'.format(int(s))
    if fill is not None:
        binary = binary.zfill(fill)
    return np.array([int(c) for c in binary])

def bitmask_np2str(a):
    '''
    Convert a numpy array of integers to a string bitmask.
    '''
    return str(int(''.join([str(c) for c in list(a)]), 2))

def update_latest_losses(save_path, generation, losses):
    with open(save_path) as file:
        lines = file.readlines()
    with open(save_path, 'w') as file:
        for l in lines[:-1]:
            file.write(l)

        # write generation bitmasks
        for g in generation:
            file.write(f'{bitmask_np2str(g)} ')
        
        # write newly-computed losses
        file.write(' '.join([str(l) for l in losses]))

def genetic_main(num_features, num_iter, config):
    '''
    Main loop for handling generations of the genetic algorithm.
    Interacts with SLURM to launch individual jobs in each generation.
    '''
    # genetic algorithm parameters
    G = config['genetic']['pop_size'] # number of members in each generation
    G_SELECT = G//2 # how many members to use for selection
    G_CHILD = 2 * G // G_SELECT # number of children for each pair
    GENERATIONS = num_iter # number of iterations to run
    M_RATE = 1/num_features # mutation rate
    
    # set save path and make if doesn't already exist
    save_path = Path(config['genetic_path'])
    save_path.parent.mkdir(exist_ok=True, parents=True)

    logging.info('Starting main genetic algorithm loop...')

    # initialize dictionary of already-calculated losses
    past_losses = {}

    # load progress if already exists
    if save_path.is_file() and save_path.stat().st_size:
        logging.info(f'Loading progress from {save_path}...')

        with open(save_path) as file:
            for line in file:
                # get subset bitmasks (first G elements) and
                # losses (last G element)
                generation = line.split(' ')
                generation_members = generation[:G]
                losses = np.array([float(l) for l in generation[G:]])
                
                # update dictionary of past losses with current row values
                # (assuming the recorded loss != uncomputed)
                past_losses.update(
                    [el for el in zip(generation_members, losses) if el[1] != NULL_LOSS]
                )
            
            # get numpy bitmasks for current generation
            generation = [bitmask_str2np(g, num_features) for g in generation_members]
        
        loaded_save_file = True
    # otherwise initialize first generation
    else:
        # initialize first generation by generating G random bitmasks
        generation = [np.random.randint(0, 2, num_features) for _ in range(G)]
        # initialize losses to uncomputed values
        losses = np.ones(G) * NULL_LOSS
        save_path.touch()
        loaded_save_file = False

    # helper function for calculating incomplete jobs    
    calc_incomplete = lambda l: list(np.argwhere(l == NULL_LOSS).flatten())
    
    # run through generations of genetic algorithm
    for e in range(GENERATIONS):
        logging.info(f'Main thread starting generation {e}...')
        # write the current row of members to the genetic data file
        if not loaded_save_file or e > 0:
            with open(save_path, 'a') as file:

                if loaded_save_file or e > 0:
                    file.write('\n')

                # write bitmasks for each member
                initial_losses = [NULL_LOSS for _ in generation]
                for i, g in enumerate(generation): # bitmask_string in past_losses
                    bitmask_string = bitmask_np2str(g)
                    file.write(f'{bitmask_string} ')

                    if bitmask_string in past_losses:
                        initial_losses[i] = past_losses[bitmask_string]
                
                # write losses for each member (some marked as uncomputed)
                file.write(' '.join(map(str, initial_losses)))

        # list of jobs that need to be run
        incomplete_jobs = calc_incomplete(losses)
        # count of jobs currently running
        running_count = 0
        # periodically poll jobs to check loss values
        while len(incomplete_jobs) > 0:

            # launch jobs if no jobs being run currently
            if running_count == 0:
                # start jobs and save job ID
                logging.info(f'Submitting {len(incomplete_jobs)} jobs')
                job_id = submit_jobs(
                    incomplete_jobs,
                    __file__,
                    'genetic',
                    submit_path,
                    '',
                    config,
                    id_logs=False
                )

            # check over each job that hasn't finished computing
            new_losses = 0
            for j in incomplete_jobs:
                # check job file that will save the loss
                job_file = Path(JOB_LOSS_PATH.format(j))
                if job_file.is_file() and job_file.stat().st_size > 0:
                    # load in loss (float representing the averaged loss over
                    # all GPS segments)
                    losses[j] = float(np.loadtxt(job_file))
                    new_losses += 1

                    # delete job loss file
                    os.remove(job_file)
                    
                    # save this loss in the dictionary of past losses computed
                    past_losses[bitmask_np2str(generation[j])] = losses[j]
            
            # if recorded new loss values, update file
            update_latest_losses(save_path, generation, losses)
            
            incomplete_jobs = calc_incomplete(losses)
            logging.debug('Current incomplete jobs: ' + 
                            ','.join(map(str, incomplete_jobs)))

            # check status of jobs with the latest job ID
            running_count = int(os.popen(
                f'''squeue -r -j {job_id} | awk '
                    BEGIN {{
                        abbrev["R"]="(Running)"
                        abbrev["PD"]="(Pending)"
                    }}
                    NR>1 {{a[$5]++}}
                    END {{
                        print a["R"]+a["PD"]
                    }}'
                '''
            ).read().strip())

            # wait to check job files and job statuses again
            time.sleep(MAIN_POLL_PERIOD)
        
        # convert losses to fitnesses by sorting then using index as fitness
        fitness = (-losses).argsort().argsort()

        # rewrite file with new losses in last line
        update_latest_losses(save_path, generation, losses)

        # clean up remaining job files
        for p in Path('.').glob('genetic/job_{}.txt'.format('*')):
            p.unlink()

        # selection
        selection = np.random.choice(np.arange(G), G_SELECT, replace=False,
                                        p=fitness/np.sum(fitness))

        # crossover
        new_generation = []
        for i in range(0, len(selection), 2):
            for j in range(G_CHILD):
                new_subset = [0]*num_features
                for k in range(num_features):
                    # mutate
                    if np.random.random() < M_RATE:
                        new_subset[k] = np.random.randint(0, 2)
                    # combine
                    else:
                        choice = np.random.random() < 0.5
                        new_subset[k] = generation[selection[i+choice]][k]
                new_generation += [new_subset]
        generation = new_generation

        losses = np.ones(G) * NULL_LOSS

    # kill jobs and remove submit file
    logging.info(f'Killing job {job_id}...')
    os.popen(f'LLkill {job_id}')
    os.remove(submit_path)

def genetic_sub(job_num, gps_ranges, num_features, config):
    '''
    Function for launching new training of individual generation members.
    '''
    logging.info(f'Job {job_num} started')

    G = config['genetic']['pop_size']
    job_file = Path(JOB_LOSS_PATH.format(job_num))

    # load latest row in genetic algorithm history file
    with open(config['genetic_path']) as file:
        for line in file:
            prev_generation = line
            
        prev_generation = prev_generation.split(' ')

    # if the loss recorded for this job's current row is uncomputed,
    # start training on this subset (assuming the loss hasn't already been
    # recorded)
    if (float(prev_generation[job_num + G]) < 0 and not job_file.is_file()):
        # use job number to get the allocated bitmask (channel subset)
        # for this job (as a numpy array)
        current_bitmask_str = prev_generation[job_num]
        current_bitmask = bitmask_str2np(current_bitmask_str, num_features)

        # execute training
        logging.info(f'Running training for {current_bitmask_str}')
        avg_loss = genetic_job(current_bitmask, gps_ranges,
                                num_features, config)

        # save average minimum loss to the output file for this job
        with open(job_file, 'w') as file:
            file.write(f'{avg_loss}')
        
        logging.info('Training written to file.')

def genetic_job(bitmask, gps_ranges, num_features, config):
    '''
    Function for training of an individual member of a generation.

    job_num = ID of job, indexes to bitmask and loss in genetic algorithm
    history file
    gps_ranges = list of tuples corresponding to starts and ends of GPS
    segments to use for calculating loss

    Returns average loss of model over GPS segments.
    '''
        
    # modify neural network config based on genetic-specific settings
    for key, value in config['genetic_network'].items():
        logging.debug(f'Setting {key} = {value}')
        config['neural_network'][key] = value

    # specify channels to be cut based on the bitmask read
    config['cut_channels'] = [
        c for i, c in enumerate(channels) if bitmask[i]==0
    ]

    logging.debug(f"{len(config['cut_channels'])} channels cut")
    assert(len(config['cut_channels']) == num_features - bitmask.sum())

    # iterate over each GPS segment to test
    min_losses = np.zeros(len(gps_ranges))
    for i, (start, end) in enumerate(gps_ranges):
        # run training (without saving models)
        model = SQZModel(None, start, end, save_period=200,
                    batch_size=4096, **config)

        # grab minimum validation loss from this GPS segment
        min_losses[i] = model.loss_history_avg[:,1].min()
    
    # calculate average minimum loss of the GPS segments
    avg_loss = np.mean(min_losses)

    return avg_loss

if __name__ == "__main__":

    #######################################
    ### set up variables for genetic run

    # show info messages
    logging.basicConfig(level=logging.INFO)

    # load config file given in command line
    with open(sys.argv[2]) as config_file:
        config = yaml.load(config_file, yaml.FullLoader)

    # count number of feature channels (subtract channels to be cut)
    num_features = len(config['channels']) - len(config['cut_channels'])
    logging.debug(f'{num_features} features')
    channels = []
    for c in config['channels'].keys():
        if c not in config['cut_channels']:
            channels += [c]

    # retrieve GPS ranges to be used for tests
    gps_ranges = config['genetic']['gps_ranges']

    ############################################################

    ###########################################
    ### main job logic

    if sys.argv[1] == 'main':
        # main job for submitting member jobs and analyzing results
        logging.info(f'Starting job manager...')
        genetic_main(num_features, int(sys.argv[3]), config)
    else:
        # individual generation member job

        job_num = int(sys.argv[1])
        logging.info(f'Starting job number {job_num}...')

        genetic_sub(job_num, gps_ranges, num_features, config)