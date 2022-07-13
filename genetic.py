import sys
import os
from pathlib import Path

import yaml
import time
import logging

import numpy as np

from train_nn import train_model

'''
Functions for running genetic algorithm to feature select channels.

run with:
`python genetic.py main config.yaml` for the main loop
`python genetic.py $jobId config.yaml` for the jobs
'''

# get path to source directory
source_directory = Path(os.path.dirname(os.path.abspath(__file__)))

# format for trained model paths (subset, gps_start, gps_end)
MODEL_SAVE_PATH_STUB = 'genetic/{}_{}-{}'
DELETE_MODELS = True

# poll job statuses every X seconds
MAIN_POLL_PERIOD = 5

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

def genetic_main(num_features, config):
    '''
    Main loop for handling generations of the genetic algorithm.
    Interacts with SLURM to launch individual jobs in each generation.
    '''
    # genetic algorithm parameters
    G = config['genetic']['pop_size'] # number of members in each generation
    G_SELECT = G//2 # how many members to use for selection
    G_CHILD = 2 * G // G_SELECT # number of children for each pair
    GENERATIONS = config['genetic']['generations'] # number of iterations to run
    M_RATE = 1/num_features # mutation rate

    # paths and template for submit scripts
    submit_stub_path = source_directory / Path('submit_stub.txt')
    submit_path = Path('genetic/genetic.sh')
    with open(submit_stub_path) as file:
        submit_stub = file.read()

    logging.info('Starting main genetic algorithm loop...')

    # load progress if already exists
    if save_path.is_file() and save_path.stat().st_size:
        logging.info(f'Loading progress from {save_path}...')

        with open(save_path) as file:
            for line in file:
                prev_generation = line
            
            prev_generation = prev_generation.split(' ')
            generation = [bitmask_str2np(g, num_features) for g in prev_generation[:G]]
            losses = np.array([float(l) for l in prev_generation[G:]])

        loaded_save_file = True
    # otherwise initialize first generation
    else:
        generation = [np.random.randint(0, 2, num_features) for _ in range(G)]
        losses = -np.ones(config['genetic']['pop_size'])
        save_path.touch()
        loaded_save_file = False
    
    # run through generations of genetic algorithm
    for e in range(GENERATIONS):
            if not loaded_save_file or e > 0:
                with open(save_path, 'a') as file:

                    if loaded_save_file or e > 0:
                        file.write('\n')

                    for g in generation:
                        file.write(f'{bitmask_np2str(g)} ')
                    
                    file.write(' '.join(['-1' for _ in generation]))

            # compute fitnesses
            jobs_submitted = False

            while len((incomplete_jobs := list(
                            np.argwhere(losses == -1).flatten())
                    )) > 0:
                if (not jobs_submitted or 
                    os.popen(
                    f'sacct -j {job_id} | awk \'$5=="RUNNING" {{ print $0 }}\''
                    ).read() == ''):
                    # new submit file containing incomplete job numbers
                    with open(submit_path, 'w') as file:
                        file.write(submit_stub.replace(
                            '[JOB_IDS]',
                            ','.join(map(str, list(incomplete_jobs)))
                        ))
                    # submit new jobs
                    job_id = (os.popen(f'LLsub {submit_path}')
                              .read().split(' ')[-1]).replace('\n', '')
                    jobs_submitted = True

                # try to grab losses from finished jobs
                for j in incomplete_jobs:
                    job_losses = np.zeros(len(gps_ranges))
                    for i, (start, end) in enumerate(gps_ranges):
                        job_file = Path(MODEL_SAVE_PATH_STUB.format(
                            bitmask_np2str(generation[j]),
                            start,
                            end
                        )) / 'loss.txt'
                        if job_file.is_file():
                            model_losses_loaded = False
                            while not model_losses_loaded:
                                try:
                                    model_losses = np.loadtxt(job_file,
                                                              skiprows=1)
                                    model_losses_loaded = True
                                except StopIteration:
                                    logging.info('Retrying loss load...')
                            job_losses[i] = np.min(model_losses[:,1])
                        else:
                            job_losses[i] = np.nan
                    
                    job_loss = np.mean(job_losses)
                    if not np.isnan(job_loss):
                        losses[j] = job_loss
                time.sleep(MAIN_POLL_PERIOD)
            fitness = (-losses).argsort().argsort()

            # write losses to file
            with open(save_path) as file:
                lines = file.readlines()
            with open(save_path, 'w') as file:
                for l in lines[:-1]:
                    file.write(l)

                for g in generation:
                    file.write(f'{bitmask_np2str(g)} ')
                
                file.write(' '.join([str(l) for l in losses]))
            
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

            losses = -np.ones(G)



def genetic_sub(job_num, gps_ranges, num_features, config):
    '''
    Function for launching training of an individual member of a generation.

    job_num = ID of job, indexes to bitmask and loss in genetic algorithm
    history file
    gps_ranges = list of tuples corresponding to starts and ends of GPS
    segments to use for calculating loss
    '''
    # load latest row in genetic algorithm history file
    with open(save_path) as file:
        for line in file:
            prev_generation = line
            
        prev_generation = prev_generation.split(' ')

        # use job number to get the allocated bitmask (channel subset)
        # for this job (as a numpy array)
        current_bitmask_str = prev_generation[job_num]
        current_bitmask = bitmask_str2np(current_bitmask_str, num_features)
        
    # modify neural network config based on genetic-specific settings
    for key, value in config['genetic_network'].items():
        logging.debug(f'Setting {key} = {value}')
        config['neural_network'][key] = value

    # specify channels to be cut based on the bitmask read
    config['cut_channels'] = [
        c for i, c in enumerate(channels) if current_bitmask[i]==0
    ]

    logging.debug(f"{len(config['cut_channels'])} channels cut")
    assert(len(config['cut_channels']) == num_features - current_bitmask.sum())

    # iterate over each GPS segment to test
    for start, end in gps_ranges:
        # create path to model based on bitmask and GPS segment boundaries
        job_file = Path(MODEL_SAVE_PATH_STUB.format(
                        current_bitmask_str,
                        start,
                        end
                    ))

        # run training
        train_model(job_file, start, end, save_period=200,
                    batch_size=4096, **config)

if __name__ == "__main__":

    #######################################
    ### set up variables for genetic run

    # show info messages
    logging.basicConfig(level=logging.DEBUG) ### TODO SET TO INFO

    # load config file given in command line
    with open(sys.argv[2]) as config_file:
        config = yaml.load(config_file, yaml.FullLoader)
    
    # set save path and make if doesn't already exist
    save_path = config['genetic_path']

    # count number of feature channels (subtract channels to be cut)
    num_features = len(config['channels']) - len(config['cut_channels'])
    logging.debug(f'{num_features} features')
    channels = []
    for c in config['channels'].keys():
        if c not in config['cut_channels']:
            channels += [c]

    # define GPS ranges to test
    ########### TODO UPDATE THIS LOGIC TO SOMETHING MORE SENSIBLE ?? ##########
    MONTH = 3600*24*30
    RANGE_COUNT = 3
    start = config['start_gps']
    gps_ranges = [(start + i*MONTH, start + (i+1)*MONTH)
                    for i in range(RANGE_COUNT)]

    ############################################################

    ###########################################
    ### main job logic

    if sys.argv[1] == 'main':
        # main job for submitting member jobs and analyzing results
        logging.info(f'Starting job manager...')
        genetic_main(num_features, config)
    else:
        # individual generation member job

        job_num = int(sys.argv[1])
        logging.info(f'Starting job number {job_num}...')

        genetic_sub(job_num, gps_ranges, num_features, config)