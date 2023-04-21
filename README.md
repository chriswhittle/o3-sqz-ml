# o3-sql-ml
Characterization of LIGO O3 squeezer performance using machine learning.

## Fetching squeezing level

Run as:

```
python fetch_sqz.py path/to/config/file.yaml
```

Should be run using an environment with appropriate packages (most notably `gwpy`).
For example, on the LDG you could use igwn-py38 (`/cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/envs/igwn-py38/bin/python`).
Also requires a keytab file for ligo.org credentials (see [here](https://gwpy.github.io/docs/v0.1/timeseries/nds#kerberos-keytabs)),
the path to which should be specified in the configuration file under `keytab_path`.

## Fetching auxiliary data

Run as:

```
python fetch_aux.py path/to/config/file.yaml
```

Same environment requirements as above.

## Processing data

Run as:

```
python process_data.py path/to/config/file.yaml
```

## Training and evaluating model

Use in Python as:

```
from train_nn import SQZModel
with open(config_path) as config_file:
    config = yaml.load(config_file, yaml.FullLoader)
model = SQZModel('model/save/path', start_gps, end_gps, **config)

# compute gradients at mean feature values
gradients = model.gradient()

# compute Sobol indices
sobol_indices = model.sobol(N=1048576)
```

## Running genetic algorithm

Run as:

```
python genetic.py main config.yaml $NUMBER_OF_GENERATIONS
```

Written to run on the [MIT Supercloud](https://supercloud.mit.edu/) with Slurm.

## Batch running over hyperparameter space

Run as e.g.:

```
python deploy_jobs.py main config.yaml save/path/for/models duration 0.05/0.5/1 neural_network/dense_layers 0/2/4/6 neural_network/activation tanh/sigmoid/relu/softplus [...]
```

Again, written to run on the [MIT Supercloud](https://supercloud.mit.edu/) with Slurm.
Can iterate over any number of hyperparameters.

Parameters that lived in nested dictionaries in the configuration file can be addressed as, e.g. `nested/dict/keys`.

Values between specified starts and ends can be iterated over as, e.g.:

```
python deploy_joys.py main config.yaml save/path/for/models parameter1 start1 end1 count1 parameter2 start2 end2 count2 ...
```

Values can be iterated from a slash-separated list as, e.g.:

```
python deploy_jobs.py main config.yaml save/path/for/models parameter1 value1/value2/...
```

Parameter values that require a list can be comma-separated, e.g.

```
python deploy_jobs.py main config.yaml save/path/for/models parameter1 1,2/1,2,3,4/5,6,7
```