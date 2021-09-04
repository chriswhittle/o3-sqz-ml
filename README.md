# o3-sql-ml
Characterization of LIGO O3 squeezer performance using machine learning.

## Fetching squeezing level

Run as:

```
python fetch_sqz.py path/to/config/file.yaml
```

Should be run using an environment with appropriate packages (most notably `gwpy`). For example, on the LDG you could use igwn-py37 (`/cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/envs/igwn-py37/bin/python`). Also requires a keytab file for ligo.org credentials (see [here](https://gwpy.github.io/docs/v0.1/timeseries/nds#kerberos-keytabs)), the path to which should be specified in the configuration file under `keytab_path`.

## Fetching auxiliary data

## Processing data

## Training model

## Running genetic algorithm
