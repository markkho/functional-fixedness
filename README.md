# Rational simplification and rigidity in human planning

Studies of functional fixedness and shifting between different sets of construals during planning.

## Overview

This repo is organized into the following high-level directories:
- `experiments`, which contains the data (zipped), analysis, and modeling files for experiments 1 and 2
- `construal_shifting`, which is a python package that includes code used for modeling
- `psiturkapp`, which is the javascript application for the experiments (requires [psiturk](https://psiturk.org/))

## Set up

It is easiest to run the notebooks in a Python virtual environment. These
steps are one way to do it: 

1. Set up the appropriate virtual environment (from root). For example:
```
$ virtualenv -p python3 env
```

2. Load the virtual environment:
```
$ source env/bin/activate
```

3. Install dependencies:
```
$ pip install -r requirements.txt
```

4. Install project code as a python package (`-e` flag tracks edits):
```
$ pip install -e construal_shifting
```


