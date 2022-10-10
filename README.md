# Enhancing Shape Bias in Vision Models with Limited Supervision

This is a project for [CS 8803 LS](https://sites.google.com/view/cs8803ls-fa22/home) class at Georgia Tech for the Fall 2022 semester. The project is built upon the PyTorch [implementation](https://github.com/sthalles/SimCLR) of SimCLR by [@sthalles](https://github.com/sthalles).

## Setup

### Option 1: Install with conda

Create a conda environment with `env.yml`:

```
$ conda env create --name morphclr --file env.yml
$ conda activate morphclr
```

If your conda stucks at solving environment, try Option 2.

### Option 2: Install with pip

1. Create a clean conda environment with Python 3.7.6.

```
$ conda create --name morphclr --no-default-packages python=3.7.6
$ conda activate morphclr
```

2. Install [pip-tools](https://pypi.org/project/pip-tools/):

```
$ pip install pip-tools
```

3. In `requirements.in`, update the link following `--find-links` to the correct repo for your CUDA version. By default the CUDA version is 10.2.

4. Resolve dependencies with pip-tools:

```
$ pip-compile requirements.in
```

5. Install dependencies with `requirements.txt`:

```
$ pip install -r requirements.txt
```

6. For distributed training, install Apex according to the [documentation](https://github.com/NVIDIA/apex#installation).

## Our Approach

To be updated.

## Feature Evaluation

See [mini_batch_logistic_regression_evaluator.ipynb](feature_eval/mini_batch_logistic_regression_evaluator.ipynb).