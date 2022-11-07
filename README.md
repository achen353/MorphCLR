# Enhancing Shape Bias in Vision Models with Limited Supervision

This is a project for [CS 8803 LS](https://sites.google.com/view/cs8803ls-fa22/home) class at Georgia Tech for the Fall 2022 semester. The project is built upon the PyTorch [implementation](https://github.com/sthalles/SimCLR) of SimCLR by [@sthalles](https://github.com/sthalles).

## Setup

### Option 1: Install with conda

Create a conda environment with `env.yml`:

```
$ conda env create --name morphclr --file fixed_env.yml
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

## Development

### Full Evaluation

To run full evaluation for the test accuracies on:

- Original STL10
- Stylized STL10
- Advesarial STL10

Run the following command:

```
$ python eval.py
```

The stylized STL10 should be automatically downloaded and placed in `./stylization/inter_class_stylized_dataset/` directory.

### Linear Evaluation (Fine-Tuning)

To fine-tune the pretrained model with linear evaluation, use [linear_evaluator.ipynb](feature_eval/linear_evaluator.ipynb).

### Adversarial Evaluation Example

The [eval.py](eval.py) script includes evaluation on adversarial examples. to see an example of plotting the results, see [adversarial_evaluator_example.ipynb](feature_eval/adversarial_evaluator_example.ipynb)

### Data Directory

By default, the STL10 cached by torchvision should be stored in the `./datasets/` directory. This includes running the following files:

- `run.py`
- `run_morphclr.py`
- `eval.py`
- `feature_eval/linear_evaluator.ipynb`
- `feature_eval/advesarial_evaluator_example.ipynb`

If you find any of the existing/new scripts caching the STL10 dataset elsewhere, please modify the file path to `./datasets/`

### Checkpoint Directory

Please save all the checkpoints to `./checkpoints/` depending on whether it is from pretraining or finetuning.