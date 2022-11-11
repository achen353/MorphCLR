# Unsupervised Domain Generalization by Learning a Bridge Across Domains
The official code of the CVPR 2022 paper:
**Unsupervised Domain Generalization by Learning a Bridge Across Domains**
*Authors:* Sivan Harary, Eli Schwartz, Assaf Arbelle, Peter Staar, Shady Abu-Hussein, Elad Amrani, Roei Herzig, Amit Alfassy, Raja Giryes, Hilde Kuehne, Dina Katabi, Kate Saenko, Rogerio Feris, Leonid Karlinsky

https://arxiv.org/abs/2112.02300

## Installation

### Create Environment

```
$ conda create --name brad --no-default-packages python=3.7.6
$ conda activate brad
```
### Full list of dependencies:

Install the dependencies via pip:
```
$ pip install -r requirements.txt
```

In case there are any problems installing the conda environment as describes above, the following is a full list of all dependecies need to run the training, testing and demo.
1. pytorch (version ~ 1.8) and the corresponding torchvision
2. scikit-image
3. scikit-learn
4. tqdm
5. requests
6. jupyterlab (for demo)
7. ipywidgets (for demo)

## Data Prep
Please see [data_split/DATA_README.m](./data_splits/DATA_README.md)

## Downloading our pre-trained models
The model can be downloaded from [Google Drive](https://drive.google.com/file/d/1T7v2xwAWQGsAv11-CEwKLmUH-TmCkue9/view?usp=sharing)

## Run Training
To run our model first activate the conda environment:

```
$ conda activate brad
```

Run the main training script using torch.distributed.launch:

```
$ python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> main_brad.py --data <DATA_ROOT>/clipart_train_test.txt,<DATA_ROOT>/painting_train_test.txt,<DATA_ROOT>/real_train_test.txt,<DATA_ROOT>/sketch_train_test.txt
```

Please see the config.py file for all available parameters or run:

```
$ python main_brad.py -h
```

## Run Test
To run our model first activate the conda environment:

```
$ conda activate brad
```

Run the main test script using torch.distributed.launch:

```
$ python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> main_brad_test.py --resume <PATH_TO_TRAINED_MODEL> --src-domain <PATH_TO_SRC_DOMAIN_TXT_FILE> --dst-domain <PATH_TO_DST_DOMAIN_TXT_FILE> 
```
For instance, for 1-shot with source Real and target Painting use: 

```
$ python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> main_brad_test.py --resume <PATH_TO_TRAINED_MODEL> --src-domain <DATA_ROOT>/real_labeled_1.txt --dst-domain <DATA_ROOT>/painting_unlabeled_1.txt 
```

Use the flag `--classifier` to choose classifier type out of [`retrieval`, `sgd`, `logistic`], the default is `retrieval`.  

## Run Demo
1. Make sure that the conda environment is set properly
2. Download the DomainNet Dataset
3. Download the pre-calculated features from [Google Drive](https://drive.google.com/drive/folders/1OvowfDCNCxPCAgaOi0nVDEpiB3AF2Uut?usp=sharing)
4. Run the jupyter notebook and open `demo.ipynb`
1. Modify the paths to the data under `data_root`
6. Run the demo section
