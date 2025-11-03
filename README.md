# A R2-based MOEA for Robust Multi-Objective Neural Architecture Optimization under Adversarial Attacks

### Setup 

Create virtual environment and install dependencies.

```bash

python3 -m venv r2-venv

source r2-venv/bin/activate

pip3 install -r requirements.txt

```

### Run

Execute the R2-based EMOA for RNAS.

```bash

python3 rnas.py --seed <seed> --algorithm r2-emoa --dataset <dataset> --params_dir <params_dir>

```
where

* `<seed>`: random seed for reproducibility
* `<dataset>`: the dataset to test ['cifar10']
* `<params_dir>`: the dir of parameters json file



