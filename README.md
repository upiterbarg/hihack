# `hihack`

This repo contains the official implementation of all methods and models from the NeurIPS 2023 paper ["NetHack is Hard to Hack"](https://arxiv.org/abs/2305.19240) (Piterbarg, Pinto, Fergus).


## Installation

Please make sure to clone this repository recursively before attempting installation.
```
git clone --recursive git@github.com:upiterbarg/hihack.git
```


To install core dependencies with conda, run,
```conda env create -f conda_config.yaml```


Next, to finish installation of the remaining dependencies on `linux`, run:

```
cd nle
python setup.py install
cd sys/unix && ./setup.sh && cd ../../..
conda install cmake
pip install git+ssh://git@github.com/facebookresearch/moolib
cd dungeonsdata-neurips2022/experiment_code
cd render_utils && pip install -e . && cd ..
pip install -e . && cd ../..
```

---

## Pre-trained Models

Pretrained checkpoints reflecting each of the model architectures (+ training paradigms) explored in our paper are available for download on the web via a single zip [**1.24GB before inflating, 1.3GB after inflating**].

```
wget horatio.cs.nyu.edu/mit/ulyana/hihack/pt_model_ckpts.zip && unzip pt_model_ckpts
```

---

## Evaluation

Code for evaluation with `moolib` (based on [Hambro et al 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/9d9258fd703057246cb341e615426e2d-Paper-Datasets_and_Benchmarks.pdf)) is provided in `eval.py`.

To launch a `$NUM_ROLLOUTS`-size evaluation of **all pretrained model checkpoints** (assuming these have been downloaded via the step above), run

```
python eval.py --model_name_or_path all -n $NUM_ROLLOUTS
```

By default, all final NLE scores/rewards from games played by each model will be saved to a text file in `eval_results` with corresponding name (e.g. final scores from games played by `flat_transformer_bc.tar` will be saved to `eval_results/flat_transformer_bc.txt`).


To launch a `$NUM_ROLLOUTS`-size evaluation of **a single pretrained model checkpoint** pass its alias to `--model_name_or_path`, e.g.,

```
python eval.py --model_name_or_path hier_trnsfrmr_bc -n $NUM_ROLLOUTS
```

---

## Data

### Generating Strategy-Labeled NLE Data with AutoAscend

We've provided a script for multi-threaded hierarchical `ttyrec` data generation with AutoAscend, `generate_data.py`.

To generate `$NUM_ROLLOUTS` with `$NUM_CORES`, run

```
python generate_data.py -n $NUM_ROLLOUTS -c $NUM_CORES
```

By default, ttyrecs will be saved to `data/test`.

### Full HiHack Dataset

The full HiHack dataset (**`~97GB` zipped, `~99GB` unzipped**) is now available for public download.

```
wget horatio.cs.nyu.edu/mit/ulyana/hihack/hihack_dataset.tar.gz && tar -xvzf hihack_dataset.tar.gz -C data/
```

Please be aware that download and extraction may take several hours, we recommend that these be run overnight.

Populating the `.db` file registering the `hihack` dataset via NLE-native dataloading functionality (lines `193-196` in `experiment.py`) may also take several hours. Please note that this operation only needs to be run once, as long as the location of `hihack` dataset files remain unchanged.


### Sample HiHack Dataset

A (very) small `HiHack`-style sample dataset consisting of 31, strategy-labeled AutoAscend games can be found in `data/toy_hihack`. We also provide a simple Jupyter notebook loading and visualizing data from this sample as `toy_hihack_explore.ipynb`.

---

## Launching Experiments

To launch an experiment (i.e. train a model from scratch or with warmstarting, via BC or BC + APPO), first confirm all paths have been properly set in `experiment_config.yaml`. Then, run `experiment.py` via a `moolib` broker,

```
python -m moolib.broker &
echo -ne '\n' | sleep 5
export BROKER_IP=`hostname -I | cut -d' ' -f1`
export BROKER_PORT=4431
python experiment.py connect=$BROKER_IP:$BROKER_PORT
```

Experiment code is based on benchmarks introduced in [Hambro et al 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/9d9258fd703057246cb341e615426e2d-Paper-Datasets_and_Benchmarks.pdf).
