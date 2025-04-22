# PFNs

This work follows the PFNs project. 

## Installation

First, install the package in development mode:

```bash
git clone https://github.com/evanyfyang/PFN-TSP.git
cd PFN-TSP
pip install -e .
```

Then, install the required PyTorch Geometric dependencies:

```bash
pip install --no-index torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-geometric==2.0.4
```

**Note:** The torch and CUDA versions in the installation commands should match your system's configuration. The example above uses torch-1.7.0 with CUDA 11.0. Adjust these versions according to your environment.

## Running Training

To train the model for the Traveling Salesman Problem (TSP), use the following command:

```bash
bash scripts/train_tsp.sh -s 20 -l 20 -g 0
```

Where:
- `-s 20`: minimum number of nodes
- `-l 20`: maximum number of nodes
- `-g 0`: specifies GPU 0 for training

## Testing

To evaluate a trained model on the TSP task, use:

```bash
python scripts/train_and_evaluate_tsp.py --emsize 256 --nhid 256 --nlayers 3 --nhead 8 --dropout 0.1 --min_nodes 20 --max_nodes 20 --test_size 20 --model_path <model_path> --decoding_strategy greedy_all
```

Parameters explanation:
- `--min_nodes` and `--max_nodes`: Define the graph size range for testing
- `--emsize`, `--nhid`, `--nlayers`, `--nhead`, `--dropout`: Model architecture parameters (must match the parameters used during training)
- `--test_size`: Number of test instances
- `--model_path`: Path to the trained model file
- `--decoding_strategy`: Method used for solving TSP, available options:
  - `greedy`: Greedy node selection
  - `beam_search`: Beam search for path finding
  - `mcmc`: Monte Carlo Markov Chain sampling
  - `greedy_all`: Greedy algorithm considering all nodes as starting points
  - `beam_search_all`: Beam search considering all nodes as starting points





<!-- Prior-data Fitted Networks (PFNs, https://arxiv.org/abs/2112.10510) are transformer encoders trained to perform supervised in-context learning on datasets randomly drawn from a prior.
Our priors can in general be described by a function that samples a datasets, or more generally a batch of datasets.
The PFN is then trained to predict a hold-out set of labels, given the rest of the dataset.

The pseudo code for a simple prior that would yield a PFN that does 1d ridge regression on datasets with 100 elements, could be something like this:

```python
def get_dataset_sample():
    x = RandomUniform(100,1)
    a = RandomNormal()
    b = RandomNormal()
    y = a * x + b
    return x, y
```

Check out our [tutorial](https://colab.research.google.com/drive/12YpI99LkuFeWcuYHt_idl142DqX7AaJf) to train your own ridge regression PFN.

### Install with pip

This way of installing allows you to use the package everywhere and still be able to edit files.
You should use a python version **>=3.10 and <=3.11**.
```bash
git clone https://github.com/automl/PFNs.git
cd PFNs
pip install -e .
```

### Get Started

Check out our [Getting Started Colab](https://colab.research.google.com/drive/12YpI99LkuFeWcuYHt_idl142DqX7AaJf).

### Tabular Data


For loading the pretrained TabPFN transformer model for classification and use it for evaluation, you can download the model like this

```python
import torch
from pfns.scripts.tabpfn_interface import TabPFNClassifier
# Load pretrained-model
classifier = TabPFNClassifier(base_path='.', model_string="prior_diff_real_checkpoint_n_0_epoch_42.cpkt")

train_xs = torch.rand(100,2)
test_xs = torch.rand(100,2)
train_ys = train_xs.mean(1) > .5
# Fit and evaluate
task_type = 'multiclass'
classifier.fit(train_xs, train_ys)
if task_type == 'multiclass':
    prediction_ = classifier.predict_proba(test_xs) # For survival [:, 1:]
else:
    prediction_ = classifier.predict(test_xs)
```


### BO

There is a BO version of this repo, with pretrained models at [github.com/automl/PFNs4BO](https://github.com/automl/PFNs4BO).
The two repos share a lot of the code, but the other is not anymore actively maintained.
You can also train your own models with our tutorial notebook [here](Tutorial_Training_for_BO.ipynb).

To run all BayesOpt experiments, please install this package with the `benchmarks` option:
```bash
pip install -e .[benchmarks]
```

### Bayes' Power for Explaining In-Context Learning Generalizations

This repository contains the code for the paper "Bayes' Power for Explaining In-Context Learning Generalizations".

Install in editable mode:
```bash
pip install -e .
```

We have a set of notebooks in this repository to reproduce the results of our paper.

- To reproduce the main ICL experiments, use the notebook `discrete_bayes.ipynb`.
- To run the Tiny-MLP generalization experiments, where we evaluate extrapolation, use the notebook `Tiny_MLP_Generalization.ipynb`.
- To run the Coin-Flipping experiments, where we show that the true posterior converges to the wrong probability, use the notebook `Cointhrowing_converging_to_wrong_posterior.ipynb`.
- To see the GP converging to the wrong solution for a step function, use the notebook `GP_fitting_a_step.ipynb`.


### Cite the work

PFNs were introduced in
```
@inproceedings{
    muller2022transformers,
    title={Transformers Can Do Bayesian Inference},
    author={Samuel M{\"u}ller and Noah Hollmann and Sebastian Pineda Arango and Josif Grabocka and Frank Hutter},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=KSugKcbNf9}
}
```

Training PFNs on tabular data (TabPFN) was enhanced in
```
@inproceedings{
  hollmann2023tabpfn,
  title={Tab{PFN}: A Transformer That Solves Small Tabular Classification Problems in a Second},
  author={Noah Hollmann and Samuel M{\"u}ller and Katharina Eggensperger and Frank Hutter},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=cp5PvcI6w8_}
}
```

The BO version of PFNs was introduced in
```
@article{muller2023pfns,
  title={PFNs4BO: In-Context Learning for Bayesian Optimization},
  author={M{\"u}ller, Samuel and Feurer, Matthias and Hollmann, Noah and Hutter, Frank},
  journal={arXiv preprint arXiv:2305.17535},
  year={2023}
}
```

The "Bayes' Power for Explaining In-Context Learning Generalizations" is
```
@article{muller2024bayes,
  title={Bayes' Power for Explaining In-Context Learning Generalizations},
  author={M{\"u}ller, Samuel and Hollmann, Noah and Hutter, Frank},
  journal={arXiv preprint arXiv:2410.01565},
  year={2024}
}
``` -->

