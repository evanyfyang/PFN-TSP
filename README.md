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

If you want to specify which GPU to use, add the `CUDA_VISIBLE_DEVICES` environment variable before the command:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python scripts/train_and_evaluate_tsp.py --emsize 256 --nhid 256 --nlayers 3 --nhead 8 --dropout 0.1 --min_nodes 20 --max_nodes 20 --test_size 20 --model_path <model_path> --decoding_strategy greedy_all
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

## Code Guide

All running scripts are located in the `scripts` directory. You can use `train_tsp.sh` for training and `train_and_evaluate_tsp.py` for testing, with parameters as described in the Training and Testing sections above.

The core source code is organized in the `pfns` directory. The modified training main function is `train_tsp()` in `train_tsp.py`, which calls and modifies the original `train()` function. The implementation consists of three main components:

### 1. Data Loader
Located in `/pfns/priors/tsp_data_loader.py`, the data loader generates batch data for each step within an epoch. It handles the TSP instance creation and provides appropriate batches for training.

### 2. Model
The model implementation is in `/pfns/transformer.py`, with the main function being `forward()`. We extended the original code by adding encoders for both inputs (x) and outputs (y). These encoders are implemented in `/pfns/priors/tsp_encoder.py`.

### 3. Encoders
There are two key encoders:
- **TSPGraphEncoder**: This utilizes GNN code from `/pfns/tsp_net.py` to transform TSP instances from point sets to fully-connected graphs, then to graph embeddings (graph_emb or x_encoded) and edge embeddings.
- **TSPTourEncoder**: This combines edge embeddings with output tours (y) to produce y_encoded.

### Model Workflow
In `transformer.py`, these components are integrated to encode each instance. Following the PFN implementation:
- Context positions receive x_encoded + y_encoded (without positional embeddings)
- Target positions receive only x_encoded

After obtaining the transformer output, we extract the target positions and compute attention with all edges of each graph (using torch.einsum) to obtain unnormalized edge values. These edge values and edge information are passed back to the train function for loss calculation.

### Loss Calculation
Since a tour is a sequence, we first convert it to edge labels in the graph. To prevent duplication, we only keep edges from nodes with smaller indices to nodes with larger indices. We then compute the BCE loss with logits, which first applies sigmoid to transform values into probability distributions before calculating BCE loss. To prevent gradient explosion, the total loss is divided by the number of nodes.

## Server Usage Guide

### Accessing the GPU Server

You can access the lab's GPU server using SSH:

```bash
ssh sfu_user_name@cs-airob-gpu01.cmpt.sfu.ca
```

Use your SFU password for authentication. We recommend using VS Code with the Remote-SSH extension for a better development experience.

### CUDA Environment Setup

Before running any code, load the CUDA modules for compatibility with the codebase:

```bash
module load LIB/CUDA/11.8
module load LIB/CUDNN/8.8.0-CUDA11.8
```

For automatic loading of these modules at login, create or edit `~/privatemodule/login` with the following content:

```
#%Module1.0
module load LIB/CUDA/11.8
module load LIB/CUDNN/8.8.0-CUDA11.8
```

### Virtual Environment Setup

Install Conda to manage your virtual environment. Create a new environment with Python 3.10.0:

```bash
conda create -n pfn_tsp python=3.10.0
conda activate pfn_tsp
```

Then proceed with the installation steps as described in the Installation section.

### GPU Management

Before running code, check GPU availability using:

```bash
gpustat
```

or

```bash
nvidia-smi
```

Please use an idle GPU to avoid conflicts with other users.

### CPU Load Management

The dataloader uses multi-threading by default. If you plan to run multiple jobs or notice high CPU load, modify the `num_processes` parameter in the dataloader:

```python
# In pfns/priors/tsp_data_loader.py
num_processes: int = 8  # or 16, depending on server load
```

### Running Jobs in Background

To run training jobs in the background, use:

```bash
nohup bash scripts/train_tsp.sh <your parameters> ><your_outputfile> 2>&1 &
```

This allows you to log out while your job continues running, with all output saved to the specified file.

<!-- Prior-data Fitted Networks (PFNs, https://arxiv.org/abs/2112.10510) are transformer encoders trained to perform supervised in-context learning on datasets randomly drawn from a prior.
// ... existing code ...

