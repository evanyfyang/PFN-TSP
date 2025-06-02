#!/bin/bash
# Online TSP training script
mkdir -p saved_models

MIN_NODES=10
MAX_NODES=20
GPU=0
MAX_CANDIDATES=5

while getopts "s:l:g:c:" opt; do
  case $opt in
    s) MIN_NODES=$OPTARG ;;
    l) MAX_NODES=$OPTARG ;;
    g) GPU=$OPTARG ;;
    c) MAX_CANDIDATES=$OPTARG ;;
    *) echo "Usage: $0 [-s min_nodes] [-l max_nodes] [-g gpu_id] [-c max_candidates]" >&2
       echo "  -s: minimum number of nodes (default: 10)" >&2
       echo "  -l: maximum number of nodes (default: 20)" >&2
       echo "  -g: GPU ID (default: 0)" >&2
       echo "  -c: max candidates per node for LKH3 (default: 5)" >&2
       exit 1 ;;
  esac
done

echo "Starting TSP online training (nodes: $MIN_NODES-$MAX_NODES)..."
echo "Using GPU: $GPU"
echo "Max candidates: $MAX_CANDIDATES"
echo "Training mode: ONLINE (generating data during training)"

export CUDA_VISIBLE_DEVICES=$GPU

python scripts/train_and_evaluate_tsp.py \
    --training_mode online \
    --emsize 128 \
    --nhid 128 \
    --nlayers 3 \
    --nhead 8 \
    --dropout 0.1 \
    --epochs 10 \
    --steps_per_epoch 100 \
    --batch_size 16 \
    --min_nodes $MIN_NODES \
    --max_nodes $MAX_NODES \
    --max_candidates $MAX_CANDIDATES \
    --test_size 10 \
    --train

echo "Online training completed!" 