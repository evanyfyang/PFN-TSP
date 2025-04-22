#!/bin/bash
mkdir -p saved_models

MIN_NODES=10
MAX_NODES=20
GPU=0

while getopts "s:l:g:" opt; do
  case $opt in
    s) MIN_NODES=$OPTARG ;;
    l) MAX_NODES=$OPTARG ;;
    g) GPU=$OPTARG ;;
    *) echo "Usage: $0 [-s min_nodes] [-l max_nodes] [-g gpu_id]" >&2
       exit 1 ;;
  esac
done

echo "Starting TSP training (nodes: $MIN_NODES-$MAX_NODES)..."
echo "Using GPU: $GPU"

export CUDA_VISIBLE_DEVICES=$GPU

python scripts/train_and_evaluate_tsp.py \
    --emsize 256 \
    --nhid 256 \
    --nlayers 3 \
    --nhead 8 \
    --dropout 0.1 \
    --epochs 20 \
    --batch_size 32 \
    --min_nodes $MIN_NODES \
    --max_nodes $MAX_NODES \
    --test_size 5 \
    --train

echo "Training completed!" 