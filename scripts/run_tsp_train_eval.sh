#!/bin/bash
mkdir -p saved_models

# Quick training mode (few steps and epochs, for testing the script)
if [ "$1" == "quick" ]; then
    python -m pdb scripts/train_and_evaluate_tsp.py \
        --epochs 1 \
        --steps_per_epoch 10 \
        --batch_size 16 \
        --min_nodes 5 \
        --max_nodes 10 \
        --test_size 5 \
        --train
    exit 0
fi

# Small-scale problem training
if [ "$1" == "small" ]; then
    echo "Training on small-scale TSP problems (5-15 nodes)..."
    python scripts/train_and_evaluate_tsp.py \
        --emsize 128 \
        --nhid 128 \
        --nlayers 4 \
        --epochs 50 \
        --min_nodes 5 \
        --max_nodes 15 \
        --test_size 5 \
        --train
    exit 0
fi

# Medium-scale problem training
if [ "$1" == "medium" ]; then
    echo "Training on medium-scale TSP problems (15-30 nodes)..."
    python scripts/train_and_evaluate_tsp.py \
        --emsize 200 \
        --nhid 200 \
        --nlayers 6 \
        --epochs 10 \
        --min_nodes 15 \
        --max_nodes 30 \
        --test_size 5 \
        --train
    exit 0
fi

# Large-scale problem training
if [ "$1" == "large" ]; then
    echo "Training on large-scale TSP problems (30-50 nodes)..."
    python scripts/train_and_evaluate_tsp.py \
        --emsize 256 \
        --nhid 256 \
        --nlayers 8 \
        --nhead 8 \
        --epochs 15 \
        --min_nodes 30 \
        --max_nodes 50 \
        --test_size 5 \
        --train
    exit 0
fi

# Default settings
python scripts/train_and_evaluate_tsp.py \
    --emsize 200 \
    --nhid 200 \
    --nlayers 6 \
    --nhead 4 \
    --dropout 0.1 \
    --epochs 10 \
    --steps_per_epoch 100 \
    --batch_size 32 \
    --min_nodes 10 \
    --max_nodes 20 \
    --test_size 20 \
    --train