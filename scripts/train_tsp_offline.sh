#!/bin/bash
# Offline TSP training script
mkdir -p saved_models

# Default parameters
MIN_NODES=31
MAX_NODES=80
GPU=0
MAX_CANDIDATES=5
DATASET_PATH=""
EPOCHS=20
BATCH_SIZE=32
EMSIZE=256
NHID=256
NLAYERS=3
NHEAD=8

# Show help information
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -d, --dataset PATH     Path to the pre-generated dataset (required)"
    echo "  -s, --min_nodes NUM   Minimum number of nodes (default: 31)"
    echo "  -l, --max_nodes NUM   Maximum number of nodes (default: 80)"
    echo "  -g, --gpu ID          GPU ID (default: 0)"
    echo "  -c, --candidates NUM  Max candidates per node for LKH3 (default: 5)"
    echo "  -e, --epochs NUM      Number of training epochs (default: 20)"
    echo "  -b, --batch_size NUM  Batch size (default: 32)"
    echo "  --emsize NUM          Embedding size (default: 256)"
    echo "  --nhid NUM            Hidden dimension (default: 256)"
    echo "  --nlayers NUM         Number of layers (default: 3)"
    echo "  --nhead NUM           Number of attention heads (default: 8)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -d /path/to/dataset.pkl -s 30 -l 50 -e 10"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -s|--min_nodes)
            MIN_NODES="$2"
            shift 2
            ;;
        -l|--max_nodes)
            MAX_NODES="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU="$2"
            shift 2
            ;;
        -c|--candidates)
            MAX_CANDIDATES="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --emsize)
            EMSIZE="$2"
            shift 2
            ;;
        --nhid)
            NHID="$2"
            shift 2
            ;;
        --nlayers)
            NLAYERS="$2"
            shift 2
            ;;
        --nhead)
            NHEAD="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_help
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$DATASET_PATH" ]; then
    echo "Error: Dataset path is required!" >&2
    echo "Use -d or --dataset to specify the dataset path." >&2
    show_help
    exit 1
fi

# Check if dataset file exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH" >&2
    exit 1
fi

echo "Starting TSP offline training..."
echo "Dataset: $DATASET_PATH"
echo "Node range: $MIN_NODES-$MAX_NODES"
echo "GPU: $GPU"
echo "Max candidates: $MAX_CANDIDATES"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Model config: emsize=$EMSIZE, nhid=$NHID, nlayers=$NLAYERS, nhead=$NHEAD"
echo "Training mode: OFFLINE (using pre-generated data)"
echo ""

export CUDA_VISIBLE_DEVICES=$GPU

python -m pdb scripts/train_and_evaluate_tsp.py \
    --training_mode offline \
    --dataset_path "$DATASET_PATH" \
    --emsize $EMSIZE \
    --nhid $NHID \
    --nlayers $NLAYERS \
    --nhead $NHEAD \
    --dropout 0.1 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --min_nodes $MIN_NODES \
    --max_nodes $MAX_NODES \
    --max_candidates $MAX_CANDIDATES \
    --test_size 10 \
    --train

echo "Offline training completed!" 