#!/bin/bash
# Offline TSP training script - trains TSP models using pre-generated datasets
# Always creates bidirectional edges for optimal GNN performance
mkdir -p saved_models

# Default parameters
MIN_NODES=31
MAX_NODES=80
GPU=0
MAX_CANDIDATES=5
DATASET_PATH=""
EPOCHS=20
BATCH_SIZE=32
EMSIZE=128
NHID=128
NLAYERS=6
NHEAD=8
GENERATION_STRATEGY="random_nodes_same_size"
USE_UNIFIED_ENCODING=false
USE_SHARED_BASIS_FILM=false
USE_INSTANCE_HYPERGRAPH=false
MERGE_DUPLICATE_COORDS=true
LOSS_DIRECTION_MODE="both"  # Control loss calculation direction (bidirectional edges always created)
EDGE_TYPE_MODE="triple"  # Edge type mode for no-merge SharedBasisFiLM: single or triple
PREDICTION_MODE="mlp_concat"  # Prediction mode: 'dot_product' or 'mlp_concat'
USE_RESIDUAL_NORM=true  # Use residual connections and LayerNorm

# Show help information
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -d, --dataset PATH     Path to the pre-generated dataset (required)"
    echo "  -s, --min_nodes NUM   Minimum number of nodes (default: 31)"
    echo "  -l, --max_nodes NUM   Maximum number of nodes (default: 80)"
    echo "  -g, --gpu ID          GPU ID (default: 0)"
    echo "  -c, --candidates NUM  Max candidates per node for graph construction (default: 5, LKH3 uses fixed 5)"
    echo "  -e, --epochs NUM      Number of training epochs (default: 20)"
    echo "  -b, --batch_size NUM  Batch size (default: 32)"
    echo "  --emsize NUM          Embedding size (default: 128)"
    echo "  --nhid NUM            Hidden dimension (default: 128)"
    echo "  --nlayers NUM         Number of layers (default: 3)"
    echo "  --nhead NUM           Number of attention heads (default: 8)"
    echo "  --strategy STR        Generation strategy (random_nodes_same_size or fix_all_nodes_same_size)"
    echo "  --unified             Use unified encoding (combines graph and tour information)"
    echo "  --shared_film         Use SharedBasisFiLM mode (merged large graph processing)"
    echo "  --instance_hypergraph Use InstanceAwareHypergraphGNN mode (mathematical formulation with FiLM and hypergraph GCN)"
    echo "  --no_merge_coords     Disable merging of duplicate coordinates (only for SharedBasisFiLM mode)"
    echo "  --loss_mode STR       Loss direction mode: 'both' (default) or 'forward'"
    echo "                        Note: Bidirectional edges are always created for optimal GNN performance"
  echo "  --edge_type_mode STR  Edge type mode for no-merge SharedBasisFiLM: 'triple' (default) or 'single'"
  echo "                        triple: 3 edge types (graph/tour/center), single: 1 edge type with features"
  echo "  --prediction_mode STR Prediction mode: 'dot_product' (default) or 'mlp_concat'"
  echo "                        dot_product: MLP([x_src, x_dst, w_emb, center]) * MLP(z_emb), mlp_concat: MLP([w_emb, x_emb, y_emb, z_emb])"
  echo "                        Note: Enhanced with embedded distance and s_uvi features"
  echo "  --use_residual_norm    Use residual connections and LayerNorm in InstanceAwareHypergraphGNN (default: false)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -d /path/to/dataset.pkl -s 30 -l 50 -e 10 --strategy fix_all_nodes_same_size"
    echo "  $0 -d /path/to/dataset.pkl --shared_film --no_merge_coords --loss_mode forward"
  echo "  $0 -d /path/to/dataset.pkl --shared_film --no_merge_coords --edge_type_mode single"
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
        --strategy)
            GENERATION_STRATEGY="$2"
            shift 2
            ;;
        --unified)
            USE_UNIFIED_ENCODING=true
            shift
            ;;
        --shared_film)
            USE_SHARED_BASIS_FILM=true
            shift
            ;;
        --instance_hypergraph)
            USE_INSTANCE_HYPERGRAPH=true
            shift
            ;;
        --no_merge_coords)
            MERGE_DUPLICATE_COORDS=false
            shift
            ;;
        --loss_mode)
            LOSS_DIRECTION_MODE="$2"
            shift 2
            ;;
        --edge_type_mode)
            EDGE_TYPE_MODE="$2"
            shift 2
            ;;
        --prediction_mode)
            PREDICTION_MODE="$2"
            shift 2
            ;;
        --use_residual_norm)
            USE_RESIDUAL_NORM=true
            shift
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

# Validate generation strategy
if [ "$GENERATION_STRATEGY" != "random_nodes_same_size" ] && [ "$GENERATION_STRATEGY" != "fix_all_nodes_same_size" ] && [ "$GENERATION_STRATEGY" != "fix_group_nodes_same_size" ] && [ "$GENERATION_STRATEGY" != "sample_from_large" ]; then
    echo "Error: Invalid generation strategy: $GENERATION_STRATEGY" >&2
    echo "Must be one of: 'random_nodes_same_size', 'fix_all_nodes_same_size', 'fix_group_nodes_same_size', 'sample_from_large'" >&2
    exit 1
fi

# Validate loss direction mode
if [ "$LOSS_DIRECTION_MODE" != "both" ] && [ "$LOSS_DIRECTION_MODE" != "forward" ]; then
    echo "Error: Invalid loss direction mode: $LOSS_DIRECTION_MODE" >&2
    echo "Must be one of: 'both', 'forward'" >&2
    exit 1
fi

# Validate edge type mode
if [ "$EDGE_TYPE_MODE" != "single" ] && [ "$EDGE_TYPE_MODE" != "triple" ]; then
    echo "Error: Invalid edge type mode: $EDGE_TYPE_MODE" >&2
    echo "Must be one of: 'single', 'triple'" >&2
    exit 1
fi

# Validate prediction mode
if [ "$PREDICTION_MODE" != "dot_product" ] && [ "$PREDICTION_MODE" != "mlp_concat" ]; then
    echo "Error: Invalid prediction mode: $PREDICTION_MODE" >&2
    echo "Must be one of: 'dot_product', 'mlp_concat'" >&2
    exit 1
fi

# Validate encoding options (cannot use multiple encoding modes simultaneously)
encoding_modes_count=0
if [ "$USE_UNIFIED_ENCODING" = true ]; then
    encoding_modes_count=$((encoding_modes_count + 1))
fi
if [ "$USE_SHARED_BASIS_FILM" = true ]; then
    encoding_modes_count=$((encoding_modes_count + 1))
fi
if [ "$USE_INSTANCE_HYPERGRAPH" = true ]; then
    encoding_modes_count=$((encoding_modes_count + 1))
fi

if [ $encoding_modes_count -gt 1 ]; then
    echo "Error: Cannot use multiple encoding modes simultaneously!" >&2
    echo "Choose only one of: --unified, --shared_film, or --instance_hypergraph" >&2
    exit 1
fi

echo "Starting TSP offline training..."
echo "Dataset: $DATASET_PATH"
echo "Node range: $MIN_NODES-$MAX_NODES"
echo "GPU: $GPU"
echo "Graph max candidates: $MAX_CANDIDATES (LKH3 uses fixed 5)"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Model config: emsize=$EMSIZE, nhid=$NHID, nlayers=$NLAYERS, nhead=$NHEAD"
echo "Generation strategy: $GENERATION_STRATEGY"
echo "Unified encoding: $USE_UNIFIED_ENCODING"
echo "SharedBasisFiLM mode: $USE_SHARED_BASIS_FILM"
echo "InstanceAwareHypergraph mode: $USE_INSTANCE_HYPERGRAPH"
echo "Merge duplicate coords: $MERGE_DUPLICATE_COORDS"
echo "Loss direction mode: $LOSS_DIRECTION_MODE (bidirectional edges always created)"
echo "Edge type mode: $EDGE_TYPE_MODE"
echo "Prediction mode: $PREDICTION_MODE"
echo "Use residual norm: $USE_RESIDUAL_NORM"
echo "Training mode: OFFLINE (using pre-generated data)"
echo ""

export CUDA_VISIBLE_DEVICES=$GPU
export TORCH_CUDNN_BENCHMARK=True
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.8,expandable_segments:True
export TORCH_CUDNN_ALLOW_TF32=1
export OMP_NUM_THREADS=4
export TORCH_COMPILE_MODE=reduce-overhead
# Debugging/diagnostics (safe to keep enabled; may slow down execution)
export TSP_DEBUG_GUARDS=${TSP_DEBUG_GUARDS:-0}
if [ -n "$TORCH_SHOW_CPP_STACKTRACES" ]; then export TORCH_SHOW_CPP_STACKTRACES=$TORCH_SHOW_CPP_STACKTRACES; fi
if [ -n "$CUDA_LAUNCH_BLOCKING" ]; then export CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING; fi
if [ -n "$TORCH_USE_CUDA_DSA" ]; then export TORCH_USE_CUDA_DSA=$TORCH_USE_CUDA_DSA; fi
export TSP_AGGREGATION_MODE=${TSP_AGGREGATION_MODE:-instance_first}
export TSP_SHARE_FILM=${TSP_SHARE_FILM:-1}
export TSP_ENABLE_FILM=${TSP_ENABLE_FILM:-1}
export TSP_STEP1_USE_Z=${TSP_STEP1_USE_Z:-1}
export TSP_PER_INSTANCE_EMBNET=${TSP_PER_INSTANCE_EMBNET:-0}

TRAIN_CMD="python scripts/train_and_evaluate_tsp.py \
    --training_mode offline \
    --dataset_path \"$DATASET_PATH\" \
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
    --generation_strategy $GENERATION_STRATEGY \
    --loss_direction_mode $LOSS_DIRECTION_MODE \
    --edge_type_mode $EDGE_TYPE_MODE \
    --prediction_mode $PREDICTION_MODE \
    --train"

if [ "$USE_UNIFIED_ENCODING" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use_unified_encoding"
fi

if [ "$USE_SHARED_BASIS_FILM" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use_shared_basis_film"
    
    if [ "$MERGE_DUPLICATE_COORDS" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --merge_duplicate_coords"
    fi
fi

if [ "$USE_INSTANCE_HYPERGRAPH" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use_instance_hypergraph"
fi

# Add residual norm option
if [ "$USE_RESIDUAL_NORM" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use_residual_norm"
fi

eval $TRAIN_CMD

echo "Offline training completed!" 