#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from datetime import datetime
import torch.nn.functional as F
import pickle
import random

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pfns.train_tsp import train_tsp
from pfns.priors.tsp_data_loader import TSPDataLoader, solve_tsp_ortools, solve_tsp_static_with_or_tools_and_initial_solutions, solve_2_opt_with_initial_solutions
from pfns.priors.tsp_offline_data_loader import TSPOfflineDataLoader
from pfns.priors.prior import Batch
from pfns.priors.tsp_decoding_strategies import *
from pfns.priors.tsp_encoder import tsp_graph_encoder_generator, tsp_tour_encoder_generator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train and evaluate PFN model for TSP')
    parser.add_argument('--emsize', type=int, default=128, help='Embedding dimension size')
    parser.add_argument('--nhid', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--nlayers', type=int, default=3, help='Number of Transformer layers')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='Steps per epoch (ignored for offline mode)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--min_nodes', type=int, default=10, help='Minimum number of nodes in TSP')
    parser.add_argument('--max_nodes', type=int, default=20, help='Maximum number of nodes in TSP')
    parser.add_argument('--max_candidates', type=int, default=5, help='Maximum number of candidates per node for LKH3')
    parser.add_argument('--test_size', type=int, default=10, help='Number of seq len')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save models')
    parser.add_argument('--cuda_device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--train', action='store_true', help='Whether to train the model (otherwise just test)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pretrained model for testing')
    parser.add_argument('--decoding_strategy', type=str, default='greedy', 
                        choices=['greedy', 'beam_search', 'mcmc', 'greedy_all', 'greedy_edge', 'sampling','sampling_edge', 'sampling_all', 'beam_search_all', 'greedy_edge', 'mcts', 'mcts_all'], 
                        help='Decoding strategy for TSP')
    parser.add_argument('--test_instances', type=int, default=1, help='Number of test instances')
    parser.add_argument('--use_complete_graph', action='store_true', default=False, 
                        help='Use complete graph instead of candidate edges (for comparison)')
    parser.add_argument('--progress_bar', action='store_true', default=True,
                        help='Show progress bar during training')
    parser.add_argument('--use_unified_encoding', action='store_true', default=False,
                        help='Use unified encoding that combines graph and tour information')
    parser.add_argument('--use_shared_basis_film', action='store_true', default=False,
                        help='Use SharedBasisFiLM mode for merged large graph processing')
    parser.add_argument('--merge_duplicate_coords', action='store_true', default=True,
                        help='Merge nodes with identical coordinates when using SharedBasisFiLM mode')
    
    # Add loss direction control argument (bidirectional edges are always created)
    parser.add_argument('--loss_direction_mode', type=str, default='both', choices=['both', 'forward'],
                        help='How to handle edge directions in loss calculation (default: both). Note: bidirectional edges are always created for optimal GNN performance.')
    
    # Add online/offline training mode arguments
    parser.add_argument('--training_mode', type=str, default='online', choices=['online', 'offline'],
                        help='Training mode: online (generate data during training) or offline (use pre-generated data)')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to pre-generated dataset (required for offline mode)')
    
    # Add generation strategy arguments
    parser.add_argument('--generation_strategy', type=str, default='random_nodes_same_size',
                        choices=['random_nodes_same_size', 'fix_all_nodes_same_size', 'fix_group_nodes_same_size', 'sample_from_large'],
                        help='Strategy for generating TSP instances')
    
    # Add test dataset argument
    parser.add_argument('--test_dataset_path', type=str, default=None,
                        help='Path to pre-generated test dataset (if not provided, will generate test instances on the fly)')
    
    return parser.parse_args()

def train_tsp_model(args):
    """Train the TSP model with online or offline mode"""
    print(f"Starting TSP model training in {args.training_mode} mode...")
    print(f"Parameters: {args}")
    
    # Validate arguments based on training mode
    if args.training_mode == 'offline':
        if args.dataset_path is None:
            raise ValueError("--dataset_path is required for offline training mode")
        if not os.path.exists(args.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Prepare training arguments based on mode
    train_kwargs = {
        'emsize': args.emsize,
        'nhid': args.nhid,
        'nlayers': args.nlayers,
        'nhead': args.nhead,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'seq_len': args.test_size,
        'lr': args.lr,
        'num_nodes_range': (args.min_nodes, args.max_nodes),
        'gpu_device': args.cuda_device,
        'progress_bar': args.progress_bar,
        'verbose': True,
        'max_candidates': args.max_candidates,
        'generation_strategy': args.generation_strategy,
        'single_eval_pos': args.test_size - 1,  # Always evaluate the last position
        'use_unified_encoding': args.use_unified_encoding,
        'use_shared_basis_film': args.use_shared_basis_film,
        'merge_duplicate_coords': args.merge_duplicate_coords,
        'loss_direction_mode': args.loss_direction_mode  # Only loss direction control, bidirectional edges always created
    }
    
    # Add mode-specific arguments
    if args.training_mode == 'online':
        print("Using online training mode - generating data during training")
        train_kwargs['steps_per_epoch'] = args.steps_per_epoch
        # Use default TSPDataLoader (online mode)
        
    else:  # offline mode
        print(f"Using offline training mode - loading data from {args.dataset_path}")
        # Add offline-specific parameters
        train_kwargs['extra_prior_kwargs_dict'] = {
            'dataset_path': args.dataset_path,
            'shuffle': True,
            'generation_strategy': args.generation_strategy,
            'fix_last_instance': True  # Flag to indicate we want to fix the last instance
        }
        # Override the dataloader class to use offline loader
        train_kwargs['priordataloader_class'] = TSPOfflineDataLoader
    
    # Log edge configuration
    print(f"Training TSP model on {args.cuda_device} with {args.emsize} embedding size")
    print(f"Edge configuration:")
    print(f"  - Always create bidirectional edges: True (optimal for GNN performance)")
    print(f"  - Loss direction mode: {train_kwargs['loss_direction_mode']}")
    
    if train_kwargs['loss_direction_mode'] == 'both':
        print("  ✓ Optimal configuration: bidirectional edge creation + bidirectional loss calculation")
    else:
        print("  ✓ Standard configuration: bidirectional edge creation + forward loss calculation")
    
    # Train the model
    start_time = time.time()
    result = train_tsp(**train_kwargs)
    training_time = time.time() - start_time
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = args.training_mode
    strategy_suffix = args.generation_strategy
    model_save_path = os.path.join(args.save_dir, f"tsp_model_{mode_suffix}_{strategy_suffix}_{timestamp}.pt")
    torch.save(result.model.state_dict(), model_save_path)
    
    print(f"Training completed in {training_time:.2f} seconds, model saved to {model_save_path}")
    
    return result.model.to(args.cuda_device), model_save_path

def predict_tsp_with_pfn(model, coords, solution, candidate_info=None, use_complete_graph=False, device='cuda', decoding_strategy='greedy'):
    """Predict TSP tour using the trained PFN model"""
    model.eval()
    
    # coords and solution are now lists, each element corresponds to a TSP instance
    # We need to process the last instance for prediction
    last_coords = coords[-1]  # Coordinates of the last TSP instance
    last_solution = solution[-1]  # Solution of the last TSP instance
    
    # Build input tensor - need to include all sequence positions
    seq_len = len(coords)
    
    # Find the maximum number of nodes across all instances for proper padding
    max_nodes = max(len(coord) for coord in coords)
    
    # Create complete x tensor (seq_len, 1, max_nodes, 2) with proper padding
    x = torch.full((seq_len, 1, max_nodes, 2), -1.0, dtype=torch.float32, device=device)
    y = torch.full((seq_len, 1, max_nodes), -1, dtype=torch.long, device=device)
    
    for i, (coord, sol) in enumerate(zip(coords, solution)):
        num_nodes_in_instance = len(coord)
        x[i, 0, :num_nodes_in_instance] = torch.tensor(coord, dtype=torch.float32, device=device)
        y[i, 0, :num_nodes_in_instance] = torch.tensor(sol, dtype=torch.long, device=device)
    
    # Step by step prediction
    with torch.no_grad():
        # Pass candidate_info to the model only if not using complete graph
        if use_complete_graph:
            print("Using complete graph for inference...")
            outputs = model((None, x, y), single_eval_pos=seq_len-1, candidate_info=None)
        else:
            print("Using candidate edges for inference...")
            outputs = model((None, x, y), single_eval_pos=seq_len-1, candidate_info=candidate_info)
        
        # Handle different output formats (standard vs SharedBasisFiLM)
        if len(outputs) == 2:
            edge_values_padded, edge_info = outputs
            
            # Check if edge_info is a tuple (new format) or dict (old format)
            if isinstance(edge_info, tuple):
                # New edge_info format is [edge_index_list, node_offset_map, edge_counts]
                edge_index_list, node_offset_map, edge_counts = edge_info
            else:
                # Old edge_info format is a dict, need to extract info
                eval_infos = edge_info.get('eval_infos', [])
                if eval_infos:
                    # Extract info from eval_infos
                    edge_index_list = []
                    edge_counts = []
                    node_offset_map = {}
                    
                    for eval_info in eval_infos:
                        edge_index = eval_info.get('edge_index')
                        if edge_index is not None:
                            edge_index_list.append(edge_index.t())  # Convert to [num_edges, 2]
                            edge_counts.append(edge_index.size(1))
                        else:
                            edge_index_list.append(torch.empty((0, 2), dtype=torch.long, device=device))
                            edge_counts.append(0)
                        
                        # Update node_offset_map
                        if 'node_offset_map' in eval_info:
                            node_offset_map.update(eval_info['node_offset_map'])
                        
                        # Handle SharedBasisFiLM mode special mappings
                        if eval_info.get('is_shared_basis_film', False):
                            global_to_originals = eval_info.get('global_to_originals', {})
                            instance_mapping = eval_info.get('instance_mapping', {})
                            pos = eval_info['pos']
                            batch = eval_info['batch']
                            
                            # Create reverse mapping for SharedBasisFiLM mode
                            for global_idx, originals_list in global_to_originals.items():
                                for orig_pos, orig_batch, orig_node in originals_list:
                                    if orig_pos == pos and orig_batch == batch:
                                        node_offset_map[(pos, batch, orig_node)] = global_idx
                else:
                    # Fallback for other edge_info formats
                    edge_index_list = []
                    edge_counts = []
                    node_offset_map = edge_info.get('node_offset_map', {})
        else:
            raise ValueError(f"Unexpected model output format: {len(outputs)} elements")
        
        # Get results for the last evaluation position
        # edge_values_padded shape is [seq_eval_len, batch_size, max_edges]
        # We take the first element (0) because seq_eval_len=1, batch_size=1
        last_edge_values = edge_values_padded[0, 0]
        
        # Get edge_index for the last evaluation position
        if len(edge_index_list) > 0:
            last_edge_index = edge_index_list[-1]
            valid_edge_count = edge_counts[-1] if edge_counts else 0
        else:
            # No edges available, create empty adjacency list
            num_nodes = len(last_coords)
            adj_list = [[] for _ in range(num_nodes)]
            tour = list(range(num_nodes))  # Default fallback tour
            total_distance = calculate_tour_length(last_coords, tour)
            return tour, total_distance
        
        # Apply sigmoid activation to get probabilities
        edge_values = F.sigmoid(last_edge_values)
        
        # Build node mapping
        node_map = {value: key for key, value in node_offset_map.items()}
        edge_index_np = last_edge_index.cpu().numpy()
        edge_values_np = edge_values.cpu().numpy()
        
        # Get the number of nodes in the last instance
        num_nodes = len(last_coords)
        
        # Build adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        
        # First pass: collect all edge probabilities
        edge_probs = {}  # (u, v) -> list of probabilities
        
        for i in range(min(valid_edge_count, len(edge_values_np))):
            # Check edge_index shape and access correctly
            if edge_index_np.ndim == 2 and edge_index_np.shape[0] == 2:
                # edge_index is in 2xE format
                u, v = edge_index_np[0, i], edge_index_np[1, i]
            elif edge_index_np.ndim == 2 and edge_index_np.shape[1] == 2:
                # edge_index is in Ex2 format
                u, v = edge_index_np[i, 0], edge_index_np[i, 1]
            else:
                print(f"Warning: Unexpected edge_index shape: {edge_index_np.shape}")
                continue
            
            # Use node_map to map global node indices back to actual node indices in the problem
            if u in node_map and v in node_map:
                u_info = node_map[u]  # (pos, batch, node)
                v_info = node_map[v]
                
                # Ensure nodes are from the last position
                if u_info[0] == seq_len-1 and v_info[0] == seq_len-1:
                    u_node = u_info[2]
                    v_node = v_info[2]
                    
                    # Check if nodes are within valid range
                    if 0 <= u_node < num_nodes and 0 <= v_node < num_nodes:
                        # Store edge probability for both directions
                        prob = edge_values_np[i]
                        edge_key = (min(u_node, v_node), max(u_node, v_node))
                        
                        if edge_key not in edge_probs:
                            edge_probs[edge_key] = []
                        edge_probs[edge_key].append(prob)
            else:
                # Fallback: try to use indices directly if they're in valid range
                if 0 <= u < num_nodes and 0 <= v < num_nodes:
                    prob = edge_values_np[i]
                    edge_key = (min(u, v), max(u, v))
                    
                    if edge_key not in edge_probs:
                        edge_probs[edge_key] = []
                    edge_probs[edge_key].append(prob)
        
        # Second pass: build adjacency list with averaged probabilities
        for (u_node, v_node), probs in edge_probs.items():
            # Calculate mean probability for both directions
            avg_prob = np.mean(probs)
            
            adj_list[u_node].append((v_node, avg_prob))
            adj_list[v_node].append((u_node, avg_prob))
        
        # Use appropriate decoding strategy
        if decoding_strategy == 'greedy':
            tour = greedy_decode(adj_list, num_nodes)
        elif decoding_strategy == 'greedy_all':
            tour = greedy_all_decode(adj_list, num_nodes)
        elif decoding_strategy == 'greedy_edge':
            tour = greedy_edge_decode(adj_list, num_nodes)
        elif decoding_strategy == 'beam_search':
            tour = beam_search_decode(adj_list, num_nodes)
        elif decoding_strategy == 'sampling':
            tour = sampling_decode(adj_list, num_nodes)
        elif decoding_strategy == 'sampling_all':
            tour = sampling_all_decode(adj_list, num_nodes)
        elif decoding_strategy == 'sampling_edge':
            tour = sampling_edge_decode(adj_list, num_nodes)
        elif decoding_strategy == 'beam_search_all':
            tour = beam_search_all_decode(adj_list, num_nodes)
        elif decoding_strategy == 'mcmc':
            tour = mcmc_decode(adj_list, node_map, edge_index_np, edge_values_np, num_nodes)
        elif decoding_strategy == 'greedy_edge':
            tour = greedy_edge_decode(adj_list, num_nodes)
        elif decoding_strategy == 'mcts':
            tour = mcts_decode(adj_list, num_nodes)
        elif decoding_strategy == 'mcts_all':
            tour = mcts_all_decode(adj_list, num_nodes)
        else:
            raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")
    
    # Calculate path length
    total_distance = calculate_tour_length(last_coords, tour)
    
    return tour, total_distance

def generate_test_instances_with_ortools(num_instances, num_nodes_range, max_candidates=15, device='cpu', test_instances=20):
    """Generate test instances and their corresponding OR-Tools solutions using TSPDataLoader"""
    
    def dummy_sampler():
        """Placeholder function, returns fixed values"""
        return 0, num_nodes_range[1]
        
    # Create TSPDataLoader instance
    dataloader = TSPDataLoader(
        num_steps=1,  
        batch_size=1,
        eval_pos_seq_len_sampler=dummy_sampler,
        seq_len_maximum=num_instances,
        device=device,
        num_nodes_range=num_nodes_range,
        include_ortools=True,  # Enable OR-Tools solutions
        max_candidates=max_candidates
    )
    
    # Get a batch of data
    
    test_instances_gen = []
    ortools_solutions = []
    ortools_times = []
    lkh_solutions = []
    candidate_infos = []  # Add candidate_info storage
    
    # Extract coordinates and solutions
    for i in range(test_instances): 
        batch = next(iter(dataloader))
        coords = batch.x[:, 0, :].cpu().numpy()
        solution = batch.target_y[:, 0].cpu().numpy()
        
        # Use the OR-Tools solution
        ortools_solution = batch.ortools_solution[:, 0].cpu().numpy()

        test_instances_gen.append(coords)
        ortools_solutions.append(ortools_solution)
        lkh_solutions.append(solution)
        candidate_infos.append(batch.candidate_info)  # Save candidate_info
        # Approximate OR-Tools time per instance
        ortools_times.append(batch.ortools_solve_time[-1])
    
    print(f"Average OR-Tools processing time: {np.mean(ortools_times):.4f} seconds")
    
    return test_instances_gen, lkh_solutions, ortools_solutions, ortools_times, candidate_infos

def plot_tour(coords, tour, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=40)
    
    for i in range(len(tour) - 1):
        ax.plot([coords[tour[i], 0], coords[tour[i+1], 0]], 
                [coords[tour[i], 1], coords[tour[i+1], 1]], 'k-', alpha=0.7)
    ax.plot([coords[tour[-1], 0], coords[tour[0], 0]], 
            [coords[tour[-1], 1], coords[tour[0], 1]], 'k-', alpha=0.7)
    
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), fontsize=12)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    return ax

def calculate_tour_length(coords, tour):
    """Calculate total length of a TSP tour"""
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += np.linalg.norm(coords[tour[i]] - coords[tour[i+1]])
    total_distance += np.linalg.norm(coords[tour[-1]] - coords[tour[0]])
    return total_distance

def load_test_instances_from_dataset(test_dataset_path, test_instances=10, generation_strategy='random_nodes_same_size', test_size=10, min_nodes=None, max_nodes=None):
    """
    Load test instances from pre-generated test dataset with improved target selection strategy.
    
    Args:
        test_dataset_path: Path to the pre-generated test dataset
        test_instances: Total number of test instances to return (across all node sizes)
        generation_strategy: Strategy used for generating the dataset
        test_size: Maximum number of instances to keep per group (for group-based strategies)
        min_nodes: Minimum number of nodes to include (filter by node range)
        max_nodes: Maximum number of nodes to include (filter by node range)
        
    Returns:
        test_instances_gen: List of test instance sequences (total length = test_instances)
        lkh_solutions: List of LKH solution sequences (total length = test_instances)
        candidate_infos: List of candidate information sequences (total length = test_instances)
    """
    print(f"Loading test instances from {test_dataset_path}...")
    print(f"Target total test instances: {test_instances}")
    print(f"Generation strategy: {generation_strategy}")
    if min_nodes is not None and max_nodes is not None:
        print(f"Node range filter: {min_nodes}-{max_nodes}")
    
    with open(test_dataset_path, 'rb') as f:
        test_dataset = pickle.load(f)
    
    # Filter by node range if specified
    if min_nodes is not None and max_nodes is not None:
        filtered_dataset = {}
        for num_nodes in range(min_nodes, max_nodes + 1):
            if num_nodes in test_dataset:
                filtered_dataset[num_nodes] = test_dataset[num_nodes]
        test_dataset = filtered_dataset
        print(f"Filtered dataset to include nodes {min_nodes}-{max_nodes}")
    
    # Collect all available test sequences with improved target selection
    all_test_sequences = []
    
    # For fix_all_nodes_same_size with specific node range, focus on that range
    if generation_strategy == 'fix_all_nodes_same_size' and min_nodes is not None and max_nodes is not None:
        # Collect all instances from the specified node range
        all_instances_in_range = []
        for num_nodes in range(min_nodes, max_nodes + 1):
            if num_nodes in test_dataset:
                instances = test_dataset[num_nodes]
                for instance in instances:
                    instance['num_nodes'] = num_nodes  # Ensure num_nodes is stored
                    all_instances_in_range.append(instance)
        
        print(f"Total instances in range {min_nodes}-{max_nodes}: {len(all_instances_in_range)}")
        
        if len(all_instances_in_range) == 0:
            print(f"Warning: No instances found in range {min_nodes}-{max_nodes}")
            return [], [], []
        
        # Select target instances from the specified range
        if len(all_instances_in_range) >= test_instances:
            # Use the last test_instances as fixed targets
            target_instances = all_instances_in_range[-test_instances:]
        else:
            # Use all available instances as targets
            target_instances = all_instances_in_range
        
        print(f"Selected {len(target_instances)} target instances from range {min_nodes}-{max_nodes}")
        
        for target_instance in target_instances:
            # Randomly select test_size-1 instances from the same range as context
            target_id = target_instance.get('instance_id', id(target_instance))
            remaining_instances = [inst for inst in all_instances_in_range if inst.get('instance_id', id(inst)) != target_id]
            
            if len(remaining_instances) > 0:
                context_count = min(test_size-1, len(remaining_instances))
                context_instances = random.sample(remaining_instances, context_count)
                # Add the target instance at the end
                selected_instances = context_instances + [target_instance]
            else:
                selected_instances = [target_instance]
            
            # Create sequence
            coords_sequence = [instance['coords'] for instance in selected_instances]
            tour_sequence = [instance['tour'] for instance in selected_instances]
            candidate_sequence = [instance['candidate_info'] for instance in selected_instances]
            
            all_test_sequences.append({
                'coords': coords_sequence,
                'tours': tour_sequence,
                'candidates': candidate_sequence,
                'num_nodes': target_instance['num_nodes'],
                'instance_id': target_instance.get('instance_id', 0),
                'target_type': 'fixed_range'
            })
    else:
        # Original logic for other strategies
        # Process each node count
        for num_nodes in sorted(test_dataset.keys()):
            # Apply node range filter if specified
            if min_nodes is not None and max_nodes is not None:
                if not (min_nodes <= num_nodes <= max_nodes):
                    continue
            
            instances = test_dataset[num_nodes]
            if len(instances) == 0:
                continue
                
            print(f"Processing TSP-{num_nodes}...")
            
            # Check if this is a group-based structure
            is_group_based = isinstance(instances[0], list)
            
            if is_group_based:
                # For group-based structures
                if generation_strategy == 'fix_group_nodes_same_size':
                    # For fix_group_nodes_same_size: select each group's last instance as fixed target
                    for group_idx, group in enumerate(instances):
                        if len(group) == 0:
                            continue
                        
                        # Use the last instance in the group as the fixed target
                        target_instance = group[-1]
                        
                        # Randomly select test_size-1 instances from the rest as context
                        if len(group) > 1:
                            context_indices = random.sample(range(len(group)-1), min(test_size-1, len(group)-1))
                            context_instances = [group[i] for i in context_indices]
                            # Add the target instance at the end
                            selected_instances = context_instances + [target_instance]
                        else:
                            selected_instances = [target_instance]
                        
                        # Create sequence
                        coords_sequence = [instance['coords'] for instance in selected_instances]
                        tour_sequence = [instance['tour'] for instance in selected_instances]
                        candidate_sequence = [instance['candidate_info'] for instance in selected_instances]
                        
                        all_test_sequences.append({
                            'coords': coords_sequence,
                            'tours': tour_sequence,
                            'candidates': candidate_sequence,
                            'num_nodes': num_nodes,
                            'group_idx': group_idx,
                            'target_type': 'group_last'
                        })
                
                elif generation_strategy == 'sample_from_large':
                    # For sample_from_large: select the base graph as fixed target
                    for group_idx, group in enumerate(instances):
                        if len(group) == 0:
                            continue
                        
                        # Find the base instance (marked with is_base=True or largest instance)
                        base_instance = None
                        for inst in group:
                            if inst.get('is_base', False):
                                base_instance = inst
                                break
                        
                        # If no marked base instance, find the largest one
                        if base_instance is None:
                            base_instance = max(group, key=lambda x: len(x['coords']))
                        
                        # Use the base instance as the fixed target
                        target_instance = base_instance
                        
                        # Get all non-base instances as potential context
                        remaining_instances = [inst for inst in group if inst != base_instance]
                        if len(remaining_instances) > 0:
                            context_count = min(test_size-1, len(remaining_instances))
                            context_instances = random.sample(remaining_instances, context_count)
                            # Add the target instance at the end
                            selected_instances = context_instances + [target_instance]
                        else:
                            selected_instances = [target_instance]
                        
                        # Create sequence
                        coords_sequence = [instance['coords'] for instance in selected_instances]
                        tour_sequence = [instance['tour'] for instance in selected_instances]
                        candidate_sequence = [instance['candidate_info'] for instance in selected_instances]
                        
                        all_test_sequences.append({
                            'coords': coords_sequence,
                            'tours': tour_sequence,
                            'candidates': candidate_sequence,
                            'num_nodes': num_nodes,
                            'group_idx': group_idx,
                            'target_type': 'base_graph',
                            'target_nodes': len(target_instance['coords'])  # Record actual target size
                        })
            else:
                # For flat list structure (random_nodes_same_size or fix_all_nodes_same_size)
                if generation_strategy == 'fix_all_nodes_same_size':
                    # For fix_all_nodes_same_size: select fixed last test_instances as targets
                    if len(instances) >= test_instances:
                        # Use the last test_instances as fixed targets
                        target_instances = instances[-test_instances:]
                        
                        for target_instance in target_instances:
                            # Randomly select test_size-1 instances from the rest as context
                            target_id = target_instance.get('instance_id', id(target_instance))
                            remaining_instances = [inst for inst in instances if inst.get('instance_id', id(inst)) != target_id]
                            if len(remaining_instances) > 0:
                                context_count = min(test_size-1, len(remaining_instances))
                                context_instances = random.sample(remaining_instances, context_count)
                                # Add the target instance at the end
                                selected_instances = context_instances + [target_instance]
                            else:
                                selected_instances = [target_instance]
                            
                            # Create sequence
                            coords_sequence = [instance['coords'] for instance in selected_instances]
                            tour_sequence = [instance['tour'] for instance in selected_instances]
                            candidate_sequence = [instance['candidate_info'] for instance in selected_instances]
                            
                            all_test_sequences.append({
                                'coords': coords_sequence,
                                'tours': tour_sequence,
                                'candidates': candidate_sequence,
                                'num_nodes': num_nodes,
                                'instance_id': target_instance.get('instance_id', 0),
                                'target_type': 'fixed_last'
                            })
                    else:
                        # If not enough instances, use all as targets
                        for target_instance in instances:
                            coords_sequence = [target_instance['coords']]
                            tour_sequence = [target_instance['tour']]
                            candidate_sequence = [target_instance['candidate_info']]
                            
                            all_test_sequences.append({
                                'coords': coords_sequence,
                                'tours': tour_sequence,
                                'candidates': candidate_sequence,
                                'num_nodes': num_nodes,
                                'instance_id': target_instance.get('instance_id', 0),
                                'target_type': 'single'
                            })
                elif generation_strategy == 'fix_group_nodes_same_size':
                    # Special handling for fix_group_nodes_same_size with flat structure
                    # Treat the flat list as if it were groups and create proper sequences
                    print(f"  Handling fix_group_nodes_same_size with flat structure")
                    
                    # Use test_size as instances_per_group for creating sequences
                    instances_per_group = test_size
                    num_possible_groups = len(instances) // instances_per_group
                    num_test_sequences = min(test_instances, num_possible_groups)
                    
                    print(f"  Using instances_per_group={instances_per_group} (test_size)")
                    print(f"  Creating {num_test_sequences} test sequences from {num_possible_groups} possible groups")
                    
                    for seq_idx in range(num_test_sequences):
                        # Select a "group" of instances (consecutive instances)
                        group_start = seq_idx * instances_per_group
                        group_end = min(group_start + instances_per_group, len(instances))
                        group_instances = instances[group_start:group_end]
                        
                        if len(group_instances) == 0:
                            continue
                        
                        # Use the last instance in this "group" as target
                        target_instance = group_instances[-1]
                        
                        # Randomly select test_size-1 instances from the rest as context
                        if len(group_instances) > 1:
                            context_indices = random.sample(range(len(group_instances)-1), min(test_size-1, len(group_instances)-1))
                            context_instances = [group_instances[i] for i in context_indices]
                            # Add the target instance at the end
                            selected_instances = context_instances + [target_instance]
                        else:
                            selected_instances = [target_instance]
                        
                        # Create sequence
                        coords_sequence = [instance['coords'] for instance in selected_instances]
                        tour_sequence = [instance['tour'] for instance in selected_instances]
                        candidate_sequence = [instance['candidate_info'] for instance in selected_instances]
                        
                        all_test_sequences.append({
                            'coords': coords_sequence,
                            'tours': tour_sequence,
                            'candidates': candidate_sequence,
                            'num_nodes': num_nodes,
                            'group_idx': seq_idx,
                            'target_type': 'group_last_flat'
                        })
                else:
                    # For random_nodes_same_size: each instance becomes a single-element sequence
                    for instance in instances:
                        coords_sequence = [instance['coords']]
                        tour_sequence = [instance['tour']]
                        candidate_sequence = [instance['candidate_info']]
                        
                        all_test_sequences.append({
                            'coords': coords_sequence,
                            'tours': tour_sequence,
                            'candidates': candidate_sequence,
                            'num_nodes': num_nodes,
                            'instance_id': instance.get('instance_id', 0),
                            'target_type': 'single'
                        })
    
    print(f"Total available test sequences: {len(all_test_sequences)}")
    
    # Select test sequences based on strategy
    if generation_strategy in ['fix_group_nodes_same_size', 'sample_from_large']:
        # For group-based strategies, we already have the right number of sequences
        selected_sequences = all_test_sequences[:test_instances] if len(all_test_sequences) > test_instances else all_test_sequences
        print(f"Selected {len(selected_sequences)} test sequences from group-based strategy")
    elif generation_strategy == 'fix_all_nodes_same_size' and min_nodes is not None and max_nodes is not None:
        # For fix_all_nodes_same_size with specific range, we already selected the right sequences
        selected_sequences = all_test_sequences
        print(f"Selected {len(selected_sequences)} test sequences from specified node range")
    elif generation_strategy == 'fix_all_nodes_same_size':
        # For fix_all_nodes_same_size without range, we already selected fixed targets
        selected_sequences = all_test_sequences[:test_instances] if len(all_test_sequences) > test_instances else all_test_sequences
        print(f"Selected {len(selected_sequences)} test sequences with fixed targets")
    else:
        # For random_nodes_same_size, randomly select
        if len(all_test_sequences) > test_instances:
            selected_sequences = random.sample(all_test_sequences, test_instances)
            print(f"Randomly selected {test_instances} test sequences")
        else:
            selected_sequences = all_test_sequences
            print(f"Using all {len(selected_sequences)} available test sequences")
    
    # Extract final outputs
    test_instances_gen = [seq['coords'] for seq in selected_sequences]
    lkh_solutions = [seq['tours'] for seq in selected_sequences]
    candidate_infos = [seq['candidates'] for seq in selected_sequences]
    
    # Print distribution by node size and target type
    node_distribution = {}
    target_type_distribution = {}
    for seq in selected_sequences:
        num_nodes = seq['num_nodes']
        target_type = seq.get('target_type', 'unknown')
        
        node_distribution[num_nodes] = node_distribution.get(num_nodes, 0) + 1
        target_type_distribution[target_type] = target_type_distribution.get(target_type, 0) + 1
    
    print(f"Selected test instances distribution by node size:")
    for num_nodes in sorted(node_distribution.keys()):
        count = node_distribution[num_nodes]
        print(f"  TSP-{num_nodes}: {count} sequences")
    
    print(f"Selected test instances distribution by target type:")
    for target_type in sorted(target_type_distribution.keys()):
        count = target_type_distribution[target_type]
        print(f"  {target_type}: {count} sequences")
    
    print(f"Total loaded test sequences: {len(test_instances_gen)}")
    return test_instances_gen, lkh_solutions, candidate_infos

def evaluate_and_compare(model, test_instances, lkh_solutions, candidate_infos, use_complete_graph=False, device='cuda', decoding_strategy='greedy', save_plot=True, plot_path='tsp_comparison.png'):
    """
    Evaluate model and compare with OR-Tools.
    
    Test Flow:
    1. Load test instances from dataset (sequences of TSP instances) - DONE
    2. For each sequence, use the last instance for evaluation
    3. Compute OR-Tools solution for the last instance and record time
    4. Use model to predict solution for the last instance and record time
    5. Compare PFN vs OR-Tools results
    
    Args:
        model: Trained PFN model
        test_instances: List of coordinate sequences
        lkh_solutions: List of LKH solution sequences
        candidate_infos: List of candidate info sequences
        use_complete_graph: Whether to use complete graph instead of candidate edges
        device: Computing device
        decoding_strategy: Decoding strategy for TSP
        save_plot: Whether to save comparison plots
        plot_path: Path to save plots
        
    Returns:
        Dictionary containing evaluation results
    """
    print(f"Starting model evaluation with {decoding_strategy} decoding strategy...")
    if use_complete_graph:
        print("Using complete graph for inference")
    else:
        print("Using candidate edges for inference")
    
    pfn_distances = []
    ortools_distances = []
    pfn_or_distances = []
    pfn_2opt_distances = []
    processing_times_pfn = []
    processing_times_ortools = []
    processing_times_pfn_or = []
    processing_times_pfn_2opt = []
    
    viz_idx = np.random.randint(0, len(test_instances))
    
    for i, (coords_seq, lkh_solution_seq, candidate_info_seq) in enumerate(zip(test_instances, lkh_solutions, candidate_infos)):
        print(f"Processing test sequence {i+1}/{len(test_instances)}...")
        
        # Use the last instance in the sequence for evaluation
        last_coords = coords_seq[-1]
        
        # Step 1: Compute OR-Tools solution and record time
        print(f"  Computing OR-Tools solution for instance with {len(last_coords)} nodes...")
        start_time = time.time()
        ortools_result = solve_tsp_ortools(last_coords)
        
        # Handle different return formats from solve_tsp_ortools
        if isinstance(ortools_result, tuple) and len(ortools_result) == 2:
            ortools_tour, ortools_solve_time = ortools_result
            if ortools_solve_time is None:
                ortools_solve_time = time.time() - start_time
        else:
            # Handle case where only tour is returned
            ortools_tour = ortools_result
            ortools_solve_time = time.time() - start_time
        
        ortools_distance = calculate_tour_length(last_coords, ortools_tour)
        ortools_distances.append(ortools_distance)
        processing_times_ortools.append(ortools_solve_time)
        
        # Step 2: Use model to predict solution and record time
        print(f"  Computing PFN solution...")
        start_time = time.time()
        pfn_tour, pfn_distance = predict_tsp_with_pfn(
            model, coords_seq, lkh_solution_seq, 
            candidate_info=candidate_info_seq, 
            use_complete_graph=use_complete_graph,
            device=device, 
            decoding_strategy=decoding_strategy
        )
        pfn_time = time.time() - start_time
        pfn_distances.append(pfn_distance)
        processing_times_pfn.append(pfn_time)
        
        # Step 3: Use model to predict solution, optimize with OR-tools, and record time
        print(f"  Computing PFN solution (again)...")
        start_time = time.time()
        pfn_or_tour_initial, _ = predict_tsp_with_pfn(
            model, coords_seq, lkh_solution_seq, 
            candidate_info=candidate_info_seq, 
            use_complete_graph=use_complete_graph,
            device=device, 
            decoding_strategy=decoding_strategy
        )
        pfn_or_tour = solve_tsp_static_with_or_tools_and_initial_solutions(pfn_or_tour_initial, last_coords, time_limit = 1)
        pfn_or_time = time.time() - start_time
        pfn_or_distance = calculate_tour_length(last_coords, pfn_or_tour)
        pfn_or_distances.append(pfn_or_distance)
        processing_times_pfn_or.append(pfn_or_time)

        # Step 4: Use model to predict solution, optimize with 2-opt, and record time
        print(f"  Computing PFN solution (again)...")
        start_time = time.time()
        pfn_or_tour_initial, _ = predict_tsp_with_pfn(
            model, coords_seq, lkh_solution_seq, 
            candidate_info=candidate_info_seq, 
            use_complete_graph=use_complete_graph,
            device=device, 
            decoding_strategy=decoding_strategy
        )
        pfn_2opt_tour = solve_2_opt_with_initial_solutions(pfn_or_tour_initial, last_coords, time_limit = 1)
        pfn_2opt_time = time.time() - start_time
        pfn_2opt_distance = calculate_tour_length(last_coords, pfn_2opt_tour)
        pfn_2opt_distances.append(pfn_2opt_distance)
        processing_times_pfn_2opt.append(pfn_2opt_time)

        # Step 5: Report comparison results
        print(f"  Results: PFN distance={pfn_distance:.4f}, OR-Tools distance={ortools_distance:.4f}, PFN-OR distance = {pfn_or_distance:.4f}, PFN-2opt distance = {pfn_2opt_distance:.4f}")
        print(f"  Times: PFN={pfn_time:.4f}s, OR-Tools={ortools_solve_time:.4f}, PFN-OR={pfn_or_time:.4f}, PFN-2opt={pfn_2opt_time:.4f}")
        
        # Save visualization for one random instance
        if i == viz_idx and save_plot:
            fig, ((ax1,ax2), (ax3, ax4))= plt.subplots(2, 2, figsize=(28, 16))
            plot_tour(last_coords, pfn_tour, f"PFN Tour ({decoding_strategy}, distance: {pfn_distance:.4f})", ax=ax1)
            plot_tour(last_coords, ortools_tour, f"OR-Tools Tour (distance: {ortools_distance:.4f})", ax=ax2)
            plot_tour(last_coords, pfn_or_tour, f"PFN + OR-Tools Tour (distance: {pfn_or_distance:.4f})", ax=ax3)
            plot_tour(last_coords, pfn_2opt_tour, f"PFN + 2-opt Tour (distance: {pfn_2opt_distance:.4f})", ax=ax4)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"  Comparison plot saved to {plot_path}")
    
    # Compute final statistics
    pfn_distances = np.array(pfn_distances)
    ortools_distances = np.array(ortools_distances)
    pfn_or_distances = np.array(pfn_or_distances)
    pfn_2opt_distances = np.array(pfn_2opt_distances)
    relative_gap_purePFN = (pfn_distances - ortools_distances) / ortools_distances * 100
    relative_gap_PFNOR = (pfn_or_distances - ortools_distances) / ortools_distances * 100
    relative_gap_PFN2opt = (pfn_2opt_distances - ortools_distances) / ortools_distances * 100
    
    print("\n===== Evaluation Results =====")
    print(f"Average path length: PFN={np.mean(pfn_distances):.4f}, OR-Tools={np.mean(ortools_distances):.4f}, PFN-OR={np.mean(pfn_or_distances):.4f}")
    print(f"Average relative gap - Pure PFN: {np.mean(relative_gap_purePFN):.2f}%")
    print(f"Maximum relative gap - Pure PFN: {np.max(relative_gap_purePFN):.2f}%")
    print(f"Minimum relative gap - Pure PFN: {np.min(relative_gap_purePFN):.2f}%")
    print(f"Average relative gap - PFN + OR: {np.mean(relative_gap_PFNOR):.2f}%")
    print(f"Maximum relative gap - PFN + OR: {np.max(relative_gap_PFNOR):.2f}%")
    print(f"Minimum relative gap - PFN + OR: {np.min(relative_gap_PFNOR):.2f}%")
    print(f"Average relative gap - PFN + 2opt: {np.mean(relative_gap_PFN2opt):.2f}%")
    print(f"Maximum relative gap - PFN + 2opt: {np.max(relative_gap_PFN2opt):.2f}%")
    print(f"Minimum relative gap - PFN + 2opt: {np.min(relative_gap_PFN2opt):.2f}%")
    print(f"Pure PFN win rate: {np.mean(pfn_distances <= ortools_distances) * 100:.2f}%")
    print(f"Pure PFN average processing time: {np.mean(processing_times_pfn):.4f} seconds")
    print(f"PFN OR win rate: {np.mean(pfn_or_distances <= ortools_distances) * 100:.2f}%")
    print(f"PFN OR average processing time: {np.mean(processing_times_pfn_or):.4f} seconds")
    print(f"PFN 2-opt win rate: {np.mean(pfn_2opt_distances <= ortools_distances) * 100:.2f}%")
    print(f"PFN 2-opt average processing time: {np.mean(processing_times_pfn_2opt):.4f} seconds")
    print(f"OR-Tools average processing time: {np.mean(processing_times_ortools):.4f} seconds")
    print(f"Speed ratio (OR-Tools/PFN): {np.mean(processing_times_ortools)/np.mean(processing_times_pfn):.2f}x")
    
    return {
        'pfn_distances': pfn_distances,
        'ortools_distances': ortools_distances,
        'relative_gap_pure_pfn': relative_gap_purePFN,
        'relative_gap_pfn_or': relative_gap_PFNOR,
        'pfn_times': processing_times_pfn,
        'ortools_times': processing_times_ortools
    }

def load_tsp_model(model_path, emsize, nhid, nlayers, nhead, dropout, device='cuda', use_unified_encoding=False, use_shared_basis_film=False, merge_duplicate_coords=True, test_size=5):
    """Load a pretrained TSP model"""
    print(f"Loading pretrained model from {model_path}...")
    
    # For SharedBasisFiLM mode, we need to automatically detect num_instances from the checkpoint
    if use_shared_basis_film:
        # Load the checkpoint to inspect the inst_emb.weight size
        checkpoint = torch.load(model_path, map_location='cpu')
        inst_emb_key = 'encoder.net.emb_net.inst_emb.weight'
        
        if inst_emb_key in checkpoint:
            num_instances_from_checkpoint = checkpoint[inst_emb_key].shape[0]
            print(f"Detected num_instances={num_instances_from_checkpoint} from checkpoint")
            
            # Calculate reasonable batch_size and seq_len that multiply to num_instances
            # Common combinations: 32*10=320, 16*20=320, 64*5=320, etc.
            possible_configs = [
                (32, 10), (16, 20), (64, 5), (10, 32), (20, 16), (5, 64),
                (1, num_instances_from_checkpoint)  # Fallback
            ]
            
            # Choose the configuration closest to typical training settings
            training_batch_size, training_seq_len = possible_configs[0]  # Default to 32*10
            for bs, sl in possible_configs:
                if bs * sl == num_instances_from_checkpoint:
                    training_batch_size, training_seq_len = bs, sl
                    break
            
            print(f"Using batch_size={training_batch_size}, seq_len={training_seq_len} (num_instances={training_batch_size * training_seq_len})")
        else:
            # Fallback if inst_emb.weight not found
            print(f"Warning: Could not find {inst_emb_key} in checkpoint, using default values")
            training_batch_size = 32
            training_seq_len = 10
    else:
        # For other modes, use smaller values for efficiency
        training_batch_size = 1
        training_seq_len = test_size
    
    result = train_tsp(
        emsize=emsize,
        nhid=nhid,
        nlayers=nlayers,
        nhead=nhead,
        dropout=dropout,
        epochs=0, 
        steps_per_epoch=1,
        batch_size=training_batch_size,
        seq_len=training_seq_len,
        lr=1e-4,
        num_nodes_range=(4, 5),  # Temporary range for model initialization
        gpu_device=device,
        use_unified_encoding=use_unified_encoding,
        use_shared_basis_film=use_shared_basis_film,
        merge_duplicate_coords=merge_duplicate_coords
    )
    
    model = result.model
    
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    return model

def main():
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.train:
        print("=== Training Mode ===")
        print(f"Using max_candidates={args.max_candidates}")
        print(f"Using unified_encoding={args.use_unified_encoding}")
        model, model_path = train_tsp_model(args)
    else:
        print("=== Testing Mode ===")
        if args.model_path is None:
            raise ValueError("Model path must be provided in testing mode. Use --model_path argument.")
        
        model = load_tsp_model(
            model_path=args.model_path,
            emsize=args.emsize,
            nhid=args.nhid,
            nlayers=args.nlayers,
            nhead=args.nhead,
            dropout=args.dropout,
            device=args.cuda_device,
            use_unified_encoding=args.use_unified_encoding,
            use_shared_basis_film=args.use_shared_basis_film,
            test_size=args.test_size
        )
        model_path = args.model_path
    
    print(f"Preparing test instances...")
    if args.test_dataset_path:
        # Load test instances from pre-generated dataset
        test_instances, lkh_solutions, candidate_infos = load_test_instances_from_dataset(
            args.test_dataset_path,
            test_instances=args.test_instances,
            generation_strategy=args.generation_strategy,
            test_size=args.test_size,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes
        )
    else:
        # Generate test instances on the fly
        test_instances, lkh_solutions, ortools_solutions, ortools_times, candidate_infos = generate_test_instances_with_ortools(
            num_instances=args.test_size,
            
            num_nodes_range=(args.min_nodes, args.max_nodes),
            max_candidates=args.max_candidates,
            device=args.cuda_device,
            test_instances=args.test_instances
        )
    
    graph_type = "complete" if args.use_complete_graph else "candidate"
    model_name = os.path.splitext(os.path.basename(args.model_path))[0] if args.model_path else "trained_model"
    plot_path = os.path.join(args.save_dir, f"{model_name}_{args.generation_strategy}_{graph_type}.png")
    
    results = evaluate_and_compare(
        model=model,
        test_instances=test_instances, 
        lkh_solutions=lkh_solutions,
        candidate_infos=candidate_infos,
        use_complete_graph=args.use_complete_graph,
        device=args.cuda_device, 
        decoding_strategy=args.decoding_strategy,
        plot_path=plot_path
    )
    
    results_path = os.path.join(args.save_dir, f"{model_name}_{args.generation_strategy}_{graph_type}.npz")
    np.savez(results_path, 
             pfn_distances=results['pfn_distances'],
             ortools_distances=results['ortools_distances'],
             relative_gap_pure_pfn = results['relative_gap_pure_pfn'],
             relative_gap_pfn_or = results['relative_gap_pfn_or'],
             pfn_times=results['pfn_times'],
             ortools_times=results['ortools_times'])
    
    print(f"Results saved to {results_path}")
    
    print("\n===== Test Summary =====")
    print(f"Model: {model_path}")
    print(f"Decoding strategy: {args.decoding_strategy}")
    print(f"Max candidates: {args.max_candidates}")
    print(f"Results file: {results_path}")
    print(f"Plot file: {plot_path}")
    
if __name__ == "__main__":
    main() 