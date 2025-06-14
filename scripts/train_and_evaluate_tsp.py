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

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pfns.train_tsp import train_tsp
from pfns.priors.tsp_data_loader import TSPDataLoader
from pfns.priors.tsp_offline_data_loader import TSPOfflineDataLoader
from pfns.priors.prior import Batch
from pfns.priors.tsp_decoding_strategies import *

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
    parser.add_argument('--max_candidates', type=int, default=15, help='Maximum number of candidates per node for LKH3')
    parser.add_argument('--test_size', type=int, default=10, help='Number of seq len')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save models')
    parser.add_argument('--cuda_device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--train', action='store_true', help='Whether to train the model (otherwise just test)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pretrained model for testing')
    parser.add_argument('--decoding_strategy', type=str, default='greedy', 
                        choices=['greedy', 'beam_search', 'mcmc', 'greedy_all', 'beam_search_all', 'greedy_edge'], 
                        help='Decoding strategy for TSP')
    parser.add_argument('--test_instances', type=int, default=20, help='Number of test instances')
    parser.add_argument('--use_complete_graph', action='store_true', 
                        help='Use complete graph instead of candidate edges (for comparison)')
    
    # Add online/offline training mode arguments
    parser.add_argument('--training_mode', type=str, default='online', choices=['online', 'offline'],
                        help='Training mode: online (generate data during training) or offline (use pre-generated data)')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to pre-generated dataset (required for offline mode)')
    
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
        'progress_bar': True,
        'verbose': True,
        'max_candidates': args.max_candidates
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
            'shuffle': True
        }
        # Override the dataloader class to use offline loader
        train_kwargs['priordataloader_class'] = TSPOfflineDataLoader
    
    # Train the model
    start_time = time.time()
    result = train_tsp(**train_kwargs)
    training_time = time.time() - start_time
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = args.training_mode
    model_save_path = os.path.join(args.save_dir, f"tsp_model_{mode_suffix}_{timestamp}.pt")
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
    num_nodes = last_coords.shape[0]
    
    # Create complete x tensor (seq_len, 1, num_nodes, 2)
    x = torch.zeros(seq_len, 1, num_nodes, 2, dtype=torch.float32, device=device)
    y = torch.zeros(seq_len, 1, num_nodes, dtype=torch.long, device=device)
    
    for i, (coord, sol) in enumerate(zip(coords, solution)):
        x[i, 0] = torch.tensor(coord, dtype=torch.float32, device=device)
        y[i, 0] = torch.tensor(sol, dtype=torch.long, device=device)
    
    # Step by step prediction
    with torch.no_grad():
        # Pass candidate_info to the model only if not using complete graph
        if use_complete_graph:
            print("Using complete graph for inference...")
            outputs = model((None, x, y), single_eval_pos=seq_len-1, candidate_info=None)
        else:
            print("Using candidate edges for inference...")
            outputs = model((None, x, y), single_eval_pos=seq_len-1, candidate_info=candidate_info)
        edge_values_padded, edge_info = outputs
        
        # New edge_info format is [edge_index_list, node_offset_map, edge_counts]
        edge_index_list, node_offset_map, edge_counts = edge_info
        
        # Get results for the last evaluation position
        # edge_values_padded shape is [seq_eval_len, batch_size, max_edges]
        # We take the first element (0) because seq_eval_len=1, batch_size=1
        last_edge_values = edge_values_padded[0, 0]
        
        # Get edge_index for the last evaluation position
        last_edge_index = edge_index_list[-1]
        
        # Apply sigmoid activation to get probabilities
        edge_values = F.sigmoid(last_edge_values)
        
        # Build node mapping
        node_map = {value: key for key, value in node_offset_map.items()}
        edge_index_np = last_edge_index.cpu().numpy()
        edge_values_np = edge_values.cpu().numpy()
        
        # Build adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        
        # First pass: collect all edge probabilities
        edge_probs = {}  # (u, v) -> list of probabilities
        
        # Only consider valid edges (according to edge_counts for the last problem)
        valid_edge_count = edge_counts[-1]
        
        for i in range(valid_edge_count):
            if i >= len(edge_values_np):
                break
                
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
                    
                    # Store edge probability for both directions
                    prob = edge_values_np[i]
                    edge_key = (min(u_node, v_node), max(u_node, v_node))
                    
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
        elif decoding_strategy == 'beam_search':
            tour = beam_search_decode(adj_list, num_nodes)
        elif decoding_strategy == 'beam_search_all':
            tour = beam_search_all_decode(adj_list, num_nodes)
        elif decoding_strategy == 'mcmc':
            tour = mcmc_decode(adj_list, node_map, edge_index_np, edge_values_np, num_nodes)
        elif decoding_strategy == 'greedy_edge':
            tour = greedy_edge_decode(adj_list, num_nodes)
        else:
            raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")
    
    # Calculate path length
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += np.linalg.norm(last_coords[tour[i]] - last_coords[tour[i+1]])
    total_distance += np.linalg.norm(last_coords[tour[-1]] - last_coords[tour[0]])
    
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

def evaluate_and_compare(model, test_instances, lkh_solutions, ortools_solutions, ortools_times, candidate_infos, use_complete_graph=False, device='cuda', decoding_strategy='greedy', save_plot=True, plot_path='tsp_comparison.png'):
    """Evaluate model and compare with OR-Tools"""
    print(f"Starting model evaluation with {decoding_strategy} decoding strategy...")
    if use_complete_graph:
        print("Using complete graph for all instances")
    else:
        print("Using candidate edges for all instances")
    
    pfn_distances = []
    ortools_distances = []
    processing_times_pfn = []
    processing_times_ortools = ortools_times
    
    viz_idx = np.random.randint(0, len(test_instances))
    
    for i, (coords, ortools_solution, lkh_solution, candidate_info) in enumerate(zip(test_instances, ortools_solutions, lkh_solutions, candidate_infos)):
        print(f"Processing test instance {i+1}/{len(test_instances)}...")
        
        # Use the last TSP instance for comparison
        ortools_tour = ortools_solution[-1].tolist()
        ortools_distance = calculate_tour_length(coords[-1], ortools_tour)
        ortools_distances.append(ortools_distance)
        
        start_time = time.time()
        pfn_tour, pfn_distance = predict_tsp_with_pfn(
            model, coords, lkh_solution, 
            candidate_info=candidate_info, 
            use_complete_graph=use_complete_graph,
            device=device, 
            decoding_strategy=decoding_strategy
        )
        pfn_time = time.time() - start_time
        pfn_distances.append(pfn_distance)
        processing_times_pfn.append(pfn_time)
        
        print(f"Instance {i+1}: PFN distance={pfn_distance:.4f}, OR-Tools distance={ortools_distance:.4f}")
        print(f"PFN processing time: {pfn_time:.4f} seconds, OR-Tools processing time: {ortools_times[i]:.4f} seconds")
        
        if i == viz_idx and save_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            plot_tour(coords[-1], pfn_tour, f"PFN Tour ({decoding_strategy}, distance: {pfn_distance:.4f})", ax=ax1)
            plot_tour(coords[-1], ortools_tour, f"OR-Tools Tour (distance: {ortools_distance:.4f})", ax=ax2)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Comparison plot saved to {plot_path}")
    
    pfn_distances = np.array(pfn_distances)
    ortools_distances = np.array(ortools_distances)
    relative_gap = (pfn_distances - ortools_distances) / ortools_distances * 100
    
    print("\n===== Evaluation Results =====")
    print(f"Average path length: PFN={np.mean(pfn_distances):.4f}, OR-Tools={np.mean(ortools_distances):.4f}")
    print(f"Average relative gap: {np.mean(relative_gap):.2f}%")
    print(f"Maximum relative gap: {np.max(relative_gap):.2f}%")
    print(f"Minimum relative gap: {np.min(relative_gap):.2f}%")
    print(f"PFN win rate: {np.mean(pfn_distances <= ortools_distances) * 100:.2f}%")
    print(f"PFN average processing time: {np.mean(processing_times_pfn):.4f} seconds")
    print(f"OR-Tools average processing time: {np.mean(processing_times_ortools):.4f} seconds")
    print(f"Speed ratio (OR-Tools/PFN): {np.mean(processing_times_ortools)/np.mean(processing_times_pfn):.2f}x")
    
    return {
        'pfn_distances': pfn_distances,
        'ortools_distances': ortools_distances,
        'relative_gap': relative_gap,
        'pfn_times': processing_times_pfn,
        'ortools_times': processing_times_ortools
    }

def load_tsp_model(model_path, emsize, nhid, nlayers, nhead, dropout, device='cuda'):
    """Load a pretrained TSP model"""
    print(f"Loading pretrained model from {model_path}...")

    result = train_tsp(
        emsize=emsize,
        nhid=nhid,
        nlayers=nlayers,
        nhead=nhead,
        dropout=dropout,
        epochs=0, 
        steps_per_epoch=1,
        batch_size=1,
        seq_len=5,
        lr=1e-4,
        num_nodes_range=(4, 5),  
        gpu_device=device
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
            device=args.cuda_device
        )
        model_path = args.model_path
    
    print(f"Generating test instances...")
    test_instances, lkh_solutions, ortools_solutions, ortools_times, candidate_infos = generate_test_instances_with_ortools(
        num_instances=args.test_size,
        num_nodes_range=(args.min_nodes, args.max_nodes),
        max_candidates=args.max_candidates,
        device=args.cuda_device,
        test_instances=args.test_instances
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_type = "complete" if args.use_complete_graph else "candidate"
    plot_path = os.path.join(args.save_dir, f"tsp_comparison_{args.decoding_strategy}_{graph_type}_{timestamp}.png")
    
    results = evaluate_and_compare(
        model=model,
        test_instances=test_instances, 
        lkh_solutions=lkh_solutions,
        ortools_solutions=ortools_solutions,
        ortools_times=ortools_times,
        candidate_infos=candidate_infos,
        use_complete_graph=args.use_complete_graph,
        device=args.cuda_device, 
        decoding_strategy=args.decoding_strategy,
        plot_path=plot_path
    )
    
    results_path = os.path.join(args.save_dir, f"tsp_results_{args.decoding_strategy}_{graph_type}_{timestamp}.npz")
    np.savez(results_path, 
             pfn_distances=results['pfn_distances'],
             ortools_distances=results['ortools_distances'],
             relative_gap=results['relative_gap'],
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