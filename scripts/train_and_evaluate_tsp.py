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
from pfns.priors.prior import Batch
from pfns.priors.tsp_decoding_strategies import greedy_decode, greedy_all_decode, beam_search_decode, beam_search_all_decode, mcmc_decode

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train and evaluate PFN model for TSP')
    parser.add_argument('--emsize', type=int, default=128, help='Embedding dimension size')
    parser.add_argument('--nhid', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--nlayers', type=int, default=4, help='Number of Transformer layers')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--min_nodes', type=int, default=10, help='Minimum number of nodes in TSP')
    parser.add_argument('--max_nodes', type=int, default=20, help='Maximum number of nodes in TSP')
    parser.add_argument('--test_size', type=int, default=20, help='Number of test instances')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save models')
    parser.add_argument('--cuda_device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--train', action='store_true', help='Whether to train the model (otherwise just test)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pretrained model for testing')
    parser.add_argument('--decoding_strategy', type=str, default='greedy', 
                        choices=['greedy', 'beam_search', 'mcmc', 'greedy_all', 'beam_search_all'], 
                        help='Decoding strategy for TSP')
    
    return parser.parse_args()

def train_tsp_model(args):
    """Train the TSP model"""
    print(f"Starting TSP model training...")
    print(f"Parameters: {args}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Train the model
    start_time = time.time()
    result = train_tsp(
        emsize=args.emsize,
        nhid=args.nhid,
        nlayers=args.nlayers,
        nhead=args.nhead,
        dropout=args.dropout,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        seq_len=args.max_nodes,
        lr=args.lr,
        num_nodes_range=(args.min_nodes, args.max_nodes),
        gpu_device=args.cuda_device,
        progress_bar=True,
        verbose=True
    )
    training_time = time.time() - start_time
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(args.save_dir, f"tsp_model_{timestamp}.pt")
    torch.save(result.model.state_dict(), model_save_path)
    
    print(f"Training completed in {training_time:.2f} seconds, model saved to {model_save_path}")
    
    return result.model.to('cuda'), model_save_path

def predict_tsp_with_pfn(model, coords, solution, device='cuda', decoding_strategy='greedy'):
    """Predict TSP tour using the trained PFN model"""
    model.eval()
    
    x = torch.tensor(coords, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1, 2)
    x = x.to(device)

    y = torch.tensor(solution, dtype=torch.float32).unsqueeze(1).to(device)
    
    seq_len = x.shape[0]
    
    # Step by step prediction
    with torch.no_grad():
        outputs = model((None, x, y), single_eval_pos=seq_len-1)
        edge_values, edge_info = outputs
        edge_index, node_offset_map = edge_info
        
        edge_values = F.sigmoid(edge_values[-1])
        
        node_map = {value: key for key, value in node_offset_map.items()}
        edge_index = edge_index.squeeze().cpu().numpy()
        edge_values = edge_values.squeeze().cpu().numpy()
        num_nodes = x.shape[-2]

        adj_list = [[] for _ in range(num_nodes)]
            
        for i in range(len(edge_index)):
            u, v = edge_index[i]
            u = node_map[u][-1]
            v = node_map[v][-1]
            prob = edge_values[i]
            adj_list[u].append((v, prob))
            adj_list[v].append((u, prob))

        #greedy search for the tour
        if decoding_strategy == 'greedy':
            tour = greedy_decode(adj_list, num_nodes)
        elif decoding_strategy == 'greedy_all':
            tour = greedy_all_decode(adj_list, num_nodes)
        elif decoding_strategy == 'beam_search':
            tour = beam_search_decode(adj_list, num_nodes)
        elif decoding_strategy == 'beam_search_all':
            tour = beam_search_all_decode(adj_list, num_nodes)
        elif decoding_strategy == 'mcmc':
            tour = mcmc_decode(adj_list, node_map, edge_index, edge_values, num_nodes)
        else:
            raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")
    
    coords_np = coords.copy()[-1]
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += np.linalg.norm(coords_np[tour[i]] - coords_np[tour[i+1]])
    total_distance += np.linalg.norm(coords_np[tour[-1]] - coords_np[tour[0]])
    
    return tour, total_distance

def generate_test_instances_with_ortools(num_instances, num_nodes_range, device='cpu'):
    """Generate test instances and their corresponding OR-Tools solutions using TSPDataLoader"""
    
    def dummy_sampler():
        """Placeholder function, returns fixed values"""
        return 0, num_nodes_range[1]
        
    # Create TSPDataLoader instance
    dataloader = TSPDataLoader(
        num_steps=1,  # Only need one batch
        batch_size=num_instances,
        eval_pos_seq_len_sampler=dummy_sampler,
        seq_len_maximum=num_nodes_range[1],
        device=device,
        num_nodes_range=num_nodes_range
    )
    
    # Get a batch of data
    batch = next(iter(dataloader))
    
    test_instances = []
    ortools_solutions = []
    
    # Extract coordinates and solutions
    for i in range(batch.x.shape[1]): 
        coords = batch.x[:, i, :].cpu().numpy()
        solution = batch.target_y[:, i].cpu().numpy()
        
        test_instances.append(coords)
        ortools_solutions.append(solution)
    
    return test_instances, ortools_solutions

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

def evaluate_and_compare(model, test_instances, ortools_solutions, device='cuda', decoding_strategy='greedy', save_plot=True, plot_path='tsp_comparison.png'):
    """Evaluate model and compare with OR-Tools"""
    print(f"Starting model evaluation with {decoding_strategy} decoding strategy...")
    
    pfn_distances = []
    ortools_distances = []
    processing_times_pfn = []
    
    viz_idx = np.random.randint(0, len(test_instances))
    
    for i, (coords, ortools_solution) in enumerate(zip(test_instances, ortools_solutions)):
        print(f"Processing test instance {i+1}/{len(test_instances)}...")
        
        ortools_tour = ortools_solution.tolist()
        ortools_distance = calculate_tour_length(coords[-1], ortools_tour[-1])
        ortools_distances.append(ortools_distance)
        
        start_time = time.time()
        pfn_tour, pfn_distance = predict_tsp_with_pfn(model, coords, ortools_tour, device=device, decoding_strategy=decoding_strategy)
        pfn_time = time.time() - start_time
        pfn_distances.append(pfn_distance)
        processing_times_pfn.append(pfn_time)
        
        print(f"Instance {i+1}: PFN distance={pfn_distance:.4f}, OR-Tools distance={ortools_distance:.4f}")
        print(f"PFN processing time: {pfn_time:.4f} seconds")
        
        if i == viz_idx and save_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            plot_tour(coords[-1], pfn_tour, f"PFN Tour ({decoding_strategy}, distance: {pfn_distance:.4f})", ax=ax1)
            plot_tour(coords[-1], ortools_tour[-1], f"OR-Tools Tour (distance: {ortools_distance:.4f})", ax=ax2)
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
    
    return {
        'pfn_distances': pfn_distances,
        'ortools_distances': ortools_distances,
        'relative_gap': relative_gap,
        'pfn_times': processing_times_pfn
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
        seq_len=10,
        lr=1e-4,
        num_nodes_range=(1, 2),
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
    test_instances, ortools_solutions = generate_test_instances_with_ortools(
        num_instances=args.test_size,
        num_nodes_range=(args.min_nodes, args.max_nodes),
        device=args.cuda_device
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(args.save_dir, f"tsp_comparison_{args.decoding_strategy}_{timestamp}.png")
    
    results = evaluate_and_compare(
        model=model,
        test_instances=test_instances, 
        ortools_solutions=ortools_solutions, 
        device=args.cuda_device, 
        decoding_strategy=args.decoding_strategy,
        plot_path=plot_path
    )
    
    results_path = os.path.join(args.save_dir, f"tsp_results_{args.decoding_strategy}_{timestamp}.npz")
    np.savez(results_path, 
             pfn_distances=results['pfn_distances'],
             ortools_distances=results['ortools_distances'],
             relative_gap=results['relative_gap'],
             pfn_times=results['pfn_times'])
    
    print(f"Results saved to {results_path}")
    
    print("\n===== Test Summary =====")
    print(f"Model: {model_path}")
    print(f"Decoding strategy: {args.decoding_strategy}")
    print(f"Results file: {results_path}")
    print(f"Plot file: {plot_path}")
    
if __name__ == "__main__":
    main() 