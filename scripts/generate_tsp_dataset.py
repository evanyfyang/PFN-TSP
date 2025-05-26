#!/usr/bin/env python3
"""
TSP dataset generation script with multiprocessing support
Generates TSP instances for specified node ranges with configurable parameters
"""

import os
import sys
import numpy as np
import torch
import pickle
import argparse
from tqdm import tqdm
import time
import multiprocessing as mp

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pfns.priors.tsp_data_loader import solve_tsp_lkh3
from pfns.priors.lkh3_wrapper import LKH3Wrapper


def generate_single_instance(args):
    """Worker function to generate a single TSP instance"""
    num_nodes, instance_id, max_candidates, alpha = args
    
    np.random.seed(instance_id * 1000 + num_nodes)
    coords = np.random.uniform(0, 1, size=(num_nodes, 2))
    
    try:
        tour, candidate_info = solve_tsp_lkh3(coords, max_candidates=max_candidates, alpha=alpha)
        return {
            'coords': coords,
            'tour': tour,
            'candidate_info': candidate_info,
            'num_nodes': num_nodes,
            'instance_id': instance_id
        }
    except Exception as e:
        print(f"Failed to generate TSP-{num_nodes}-{instance_id}: {e}")
        return None


def generate_tsp_instances(num_nodes, num_instances, max_candidates=15, alpha=None, num_processes=8):
    """Generate TSP instances for specified number of nodes"""
    print(f"Generating {num_instances} TSP instances with {num_nodes} nodes...")
    
    args_list = [(num_nodes, i, max_candidates, alpha) for i in range(num_instances)]
    
    start_time = time.time()
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(generate_single_instance, args_list),
            total=num_instances,
            desc=f"TSP-{num_nodes}"
        ))
    
    valid_results = [r for r in results if r is not None]
    
    generation_time = time.time() - start_time
    print(f"Completed TSP-{num_nodes}: {len(valid_results)}/{num_instances} instances, time: {generation_time:.2f}s")
    
    return valid_results


def save_dataset(dataset, save_path):
    """Save dataset to file"""
    print(f"Saving dataset to: {save_path}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
    print(f"Dataset saved, file size: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Generate TSP dataset with multiprocessing')
    parser.add_argument('--min_nodes', type=int, default=30, help='Minimum number of nodes')
    parser.add_argument('--max_nodes', type=int, default=80, help='Maximum number of nodes')
    parser.add_argument('--instances_per_size', type=int, default=256, help='Number of instances per node size')
    parser.add_argument('--max_candidates', type=int, default=15, help='LKH3 maximum candidates')
    parser.add_argument('--alpha', type=float, default=None, help='LKH3 alpha parameter')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of parallel processes')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--dataset_name', type=str, default='tsp_dataset', help='Dataset name prefix')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.output_dir = os.path.join(project_root, 'pfns', 'datasets', 'tsp')
    
    if args.num_processes is None:
        args.num_processes = min(16, mp.cpu_count())
    
    print("TSP Dataset Generator (Multiprocessing)")
    print("=" * 60)
    print(f"Node range: {args.min_nodes}-{args.max_nodes}")
    print(f"Instances per size: {args.instances_per_size}")
    print(f"Max candidates: {args.max_candidates}")
    print(f"Alpha parameter: {args.alpha}")
    print(f"Parallel processes: {args.num_processes}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset name: {args.dataset_name}")
    print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_datasets = {}
    total_start_time = time.time()
    
    for num_nodes in range(args.min_nodes, args.max_nodes + 1):
        instances = generate_tsp_instances(
            num_nodes=num_nodes,
            num_instances=args.instances_per_size,
            max_candidates=args.max_candidates,
            alpha=args.alpha,
            num_processes=args.num_processes
        )
        
        dataset_path = os.path.join(args.output_dir, f'{args.dataset_name}_{num_nodes}nodes.pkl')
        save_dataset(instances, dataset_path)
        
        all_datasets[num_nodes] = instances
    
    complete_dataset_path = os.path.join(args.output_dir, f'{args.dataset_name}_complete.pkl')
    save_dataset(all_datasets, complete_dataset_path)
    
    total_time = time.time() - total_start_time
    total_instances = sum(len(instances) for instances in all_datasets.values())
    
    print("\n" + "=" * 60)
    print("Dataset generation completed!")
    print(f"Total instances: {total_instances}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per instance: {total_time/total_instances:.3f}s")
    print(f"Complete dataset path: {complete_dataset_path}")
    
    print("\nDataset statistics:")
    for num_nodes, instances in all_datasets.items():
        avg_edges = 0
        if instances and instances[0]['candidate_info']:
            total_edges = 0
            valid_count = 0
            for inst in instances:
                if inst['candidate_info'] and 'candidates' in inst['candidate_info']:
                    inst_edges = sum(len(candidates) for candidates in inst['candidate_info']['candidates'].values())
                    total_edges += inst_edges
                    valid_count += 1
            if valid_count > 0:
                avg_edges = total_edges / valid_count
        
        print(f"  TSP-{num_nodes}: {len(instances)} instances, avg edges: {avg_edges:.1f}")


if __name__ == "__main__":
    main() 