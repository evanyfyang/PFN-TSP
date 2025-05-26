#!/usr/bin/env python3
"""
TSP dataset generation script with multiprocessing support
Generates TSP instances for specified node ranges with configurable parameters
Supports merging existing datasets
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
import glob

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


def load_dataset(dataset_path):
    """Load dataset from file"""
    print(f"Loading dataset from: {dataset_path}")
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Successfully loaded dataset")
        return dataset
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None


def merge_datasets(dataset_paths, output_path, new_name):
    """Merge multiple datasets into one"""
    print(f"Merging {len(dataset_paths)} datasets...")
    
    merged_dataset = {}
    total_instances = 0
    
    for i, path in enumerate(dataset_paths):
        print(f"Processing dataset {i+1}/{len(dataset_paths)}: {os.path.basename(path)}")
        dataset = load_dataset(path)
        
        if dataset is None:
            print(f"Skipping {path} due to loading error")
            continue
            
        # Handle both individual node datasets and complete datasets
        if isinstance(dataset, dict) and all(isinstance(k, int) for k in dataset.keys()):
            # This is a complete dataset (node_count -> instances)
            for num_nodes, instances in dataset.items():
                if num_nodes not in merged_dataset:
                    merged_dataset[num_nodes] = []
                merged_dataset[num_nodes].extend(instances)
                total_instances += len(instances)
        elif isinstance(dataset, list):
            # This is a single node dataset
            # Try to infer node count from first instance
            if dataset and 'num_nodes' in dataset[0]:
                num_nodes = dataset[0]['num_nodes']
                if num_nodes not in merged_dataset:
                    merged_dataset[num_nodes] = []
                merged_dataset[num_nodes].extend(dataset)
                total_instances += len(dataset)
            else:
                print(f"Warning: Cannot determine node count for dataset {path}")
        else:
            print(f"Warning: Unknown dataset format for {path}")
    
    # Update instance IDs to avoid conflicts
    print("Updating instance IDs to avoid conflicts...")
    for num_nodes in merged_dataset:
        for i, instance in enumerate(merged_dataset[num_nodes]):
            instance['instance_id'] = i
    
    # Save merged dataset
    print(f"Saving merged dataset to: {output_path}")
    save_dataset(merged_dataset, output_path)
    
    print(f"\nMerge completed!")
    print(f"Total instances: {total_instances}")
    print(f"Node sizes: {sorted(merged_dataset.keys())}")
    
    # Print statistics
    print("\nMerged dataset statistics:")
    for num_nodes in sorted(merged_dataset.keys()):
        instances = merged_dataset[num_nodes]
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
    
    return merged_dataset


def main():
    parser = argparse.ArgumentParser(description='Generate or merge TSP datasets with multiprocessing')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['generate', 'merge'], default='generate',
                       help='Mode: generate new dataset or merge existing datasets')
    
    # Generation parameters
    parser.add_argument('--min_nodes', type=int, default=30, help='Minimum number of nodes')
    parser.add_argument('--max_nodes', type=int, default=80, help='Maximum number of nodes')
    parser.add_argument('--instances_per_size', type=int, default=2560, help='Number of instances per node size')
    parser.add_argument('--max_candidates', type=int, default=5, help='LKH3 maximum candidates')
    parser.add_argument('--alpha', type=float, default=None, help='LKH3 alpha parameter')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of parallel processes')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--dataset_name', type=str, default='tsp_dataset', help='Dataset name prefix')
    
    # Merge parameters
    parser.add_argument('--merge_datasets', type=str, nargs='+', default=None,
                       help='Paths to datasets to merge (supports wildcards)')
    parser.add_argument('--merged_name', type=str, default='merged_tsp_dataset',
                       help='Name for the merged dataset')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.output_dir = os.path.join(project_root, 'pfns', 'datasets', 'tsp')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'merge':
        if args.merge_datasets is None:
            print("Error: --merge_datasets is required for merge mode")
            return
        
        # Expand wildcards
        dataset_paths = []
        for pattern in args.merge_datasets:
            if '*' in pattern or '?' in pattern:
                expanded = glob.glob(pattern)
                dataset_paths.extend(expanded)
            else:
                dataset_paths.append(pattern)
        
        # Remove duplicates and filter existing files
        dataset_paths = list(set(dataset_paths))
        dataset_paths = [p for p in dataset_paths if os.path.exists(p)]
        
        if not dataset_paths:
            print("Error: No valid dataset files found")
            return
        
        print("TSP Dataset Merger")
        print("=" * 60)
        print(f"Datasets to merge: {len(dataset_paths)}")
        for path in dataset_paths:
            print(f"  - {path}")
        print(f"Output directory: {args.output_dir}")
        print(f"Merged dataset name: {args.merged_name}")
        print()
        
        output_path = os.path.join(args.output_dir, f'{args.merged_name}_complete.pkl')
        merge_datasets(dataset_paths, output_path, args.merged_name)
        
    else:  # generate mode
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