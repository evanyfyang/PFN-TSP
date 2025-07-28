#!/usr/bin/env python3
"""
TSP dataset generation script with multiprocessing support
Generates TSP instances for specified node ranges with configurable parameters
Supports merging existing datasets

STRATEGIES:
- random_nodes_same_size: Generate random instances with same node count
- fix_all_nodes_same_size: Use fixed node set, sample different subsets
- fix_group_nodes_same_size: Each group has its own fixed node set
- sample_from_large: IMPROVED - Generate base instance of target size, sample smaller subgraphs
  * Each group has one base instance (num_nodes size) 
  * Multiple sampled instances (sampling_ratio * num_nodes size) from the base
  * All instances (base + sampled) are included in each group
  * Fully parallel processing across all groups and instances for maximum efficiency

MULTIPROCESSING IMPROVEMENTS:
- sample_from_large strategy now processes ALL instances across ALL groups in parallel
- No more sequential group processing - maximum utilization of available cores
- Significantly faster generation for large datasets
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
from datetime import datetime
from multiprocessing import Pool
import hashlib

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pfns.priors.tsp_data_loader import solve_tsp_lkh3, solve_tsp_ortools
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


def generate_tsp_coordinates(num_instances, num_nodes):
    """
    Generate random coordinates for TSP instances.
    
    Args:
        num_instances: Number of instances to generate
        num_nodes: Number of nodes in each instance
        
    Returns:
        List of numpy arrays containing coordinates for each instance
    """
    coordinates = []
    for _ in range(num_instances):
        # Generate random coordinates in [0, 1] x [0, 1] square
        coords = np.random.rand(num_nodes, 2)
        coordinates.append(coords)
    return coordinates


def generate_tsp_instances(num_instances, num_nodes, max_candidates=5, alpha=None, num_processes=16):
    """
    Generate TSP instances with specified number of nodes.
    
    Args:
        num_instances: Number of instances to generate
        num_nodes: Number of nodes in each instance
        max_candidates: Maximum number of candidates per node for LKH3
        alpha: Alpha parameter for LKH3
        num_processes: Number of processes to use for parallel generation
        
    Returns:
        List of dictionaries containing instance information
    """
    # Prepare arguments for parallel processing
    args_list = []
    for i in range(num_instances):
        np.random.seed(i * 1000 + num_nodes)
        coords = np.random.uniform(0, 1, size=(num_nodes, 2))
        args_list.append((coords, max_candidates, alpha))
    
    # Generate instances in parallel
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.starmap(generate_single_instance, args_list),
            total=len(args_list),
            desc=f"Generating TSP-{num_nodes} instances",
            ncols=100
        ))
    
    # Process results
    instances = []
    for i, result in enumerate(results):
        if result is not None:
            instances.append(result)
    
    return instances


def generate_single_instance(coords, max_candidates=5, alpha=None):
    """
    Generate a single TSP instance with LKH3 solution and candidate information.
    
    Args:
        coords: Coordinates of the TSP instance
        max_candidates: Maximum number of candidates per node for LKH3
        alpha: Alpha parameter for LKH3
        
    Returns:
        Tuple of (coords, tour, candidate_info)
    """
    # Solve TSP using LKH3
    tour, candidate_info = solve_tsp_lkh3(coords, max_candidates=max_candidates, alpha=alpha)
    
    return coords, tour, candidate_info


def generate_tsp_instances_with_fixed_nodes(min_nodes, max_nodes, num_instances, max_candidates=5, alpha=None, num_processes=16, fixed_nodes=200):
    """
    Generate TSP instances using a fixed set of nodes.
    
    Args:
        min_nodes: Minimum number of nodes in TSP instances
        max_nodes: Maximum number of nodes in TSP instances
        num_instances: Number of instances to generate for each node count
        max_candidates: Maximum number of candidates per node for LKH3
        alpha: Alpha value for LKH3 candidate generation
        num_processes: Number of processes for parallel generation
        fixed_nodes: Number of fixed nodes to generate
        
    Returns:
        (all_instances, fixed_coords):
            all_instances: Dictionary mapping node counts to lists of TSP instances
            fixed_coords: The fixed set of node coordinates used
    """
    # First generate fixed nodes
    print(f"Generating {fixed_nodes} fixed nodes...")
    fixed_coords = np.random.uniform(0, 1, size=(fixed_nodes, 2))
    
    # Generate instances for each node count
    all_instances = {}
    for num_nodes in range(min_nodes, max_nodes + 1):
        print(f"\nGenerating {num_instances} instances for TSP-{num_nodes}...")
        instances = generate_single_instance_with_fixed_nodes(
            num_nodes, num_instances, fixed_coords,
            max_candidates, alpha, num_processes
        )
        all_instances[num_nodes] = instances
        print(f"Generated {len(instances)} instances for TSP-{num_nodes}")
    
    return all_instances, fixed_coords


def generate_single_instance_with_fixed_nodes(num_nodes, num_instances, fixed_coords, max_candidates=5, alpha=None, num_processes=16):
    """
    Generate TSP instances using fixed nodes.
    
    Args:
        num_nodes: Number of nodes in each instance
        num_instances: Number of instances to generate
        fixed_coords: Fixed set of node coordinates
        max_candidates: Maximum number of candidates per node for LKH3
        alpha: Alpha parameter for LKH3
        num_processes: Number of processes for parallel generation
        
    Returns:
        List of dictionaries containing instance information
    """
    instances = []
    
    # Process instances in parallel
    with Pool(num_processes) as pool:
        # Prepare arguments for parallel processing
        args = [(num_nodes, i, max_candidates, alpha, fixed_coords) for i in range(num_instances)]
        
        # Generate instances in parallel with progress bar
        results = list(tqdm(
            pool.imap(generate_single_instance_with_fixed_nodes_worker, args),
            total=len(args),
            desc=f"Generating TSP-{num_nodes} instances",
            ncols=100
        ))
        
        # Collect results
        for result in results:
            if result is not None:
                instances.append(result)
    
    return instances


def generate_single_instance_with_fixed_nodes_worker(args):
    """
    Worker function to generate a single TSP instance using fixed nodes.
    
    Args:
        args: Tuple containing (num_nodes, instance_id, max_candidates, alpha, fixed_coords)
        
    Returns:
        Dictionary containing instance data
    """
    num_nodes, instance_id, max_candidates, alpha, fixed_coords = args
    
    np.random.seed(instance_id * 1000 + num_nodes)
    # Randomly select nodes from fixed set
    selected_indices = np.random.choice(fixed_coords.shape[0], num_nodes, replace=False)
    coords = fixed_coords[selected_indices]
    
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


def generate_test_instances_with_fixed_nodes(min_nodes, max_nodes, num_instances, max_candidates=5, alpha=None, num_processes=16, test_instances_multiplier=1, fixed_coords=None, train_instances=None):
    """
    Generate test TSP instances using a fixed set of nodes, ensuring no overlap with training data.
    
    Args:
        min_nodes: Minimum number of nodes in TSP instances
        max_nodes: Maximum number of nodes in TSP instances
        num_instances: Number of instances to generate for each node count
        max_candidates: Maximum number of candidates per node for LKH3
        alpha: Alpha value for LKH3 candidate generation
        num_processes: Number of processes for parallel generation
        test_instances_multiplier: Multiplier for number of test instances to generate
        fixed_coords: Fixed set of node coordinates to use
        train_instances: Dictionary of training instances to check for overlap
        
    Returns:
        Dictionary mapping node counts to lists of TSP instances
    """
    if fixed_coords is None:
        fixed_nodes = 200
        print(f"Generating {fixed_nodes} fixed nodes for test set...")
        fixed_coords = np.random.uniform(0, 1, size=(fixed_nodes, 2))
    else:
        print(f"Using provided fixed_coords for test set...")
    
    # Generate instances for each node count
    all_instances = {}
    for num_nodes in range(min_nodes, max_nodes + 1):
        print(f"\nGenerating {num_instances} test instances for TSP-{num_nodes}...")
        extra_instances = int(num_instances * test_instances_multiplier * 1.2)
        instances = generate_single_instance_with_fixed_nodes(
            num_nodes, extra_instances, fixed_coords,
            max_candidates, alpha, num_processes
        )
        # Filter out instances that overlap with training data
        if train_instances and num_nodes in train_instances:
            train_coords = {tuple(inst['coords'].flatten()) for inst in train_instances[num_nodes]}
            filtered_instances = []
            for inst in instances:
                inst_coords = tuple(inst['coords'].flatten())
                if inst_coords not in train_coords:
                    filtered_instances.append(inst)
                    if len(filtered_instances) >= num_instances * test_instances_multiplier:
                        break
            instances = filtered_instances[:num_instances * test_instances_multiplier]
        else:
            instances = instances[:num_instances * test_instances_multiplier]
        all_instances[num_nodes] = instances
        print(f"Generated {len(instances)} test instances for TSP-{num_nodes}")
    return all_instances


def generate_group_fixed_nodes_worker(args):
    """
    Worker function for parallel generation of group-based fixed nodes TSP instances.
    
    Args:
        args: Tuple containing (num_nodes, group_idx, instances_per_group, max_candidates, alpha, min_fixed_nodes)
    
    Returns:
        Tuple of (num_nodes, group_idx, instances) where instances is the list of generated TSP instances
    """
    num_nodes, group_idx, instances_per_group, max_candidates, alpha, min_fixed_nodes = args
    
    # 🔧 修复：使用安全的种子生成策略，避免种子值过大
    # 使用哈希函数确保种子在合理范围内且分布均匀
    
    # 为group生成固定节点的种子
    group_seed_str = f"group_{group_idx}_nodes_{num_nodes}_fixed"
    group_seed_hash = hashlib.md5(group_seed_str.encode()).hexdigest()
    group_seed = int(group_seed_hash[:8], 16) % (2**31 - 1)  # 确保在int32范围内
    np.random.seed(group_seed)
    
    # Generate fixed nodes for this group
    fixed_nodes = max(min_fixed_nodes, num_nodes + 10)  # Ensure fixed_nodes >= num_nodes
    fixed_coords = np.random.uniform(0, 1, size=(fixed_nodes, 2))
    
    # Generate instances serially within this worker (no nested multiprocessing)
    instances = []
    for i in range(instances_per_group):
        # 🔧 修复：为每个instance生成独特的种子
        inst_seed_str = f"group_{group_idx}_nodes_{num_nodes}_inst_{i}"
        inst_seed_hash = hashlib.md5(inst_seed_str.encode()).hexdigest()
        inst_seed = int(inst_seed_hash[:8], 16) % (2**31 - 1)  # 确保在int32范围内
        np.random.seed(inst_seed)
        
        # Randomly select subset of fixed nodes
        if fixed_nodes > num_nodes:
            selected_indices = np.random.choice(fixed_nodes, num_nodes, replace=False)
            coords = fixed_coords[selected_indices]
        else:
            coords = fixed_coords[:num_nodes]
        
        try:
            # Solve TSP for this instance
            tour, candidate_info = solve_tsp_lkh3(coords, max_candidates=max_candidates, alpha=alpha)
            
            # Create instance dictionary
            instance = {
                'coords': coords,
                'tour': tour,
                'candidate_info': candidate_info,
                'num_nodes': num_nodes,
                'instance_id': group_idx * instances_per_group + i,
                'group_id': group_idx
            }
            instances.append(instance)
            
        except Exception as e:
            print(f"Failed to generate instance {i} in group {group_idx} for TSP-{num_nodes}: {e}")
            # Create fallback instance
            fallback_tour = list(range(num_nodes))
            fallback_candidate_info = {
                'dimension': num_nodes,
                'candidates': {},
                'mst_parents': {}
            }
            instance = {
                'coords': coords,
                'tour': fallback_tour,
                'candidate_info': fallback_candidate_info,
                'num_nodes': num_nodes,
                'instance_id': group_idx * instances_per_group + i,
                'group_id': group_idx
            }
            instances.append(instance)
    
    return num_nodes, group_idx, instances


def generate_tsp_instances_with_group_fixed_nodes(min_nodes, max_nodes, num_instances, max_candidates=5, alpha=None, num_processes=16, min_fixed_nodes=None, instances_per_group=100):
    """
    Generate TSP instances where each node size has multiple groups, each with its own fixed set of nodes.
    Uses parallel processing to speed up generation while preserving all original logic.
    
    Args:
        min_nodes: Minimum number of nodes in TSP instances
        max_nodes: Maximum number of nodes in TSP instances
        num_instances: Total number of instances to generate for each node count
        max_candidates: Maximum number of candidates per node for LKH3
        alpha: Alpha value for LKH3 candidate generation
        num_processes: Number of processes for parallel generation
        min_fixed_nodes: Minimum number of fixed nodes per group (default: max(instances_per_group + 10, max_nodes))
        instances_per_group: Number of instances per group
        
    Returns:
        Dictionary mapping node counts to lists of TSP instances
    """
    if num_instances % instances_per_group != 0:
        raise ValueError(f"num_instances ({num_instances}) must be divisible by instances_per_group ({instances_per_group})")
    
    num_groups = num_instances // instances_per_group
    
    if min_fixed_nodes is None:
        min_fixed_nodes = max(instances_per_group + 10, max_nodes)
    
    print(f"Generating TSP instances with group fixed nodes using {num_processes} processes...")
    print(f"Node range: {min_nodes}-{max_nodes}, Groups per node size: {num_groups}, Instances per group: {instances_per_group}")
    
    # Prepare all work items for parallel processing
    work_items = []
    for num_nodes in range(min_nodes, max_nodes + 1):
        for group_idx in range(num_groups):
            work_items.append((num_nodes, group_idx, instances_per_group, max_candidates, alpha, min_fixed_nodes))
    
    print(f"Total work items: {len(work_items)} (processing groups in parallel)")
    
    # Process all work items in parallel with progress bar
    all_instances = {}
    
    # Initialize result structure
    for num_nodes in range(min_nodes, max_nodes + 1):
        all_instances[num_nodes] = [[] for _ in range(num_groups)]
    
    # Process all work items in parallel with real-time progress updates
    results = []
    completed_count = 0
    
    def update_progress(result):
        """Callback function to update progress in real-time"""
        nonlocal completed_count
        results.append(result)
        completed_count += 1
        pbar.update(1)
    
    def error_callback(error):
        """Callback function to handle errors"""
        print(f"Error in worker: {error}")
        pbar.update(1)
    
    with mp.Pool(processes=num_processes) as pool:
        # Initialize progress bar
        pbar = tqdm(total=len(work_items), desc="Generating groups", ncols=100)
        
        # Submit all work items with callback
        async_results = []
        for work_item in work_items:
            async_result = pool.apply_async(
                generate_group_fixed_nodes_worker,
                args=(work_item,),
                callback=update_progress,
                error_callback=error_callback
            )
            async_results.append(async_result)
        
        # Wait for all tasks to complete
        for async_result in async_results:
            async_result.wait()
        
        pbar.close()
    
    # Organize results back into the expected structure
    for num_nodes, group_idx, instances in results:
        all_instances[num_nodes][group_idx] = instances
        
    # Flatten group structure to maintain original output format
    final_instances = {}
    for num_nodes in range(min_nodes, max_nodes + 1):
        group_instances = []
        for group_idx in range(num_groups):
            instances = all_instances[num_nodes][group_idx]
            group_instances.extend(instances)
            print(f"  Generated {len(instances)} instances for TSP-{num_nodes} group {group_idx + 1}")
        
        final_instances[num_nodes] = group_instances
        print(f"Total: Generated {len(group_instances)} instances for TSP-{num_nodes}")
    
    return final_instances


def generate_test_instances_with_group_fixed_nodes(min_nodes, max_nodes, test_instances, max_candidates=5, alpha=None, num_processes=16, min_fixed_nodes=None, train_instances=None, test_instances_multiplier=5, instances_per_group=100):
    """
    Generate test TSP instances where each node size has multiple groups, each with its own fixed set of nodes.
    
    Args:
        min_nodes: Minimum number of nodes in TSP instances
        max_nodes: Maximum number of nodes in TSP instances
        test_instances: Total number of test instances to generate for each node count
        max_candidates: Maximum number of candidates per node for LKH3
        alpha: Alpha value for LKH3 candidate generation
        num_processes: Number of processes for parallel generation
        min_fixed_nodes: Minimum number of fixed nodes per group (default: max(instances_per_group + 10, max_nodes))
        train_instances: Dictionary of training instances to check for overlap
        test_instances_multiplier: Multiplier for number of test instances to generate (default: 5)
        instances_per_group: Number of instances per group
        
    Returns:
        Dictionary mapping node counts to lists of TSP instances
    """
    if test_instances % instances_per_group != 0:
        raise ValueError(f"test_instances ({test_instances}) must be divisible by instances_per_group ({instances_per_group})")
    
    num_groups = test_instances // instances_per_group
    
    if min_fixed_nodes is None:
        min_fixed_nodes = max(instances_per_group + 10, max_nodes)
    
    # Generate instances for each node count
    all_instances = {}
    for num_nodes in range(min_nodes, max_nodes + 1):
        print(f"\nGenerating test instances for TSP-{num_nodes} ({num_groups} groups)...")
        
        # Generate instances for each group
        group_instances = []
        for group_idx in range(num_groups):
            print(f"  Group {group_idx + 1}/{num_groups}:")
            
            # Generate fixed nodes for this group
            fixed_nodes = max(int(num_nodes * 1.4), num_nodes + 10)  # Ensure fixed_nodes >= num_nodes
            print(f"    Using {fixed_nodes} fixed nodes for group {group_idx + 1}")
            fixed_coords = np.random.uniform(0, 1, size=(fixed_nodes, 2))
            
            # Generate more instances than needed to account for potential overlaps
            extra_instances = int(instances_per_group * test_instances_multiplier * 1.2)  # Generate 20% more instances
            instances = generate_single_instance_with_fixed_nodes(
                num_nodes, extra_instances, fixed_coords,
                max_candidates, alpha, num_processes
            )
            
            # Filter out instances that overlap with training data
            if train_instances and num_nodes in train_instances:
                train_coords = {tuple(inst['coords'].flatten()) for inst in train_instances[num_nodes]}
                filtered_instances = []
                for inst in instances:
                    inst_coords = tuple(inst['coords'].flatten())
                    if inst_coords not in train_coords:
                        filtered_instances.append(inst)
                        if len(filtered_instances) >= instances_per_group * test_instances_multiplier:
                            break
                instances = filtered_instances[:instances_per_group * test_instances_multiplier]
            
            group_instances.extend(instances)
            print(f"    Generated {len(instances)} test instances for group {group_idx + 1}")
        
        all_instances[num_nodes] = group_instances
        print(f"Total: Generated {len(group_instances)} test instances for TSP-{num_nodes}")
    
    return all_instances


def generate_covering_samples(num_nodes, sample_size, num_samples):
    """
    Generate multiple samples from a base instance, ensuring all base nodes are covered.
    Uses a greedy coverage strategy to guarantee that every node appears in at least one sample.
    
    Args:
        num_nodes: Total number of nodes in the base instance
        sample_size: Number of nodes in each sample
        num_samples: Number of samples to generate
        
    Returns:
        List of numpy arrays, each containing indices for one sample
    """
    if sample_size * num_samples < num_nodes:
        print(f"Warning: Total sample capacity ({sample_size} * {num_samples} = {sample_size * num_samples}) "
              f"is less than base nodes ({num_nodes}). Some nodes may not be covered.")
    
    # Track which nodes have been covered
    uncovered_nodes = set(range(num_nodes))
    all_samples = []
    
    for sample_idx in range(num_samples):
        sample_indices = []
        
        # Determine how many uncovered nodes to include in this sample
        remaining_samples = num_samples - sample_idx
        remaining_uncovered = len(uncovered_nodes)
        
        if remaining_uncovered > 0:
            # Calculate how many uncovered nodes to include in this sample
            # Distribute uncovered nodes roughly evenly across remaining samples
            uncovered_to_include = min(
                sample_size,
                max(1, remaining_uncovered // remaining_samples),
                remaining_uncovered
            )
            
            # If this is the last sample, include all remaining uncovered nodes
            if sample_idx == num_samples - 1:
                uncovered_to_include = min(sample_size, remaining_uncovered)
            
            # Select uncovered nodes for this sample
            if uncovered_to_include > 0:
                selected_uncovered = np.random.choice(
                    list(uncovered_nodes), 
                    size=uncovered_to_include, 
                    replace=False
                )
                sample_indices.extend(selected_uncovered)
                # Remove selected nodes from uncovered set
                uncovered_nodes -= set(selected_uncovered)
        
        # Fill remaining positions with random nodes (can be already covered)
        remaining_positions = sample_size - len(sample_indices)
        if remaining_positions > 0:
            # Choose from all nodes except those already in this sample
            available_nodes = [i for i in range(num_nodes) if i not in sample_indices]
            if len(available_nodes) >= remaining_positions:
                additional_nodes = np.random.choice(
                    available_nodes, 
                    size=remaining_positions, 
                    replace=False
                )
                sample_indices.extend(additional_nodes)
            else:
                # If not enough available nodes, fill with what we have
                sample_indices.extend(available_nodes)
        
        # Convert to numpy array and sort for consistency
        sample_indices = np.array(sample_indices)
        sample_indices = np.sort(sample_indices)
        all_samples.append(sample_indices)
    
    # Verify coverage
    covered_nodes = set()
    for sample in all_samples:
        covered_nodes.update(sample)
    
    coverage_rate = len(covered_nodes) / num_nodes
    print(f"    Coverage: {len(covered_nodes)}/{num_nodes} nodes ({coverage_rate:.1%})")
    
    if len(covered_nodes) < num_nodes:
        missing_nodes = set(range(num_nodes)) - covered_nodes
        print(f"    Warning: {len(missing_nodes)} nodes not covered: {sorted(missing_nodes)}")
    
    return all_samples


def generate_subgraph_from_large_instance(large_coords, num_nodes):
    """
    Generate a subgraph from a larger instance by sampling nodes.
    
    Args:
        large_coords: Coordinates of the larger instance
        num_nodes: Number of nodes to sample
        
    Returns:
        Tuple of (subgraph_coords, selected_indices)
    """
    total_nodes = len(large_coords)
    # Ensure we have enough nodes to sample from
    if total_nodes < num_nodes:
        raise ValueError(f"Base instance has {total_nodes} nodes, which is less than the required {num_nodes} nodes")
    
    # Select the required number of nodes from the base instance
    selected_indices = np.random.choice(total_nodes, num_nodes, replace=False)
    selected_indices = np.sort(selected_indices)  # Sort indices for consistency
    return large_coords[selected_indices], selected_indices


def generate_tsp_instances_from_large(min_nodes, max_nodes, num_instances, sampling_ratio, max_candidates=5, alpha=None, num_processes=16, instances_per_group=100):
    """
    Generate TSP instances by sampling from large base instances.
    Each node size has multiple groups, each with its own base instance.
    The base instance and sampled instances are all included in each group.
    
    Args:
        min_nodes: Minimum number of nodes in TSP instances (base instance size)
        max_nodes: Maximum number of nodes in TSP instances (base instance size)
        num_instances: Total number of instances to generate for each node count
        sampling_ratio: Ratio of nodes to sample from base instance (must be < 1.0)
        max_candidates: Maximum number of candidates per node for LKH3
        alpha: Alpha value for LKH3 candidate generation
        num_processes: Number of processes for parallel generation
        instances_per_group: Number of instances per group
        
    Returns:
        Dictionary mapping node counts to lists of TSP instances (flattened from groups)
    """
    if sampling_ratio >= 1.0:
        raise ValueError("sampling_ratio must be less than 1.0")
    
    if num_instances % instances_per_group != 0:
        raise ValueError(f"num_instances ({num_instances}) must be divisible by instances_per_group ({instances_per_group})")
    
    num_groups = num_instances // instances_per_group
    
    # Generate instances for each node count
    all_instances = {}
    for num_nodes in range(min_nodes, max_nodes + 1):
        print(f"\nGenerating {num_instances} instances for TSP-{num_nodes} ({num_groups} groups)...")
        
        # Step 1: Prepare all work (base instances + sampled coordinates) for all groups
        all_work_items = []
        base_instances_info = []
        
        for group_idx in range(num_groups):
            # Generate base instance coordinates for this group
            base_coords = np.random.uniform(0, 1, size=(num_nodes, 2))
            
            # Add base instance to work items
            all_work_items.append((base_coords, max_candidates, alpha, group_idx, 'base'))
            base_instances_info.append((group_idx, base_coords))
            
            # Generate sampled coordinates from base instance with coverage guarantee
            sample_size = max(2, round(sampling_ratio * num_nodes))
            num_sampled_instances = instances_per_group - 1  # -1 because base instance is already added
            
            # Use greedy coverage strategy to ensure all base nodes are covered
            sampled_indices_list = generate_covering_samples(
                num_nodes, sample_size, num_sampled_instances
            )
            
            for i, sampled_indices in enumerate(sampled_indices_list):
                sampled_coords = base_coords[sampled_indices]
                all_work_items.append((sampled_coords, max_candidates, alpha, group_idx, f'sampled_{i}'))
        
        print(f"  Prepared {len(all_work_items)} work items ({num_groups} base + {len(all_work_items) - num_groups} sampled)")
        
        # Step 2: Process all work items in parallel
        print(f"  Processing all instances in parallel with {num_processes} processes...")
        with Pool(num_processes) as pool:
            results = list(tqdm(
                pool.starmap(solve_tsp_lkh3_with_metadata, 
                           [(coords, max_candidates, alpha, group_idx, instance_type) 
                            for coords, max_candidates, alpha, group_idx, instance_type in all_work_items]),
                total=len(all_work_items),
                desc=f"  Generating all TSP-{num_nodes} instances",
                ncols=100
            ))
        
        # Step 3: Organize results by groups
        group_results = {}
        for (tour, candidate_info, group_idx, instance_type), (coords, _, _, _, _) in zip(results, all_work_items):
            if group_idx not in group_results:
                group_results[group_idx] = []
            
            instance = {
                'coords': coords,
                'tour': tour,
                'candidate_info': candidate_info,
                'is_base': (instance_type == 'base'),
                'group_id': group_idx,
                'instance_type': instance_type
            }
            group_results[group_idx].append(instance)
        
        # Step 4: Create groups with proper structure (same as test data)
        groups = []
        for group_idx in range(num_groups):
            if group_idx in group_results:
                # Sort instances: base instance first, then sampled instances
                group_instances_sorted = sorted(group_results[group_idx], 
                                              key=lambda x: (0 if x['is_base'] else 1, x['instance_type']))
                groups.append(group_instances_sorted)
                print(f"  Group {group_idx + 1}: {len(group_results[group_idx])} instances")
        
        all_instances[num_nodes] = groups  # Return grouped structure, not flattened
        print(f"Total: Generated {len(groups)} groups for TSP-{num_nodes}")
    
    return all_instances


def solve_tsp_lkh3_with_metadata(coords, max_candidates, alpha, group_idx, instance_type):
    """
    Wrapper function for solve_tsp_lkh3 that includes metadata.
    Used for parallel processing with additional information.
    """
    try:
        tour, candidate_info = solve_tsp_lkh3(coords, max_candidates=max_candidates, alpha=alpha)
        return tour, candidate_info, group_idx, instance_type
    except Exception as e:
        print(f"Warning: Failed to solve TSP for group {group_idx}, {instance_type}: {e}")
        # Return fallback solution
        num_nodes = len(coords)
        fallback_tour = list(range(num_nodes))
        fallback_candidate_info = {
            'dimension': num_nodes,
            'candidates': {},
            'mst_parents': {}
        }
        return fallback_tour, fallback_candidate_info, group_idx, instance_type


def generate_test_instances_from_large(min_nodes, max_nodes, test_instances, sampling_ratio, max_candidates=5, alpha=None, num_processes=16, test_instances_multiplier=5, instances_per_group=100):
    """
    Generate test TSP instances by sampling from large base instances.
    Each node size has multiple groups, each with its own base instance.
    The base instance and sampled instances are all included in each group.
    
    Args:
        min_nodes: Minimum number of nodes in TSP instances (base instance size)
        max_nodes: Maximum number of nodes in TSP instances (base instance size)
        test_instances: Total number of test instances to generate for each node count
        sampling_ratio: Ratio of nodes to sample from base instance (must be < 1.0)
        max_candidates: Maximum number of candidates per node for LKH3
        alpha: Alpha value for LKH3 candidate generation
        num_processes: Number of processes for parallel generation
        test_instances_multiplier: Multiplier for number of test instances to generate (default: 5)
        instances_per_group: Number of instances per group
        
    Returns:
        Dictionary mapping node counts to lists of groups, where each group contains sampled instances + base instance
    """
    if sampling_ratio >= 1.0:
        raise ValueError("sampling_ratio must be less than 1.0")
    
    if test_instances % instances_per_group != 0:
        raise ValueError(f"test_instances ({test_instances}) must be divisible by instances_per_group ({instances_per_group})")
    
    num_groups = test_instances // instances_per_group
    
    # Generate instances for each node count
    all_instances = {}
    for num_nodes in range(min_nodes, max_nodes + 1):
        print(f"\nGenerating test instances for TSP-{num_nodes} ({num_groups} groups)...")
        
        # Step 1: Prepare all work (base instances + sampled coordinates) for all groups
        all_work_items = []
        base_instances_info = []
        
        for group_idx in range(num_groups):
            # Generate base instance coordinates for this group
            base_coords = np.random.uniform(0, 1, size=(num_nodes, 2))
            
            # Add base instance to work items
            all_work_items.append((base_coords, max_candidates, alpha, group_idx, 'base'))
            base_instances_info.append((group_idx, base_coords))
            
            # Generate extra sampled instances (with multiplier for testing) using coverage strategy
            extra_instances = int(instances_per_group * test_instances_multiplier * 1.2)  # Generate 20% more instances
            sample_size = max(2, round(sampling_ratio * num_nodes))
            
            # Use greedy coverage strategy for test instances as well
            sampled_indices_list = generate_covering_samples(
                num_nodes, sample_size, extra_instances
            )
            
            for i, sampled_indices in enumerate(sampled_indices_list):
                sampled_coords = base_coords[sampled_indices]
                all_work_items.append((sampled_coords, max_candidates, alpha, group_idx, f'sampled_{i}'))
        
        print(f"  Prepared {len(all_work_items)} work items ({num_groups} base + {len(all_work_items) - num_groups} sampled)")
        
        # Step 2: Process all work items in parallel
        print(f"  Processing all test instances in parallel with {num_processes} processes...")
        with Pool(num_processes) as pool:
            results = list(tqdm(
                pool.starmap(solve_tsp_lkh3_with_metadata, 
                           [(coords, max_candidates, alpha, group_idx, instance_type) 
                            for coords, max_candidates, alpha, group_idx, instance_type in all_work_items]),
                total=len(all_work_items),
                desc=f"  Generating all test TSP-{num_nodes} instances",
                ncols=100
            ))
        
        # Step 3: Organize results by groups
        group_results = {}
        for (tour, candidate_info, group_idx, instance_type), (coords, _, _, _, _) in zip(results, all_work_items):
            if group_idx not in group_results:
                group_results[group_idx] = []
            
            instance = {
                'coords': coords,
                'tour': tour,
                'candidate_info': candidate_info,
                'is_base': (instance_type == 'base'),
                'group_id': group_idx,
                'instance_type': instance_type
            }
            group_results[group_idx].append(instance)
        
        # Step 4: Create groups with proper instance counts
        groups = []
        for group_idx in range(num_groups):
            if group_idx in group_results:
                group_instances = group_results[group_idx]
                
                # Separate base and sampled instances
                base_instances = [inst for inst in group_instances if inst['is_base']]
                sampled_instances = [inst for inst in group_instances if not inst['is_base']]
                
                # Take the required number of sampled instances
                required_sampled = instances_per_group * test_instances_multiplier - len(base_instances)
                sampled_instances = sampled_instances[:required_sampled]
                
                # Create group: base instance first, then sampled instances
                group = base_instances + sampled_instances
                groups.append(group)
                
                print(f"  Group {group_idx + 1}: {len(base_instances)} base + {len(sampled_instances)} sampled = {len(group)} total instances")
        
        all_instances[num_nodes] = groups
        print(f"Total: Generated {len(groups)} groups for TSP-{num_nodes}")
    
    return all_instances


def generate_test_instances_random_nodes(min_nodes, max_nodes, num_instances, max_candidates=15, alpha=1.0, num_processes=4, test_instances_multiplier=1):
    """Generate test instances using random nodes strategy"""
    test_dataset = {}
    for num_nodes in range(min_nodes, max_nodes + 1):
        instances = generate_tsp_instances(
            num_nodes, num_instances * test_instances_multiplier, max_candidates, alpha, num_processes
        )
        test_dataset[num_nodes] = instances
        print(f"Generated {len(instances)} test instances for TSP-{num_nodes}")
    return test_dataset


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


def merge_datasets(dataset_paths, output_path, generation_strategy):
    """
    Merge multiple datasets into one.
    
    Args:
        dataset_paths: List of paths to datasets to merge
        output_path: Path to save the merged dataset
        generation_strategy: Strategy used for generating the datasets
            - 'random_nodes_same_size': All instances have the same number of nodes
            - 'fix_all_nodes_same_size': Each dataset contains instances of different sizes
            - 'fix_group_nodes_same_size': Instances are grouped by node count
            - 'sample_from_large': Instances are sampled from a larger base instance
    """
    print(f"Merging {len(dataset_paths)} datasets...")
    print(f"Generation strategy: {generation_strategy}")
    
    merged_dataset = {}
    total_instances = 0
    
    for i, path in enumerate(dataset_paths):
        print(f"Processing dataset {i+1}/{len(dataset_paths)}: {os.path.basename(path)}")
        dataset = load_dataset(path)
        
        if dataset is None:
            print(f"Skipping {path} due to loading error")
            continue
            
        if generation_strategy == 'random_nodes_same_size':
            # For random_nodes_same_size, all instances have the same number of nodes
            if isinstance(dataset, list):
                # Single node dataset
                if dataset and 'coords' in dataset[0]:
                    num_nodes = len(dataset[0]['coords'])
                    if num_nodes not in merged_dataset:
                        merged_dataset[num_nodes] = []
                    merged_dataset[num_nodes].extend(dataset)
                    total_instances += len(dataset)
            elif isinstance(dataset, dict):
                # Complete dataset
                for num_nodes, instances in dataset.items():
                    if num_nodes not in merged_dataset:
                        merged_dataset[num_nodes] = []
                    merged_dataset[num_nodes].extend(instances)
                    total_instances += len(instances)
        
        elif generation_strategy == 'fix_all_nodes_same_size':
            # For fix_all_nodes_same_size, each dataset contains instances of different sizes
            if isinstance(dataset, dict):
                for num_nodes, instances in dataset.items():
                    if num_nodes not in merged_dataset:
                        merged_dataset[num_nodes] = []
                    merged_dataset[num_nodes].extend(instances)
                    total_instances += len(instances)
            else:
                print(f"Warning: Unexpected dataset format for {path} with fix_all_nodes_same_size strategy")
        
        elif generation_strategy == 'fix_group_nodes_same_size':
            # For fix_group_nodes_same_size, instances are grouped by node count
            if isinstance(dataset, dict):
                for num_nodes, instances in dataset.items():
                    if num_nodes not in merged_dataset:
                        merged_dataset[num_nodes] = []
                    merged_dataset[num_nodes].extend(instances)
                    total_instances += len(instances)
            else:
                print(f"Warning: Unexpected dataset format for {path} with fix_group_nodes_same_size strategy")
        
        elif generation_strategy == 'sample_from_large':
            # For sample_from_large, all instances are sampled from a larger base instance
            if isinstance(dataset, dict):
                for num_nodes, instances in dataset.items():
                    if num_nodes not in merged_dataset:
                        merged_dataset[num_nodes] = []
                    merged_dataset[num_nodes].extend(instances)
                    total_instances += len(instances)
            else:
                print(f"Warning: Unexpected dataset format for {path} with sample_from_large strategy")
        
        else:
            print(f"Warning: Unknown generation strategy: {generation_strategy}")
            continue
    
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
        total_instances_count = 0
        
        # Check if this is group-based format or flat format
        if len(instances) > 0 and isinstance(instances[0], list):
            # Group-based format: instances = [[group1], [group2], ...]
            total_instances_count = sum(len(group) for group in instances)
            
            # Calculate average edges from all instances
            total_edges = 0
            valid_count = 0
            for group in instances:
                for inst in group:
                    if isinstance(inst, dict) and inst.get('candidate_info') and 'candidates' in inst['candidate_info']:
                        inst_edges = sum(len(candidates) for candidates in inst['candidate_info']['candidates'].values())
                        total_edges += inst_edges
                        valid_count += 1
            if valid_count > 0:
                avg_edges = total_edges / valid_count
                
            print(f"  TSP-{num_nodes}: {len(instances)} groups, {total_instances_count} total instances, avg edges: {avg_edges:.1f}")
        else:
            # Flat format: instances = [instance1, instance2, ...]
            total_instances_count = len(instances)
            if instances and isinstance(instances[0], dict) and instances[0].get('candidate_info'):
                total_edges = 0
                valid_count = 0
                for inst in instances:
                    if isinstance(inst, dict) and inst.get('candidate_info') and 'candidates' in inst['candidate_info']:
                        inst_edges = sum(len(candidates) for candidates in inst['candidate_info']['candidates'].values())
                        total_edges += inst_edges
                        valid_count += 1
                if valid_count > 0:
                    avg_edges = total_edges / valid_count
            
            print(f"  TSP-{num_nodes}: {total_instances_count} instances, avg edges: {avg_edges:.1f}")
    
    return merged_dataset


def main():
    parser = argparse.ArgumentParser(description='Generate TSP dataset')
    parser.add_argument('--min_nodes', type=int, default=31, help='Minimum number of nodes')
    parser.add_argument('--max_nodes', type=int, default=80, help='Maximum number of nodes')
    parser.add_argument('--num_instances', type=int, default=1000, help='Number of instances per node count')
    parser.add_argument('--max_candidates', type=int, default=5, help='Maximum number of candidates per node')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha value for LKH3')
    parser.add_argument('--num_processes', type=int, default=16, help='Number of processes for parallel generation')
    parser.add_argument('--output_dir', type=str, default='./datasets', help='Output directory for datasets')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (e.g., "tsp_51to60")')
    parser.add_argument('--generation_strategy', type=str, default='random_nodes_same_size',
                        choices=['random_nodes_same_size', 'fix_all_nodes_same_size', 'fix_group_nodes_same_size', 'sample_from_large'],
                        help='Strategy for generating TSP instances')
    parser.add_argument('--fixed_nodes', type=int, default=200,
                        help='Number of fixed nodes for fix_all_nodes_same_size strategy')
    parser.add_argument('--min_fixed_nodes', type=int, default=None,
                        help='Minimum number of fixed nodes per group for fix_group_nodes_same_size strategy')
    parser.add_argument('--test_instances', type=int, default=100,
                        help='Number of test instances to generate')
    parser.add_argument('--test_instances_multiplier', type=int, default=2,
                        help='Multiplier for number of test instances to generate')
    parser.add_argument('--sampling_ratio', type=float, default=0.5,
                        help='Ratio of nodes to sample from base instance (must be < 1.0)')
    parser.add_argument('--instances_per_group', type=int, default=100,
                        help='Number of instances per group for fix_group_nodes_same_size and sample_from_large strategies')
    parser.add_argument('--test_only', action='store_true', 
                        help='Generate only test dataset, skip training dataset generation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use provided dataset name
    dataset_name = args.dataset_name
    
    # Generate training dataset (skip if test_only is True)
    if not args.test_only:
        print(f"Generating training dataset using {args.generation_strategy} strategy...")
        
        if args.generation_strategy == 'random_nodes_same_size':
            dataset = generate_tsp_instances(
                args.num_instances, args.max_nodes, args.max_candidates, args.alpha, args.num_processes
            )
        elif args.generation_strategy == 'fix_all_nodes_same_size':
            dataset, fixed_coords = generate_tsp_instances_with_fixed_nodes(
                args.min_nodes, args.max_nodes, args.num_instances,
                args.max_candidates, args.alpha, args.num_processes, args.fixed_nodes
            )
        elif args.generation_strategy == 'fix_group_nodes_same_size':
            dataset = generate_tsp_instances_with_group_fixed_nodes(
                args.min_nodes, args.max_nodes, args.num_instances,
                args.max_candidates, args.alpha, args.num_processes,
                args.min_fixed_nodes, args.instances_per_group
            )
        elif args.generation_strategy == 'sample_from_large':
            dataset = generate_tsp_instances_from_large(
                args.min_nodes, args.max_nodes, args.num_instances, args.sampling_ratio,
                args.max_candidates, args.alpha, args.num_processes, args.instances_per_group
            )
        else:
            raise ValueError(f"Unknown generation strategy: {args.generation_strategy}")
        
        # Save training dataset
        train_dataset_path = os.path.join(args.output_dir, f"{dataset_name}__{args.generation_strategy}.pkl")
        with open(train_dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Training dataset saved to {train_dataset_path}")
    else:
        print("Skipping training dataset generation (--test_only flag is set)")
        dataset = None
        fixed_coords = None
    
    # Generate test dataset
    print("\nGenerating test dataset...")
    if args.generation_strategy == 'random_nodes_same_size':
        test_dataset = generate_test_instances_random_nodes(
            args.min_nodes, args.max_nodes, args.test_instances,
            args.max_candidates, args.alpha, args.num_processes,
            args.test_instances_multiplier
        )
    elif args.generation_strategy == 'fix_all_nodes_same_size':
        if args.test_only:
            # Generate fixed_coords for test_only mode
            print("Generating fixed coordinates for test dataset...")
            fixed_coords = np.random.uniform(0, 1, size=(args.fixed_nodes, 2))
        test_dataset = generate_test_instances_with_fixed_nodes(
            args.min_nodes, args.max_nodes, args.test_instances,
            args.max_candidates, args.alpha, args.num_processes,
            args.test_instances_multiplier, fixed_coords
        )
    elif args.generation_strategy == 'fix_group_nodes_same_size':
        test_dataset = generate_test_instances_with_group_fixed_nodes(
            args.min_nodes, args.max_nodes, args.test_instances,
            args.max_candidates, args.alpha, args.num_processes,
            args.min_fixed_nodes, dataset, args.test_instances_multiplier,
            args.instances_per_group
        )
    elif args.generation_strategy == 'sample_from_large':
        test_dataset = generate_test_instances_from_large(
            args.min_nodes, args.max_nodes, args.test_instances, args.sampling_ratio,
            args.max_candidates, args.alpha, args.num_processes, args.test_instances_multiplier,
            args.instances_per_group
        )
    
    # Save test dataset
    test_dataset_path = os.path.join(args.output_dir, f"{dataset_name}__{args.generation_strategy}_test.pkl")
    with open(test_dataset_path, 'wb') as f:
        pickle.dump(test_dataset, f)
    print(f"Test dataset saved to {test_dataset_path}")
    
    # Print dataset statistics
    if not args.test_only and dataset:
        print("\nTraining dataset statistics:")
        for num_nodes in sorted(dataset.keys()):
            instances = dataset[num_nodes]
            avg_edges = 0
            total_instances_count = 0
            
            # Check if this is group-based format or flat format
            if len(instances) > 0 and isinstance(instances[0], list):
                # Group-based format: instances = [[group1], [group2], ...]
                total_instances_count = sum(len(group) for group in instances)
                
                # Calculate average edges from all instances
                total_edges = 0
                valid_count = 0
                for group in instances:
                    for inst in group:
                        if isinstance(inst, dict) and inst.get('candidate_info') and 'candidates' in inst['candidate_info']:
                            inst_edges = sum(len(candidates) for candidates in inst['candidate_info']['candidates'].values())
                            total_edges += inst_edges
                            valid_count += 1
                if valid_count > 0:
                    avg_edges = total_edges / valid_count
                    
                print(f"  TSP-{num_nodes}: {len(instances)} groups, {total_instances_count} total instances, avg edges: {avg_edges:.1f}")
            else:
                # Flat format: instances = [instance1, instance2, ...]
                total_instances_count = len(instances)
                if instances and isinstance(instances[0], dict) and instances[0].get('candidate_info'):
                    total_edges = 0
                    valid_count = 0
                    for inst in instances:
                        if isinstance(inst, dict) and inst.get('candidate_info') and 'candidates' in inst['candidate_info']:
                            inst_edges = sum(len(candidates) for candidates in inst['candidate_info']['candidates'].values())
                            total_edges += inst_edges
                            valid_count += 1
                    if valid_count > 0:
                        avg_edges = total_edges / valid_count
                
                print(f"  TSP-{num_nodes}: {total_instances_count} instances, avg edges: {avg_edges:.1f}")
    
    print("\nTest dataset statistics:")
    for num_nodes in sorted(test_dataset.keys()):
        instances = test_dataset[num_nodes]
        if isinstance(instances[0], list):
            # Group-based structure
            total_test_instances = sum(len(group) for group in instances)
            print(f"  TSP-{num_nodes}: {len(instances)} groups, {total_test_instances} total instances")
        else:
            # Flat structure
            print(f"  TSP-{num_nodes}: {len(instances)} instances")
    
    return test_dataset if args.test_only else dataset


if __name__ == "__main__":
    main() 