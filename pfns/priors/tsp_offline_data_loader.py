"""
Offline TSP DataLoader for pre-generated datasets
"""

import os
import pickle
import numpy as np
import torch
from .prior import PriorDataLoader, Batch
import random
from typing import Dict, List, Any


class TSPOfflineDataLoader(PriorDataLoader):
    """
    Offline DataLoader for pre-generated TSP instances.
    Loads data from pickle files and creates batches with same node counts.
    """
    
    def __init__(self, num_steps: int, batch_size: int, eval_pos_seq_len_sampler: callable,
                 seq_len_maximum: int, device: str, dataset_path: str, 
                 num_nodes_range: tuple = (30, 80), shuffle: bool = True, **kwargs):
        """
        Parameters:
            num_steps: Number of iterations per epoch (will be calculated based on data size)
            batch_size: Number of instances per batch
            eval_pos_seq_len_sampler: Function to sample the single evaluation position
            seq_len_maximum: Maximum number of different graphs per sequence
            device: Device for computation
            dataset_path: Path to the pre-generated dataset
            num_nodes_range: Range of node numbers to include
            shuffle: Whether to shuffle data each epoch
            **kwargs: Additional keyword arguments
        """
        self.batch_size = batch_size
        self.eval_pos_seq_len_sampler = eval_pos_seq_len_sampler
        self.seq_len_maximum = seq_len_maximum
        self.device = device
        self.dataset_path = dataset_path
        self.num_nodes_range = num_nodes_range
        self.shuffle = shuffle
        
        # Add flag to track first batch generation
        self._first_batch_generated = False
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Calculate actual number of steps based on data size
        self.num_steps = self._calculate_num_steps()
        
        # Fixed features
        self.num_features = 2  # x, y coordinates
        
        print(f"Loaded offline TSP dataset from: {dataset_path}")
        print(f"Node range: {num_nodes_range}")
        print(f"Total instances: {sum(len(instances) for instances in self.dataset.values())}")
        print(f"Steps per epoch: {self.num_steps}")
    
    def _load_dataset(self) -> Dict[int, List[Dict]]:
        """Load dataset from pickle file"""
        print(f"Loading dataset from: {self.dataset_path}")
        
        if os.path.isfile(self.dataset_path):
            # Single file containing all data
            with open(self.dataset_path, 'rb') as f:
                full_dataset = pickle.load(f)
            
            # Filter by node range
            filtered_dataset = {}
            for num_nodes in range(self.num_nodes_range[0], self.num_nodes_range[1] + 1):
                if num_nodes in full_dataset:
                    filtered_dataset[num_nodes] = full_dataset[num_nodes]
                    print(f"  TSP-{num_nodes}: {len(full_dataset[num_nodes])} instances")
        
        elif os.path.isdir(self.dataset_path):
            # Directory containing separate files
            filtered_dataset = {}
            for num_nodes in range(self.num_nodes_range[0], self.num_nodes_range[1] + 1):
                file_path = os.path.join(self.dataset_path, f'tsp_{num_nodes}nodes.pkl')
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        instances = pickle.load(f)
                    filtered_dataset[num_nodes] = instances
                    print(f"  TSP-{num_nodes}: {len(instances)} instances")
        
        else:
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        if not filtered_dataset:
            raise ValueError(f"No valid data found in range {self.num_nodes_range}")
        
        return filtered_dataset
    
    def _calculate_num_steps(self) -> int:
        """Calculate number of steps per epoch based on data size"""
        total_instances = sum(len(instances) for instances in self.dataset.values())
        # Each step processes seq_len_maximum * batch_size instances
        instances_per_step = self.seq_len_maximum * self.batch_size
        steps = max(1, total_instances // instances_per_step)
        return steps
    
    def __len__(self):
        return self.num_steps
    
    def get_test_batch(self) -> Batch:
        """Generate a test batch for initializing the model"""
        # Use middle node count for test batch
        available_nodes = list(self.dataset.keys())
        test_num_nodes = available_nodes[len(available_nodes) // 2]
        
        # Generate test batch
        x, y, candidate_info = self._create_batch_from_instances(
            self.dataset[test_num_nodes][:self.seq_len_maximum * self.batch_size],
            test_num_nodes
        )
        
        # Sample single evaluation position
        single_eval_pos, _ = self.eval_pos_seq_len_sampler()
        single_eval_pos = min(single_eval_pos, self.seq_len_maximum - 1)
        
        return Batch(x=x, y=y, target_y=y, candidate_info=candidate_info, 
                    style=None, single_eval_pos=single_eval_pos)
    
    def __iter__(self):
        # Create shuffled instance pools for each node count
        instance_pools = {}
        for num_nodes, instances in self.dataset.items():
            pool = instances.copy()
            if self.shuffle:
                random.shuffle(pool)
            instance_pools[num_nodes] = pool
        
        # Track current position in each pool
        pool_positions = {num_nodes: 0 for num_nodes in instance_pools.keys()}
        
        for step in range(self.num_steps):
            # Randomly select a node count for this batch
            available_nodes = [num_nodes for num_nodes, pos in pool_positions.items() 
                             if pos < len(instance_pools[num_nodes])]
            
            if not available_nodes:
                # Reset pools if all are exhausted
                for num_nodes in instance_pools.keys():
                    if self.shuffle:
                        random.shuffle(instance_pools[num_nodes])
                    pool_positions[num_nodes] = 0
                available_nodes = list(instance_pools.keys())
            
            current_num_nodes = random.choice(available_nodes)
            
            # Get instances for this batch
            pool = instance_pools[current_num_nodes]
            start_pos = pool_positions[current_num_nodes]
            instances_needed = self.seq_len_maximum * self.batch_size
            
            # Handle case where we don't have enough instances
            if start_pos + instances_needed > len(pool):
                # Take remaining instances and wrap around
                remaining = pool[start_pos:]
                needed_more = instances_needed - len(remaining)
                if self.shuffle:
                    random.shuffle(pool)
                batch_instances = remaining + pool[:needed_more]
                pool_positions[current_num_nodes] = needed_more
            else:
                batch_instances = pool[start_pos:start_pos + instances_needed]
                pool_positions[current_num_nodes] = start_pos + instances_needed
            
            # Create batch tensors
            x, y, candidate_info = self._create_batch_from_instances(batch_instances, current_num_nodes)
            
            # Sample single evaluation position
            single_eval_pos, _ = self.eval_pos_seq_len_sampler()
            single_eval_pos = min(single_eval_pos, self.seq_len_maximum - 1)
            
            yield Batch(x=x, y=y, target_y=y, candidate_info=candidate_info,
                       style=None, single_eval_pos=single_eval_pos)
    
    def _create_batch_from_instances(self, instances: List[Dict], num_nodes: int):
        """Create batch tensors from instance list"""
        # Initialize tensors
        x = torch.zeros(self.seq_len_maximum, self.batch_size, num_nodes, 2)
        y = torch.zeros(self.seq_len_maximum, self.batch_size, num_nodes, dtype=torch.long)
        
        candidate_info_flat = []
        
        # Fill tensors
        for i, instance in enumerate(instances):
            seq_pos = i // self.batch_size
            batch_pos = i % self.batch_size
            
            if seq_pos >= self.seq_len_maximum:
                break
            
            # Set coordinates
            x[seq_pos, batch_pos] = torch.tensor(instance['coords'], dtype=torch.float32)
            
            # Set tour
            y[seq_pos, batch_pos] = torch.tensor(instance['tour'], dtype=torch.long)
            
            # Add candidate info
            candidate_info_flat.append(instance['candidate_info'])
        
        # Print debug information for the first batch
        if not self._first_batch_generated:
            self._first_batch_generated = True
            print(f"\n=== First Batch Debug Info (Offline) ===")
            print(f"x tensor shape: {x.shape}")
            print(f"y tensor shape: {y.shape}")
            
            # Calculate average number of edges from candidate info
            if candidate_info_flat:
                total_edges = 0
                valid_instances = 0
                
                for candidate_info in candidate_info_flat:
                    if candidate_info and 'candidates' in candidate_info:
                        instance_edges = 0
                        for node_id, candidates in candidate_info['candidates'].items():
                            instance_edges += len(candidates)
                        total_edges += instance_edges
                        valid_instances += 1
                
                if valid_instances > 0:
                    avg_edges = total_edges / valid_instances
                    print(f"Average edges per graph (from LKH3 candidates): {avg_edges:.1f}")
                    print(f"Total instances with candidate info: {valid_instances}")
                else:
                    print("No valid candidate information found")
            else:
                print("No candidate information available")
            
            print(f"Number of nodes per instance: {num_nodes}")
            print(f"Batch size: {self.batch_size}")
            print(f"Sequence length (num_graphs): {self.seq_len_maximum}")
            print(f"Total TSP instances in this batch: {self.seq_len_maximum * self.batch_size}")
            print("=" * 30 + "\n")
        
        return x, y, candidate_info_flat 