from __future__ import annotations
import numpy as np
import torch
from .prior import PriorDataLoader, Batch
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import multiprocessing as mp
from functools import partial
import time
from scipy.spatial import Delaunay
import elkai
import sys
import os

# Import LKH3Wrapper from the same directory
from .lkh3_wrapper import LKH3Wrapper


def solve_tsp_static(coords: np.ndarray) -> list:
    """
    Solve TSP using Delaunay triangulation and LKH algorithm.
    
    Args:
        coords: Numpy array of node coordinates.
        
    Returns:
        List representing the TSP tour.
    """
    num_nodes = len(coords)
    
    if num_nodes < 4:
        return list(range(num_nodes))
    
    # Build sparse graph using Delaunay triangulation
    distance_matrix = np.full((num_nodes, num_nodes), np.inf)
    
    tri = Delaunay(coords)
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = simplex[i], simplex[j]
                dist = np.linalg.norm(coords[a] - coords[b])
                distance_matrix[a][b] = dist
                distance_matrix[b][a] = dist
    
    # Solve TSP using LKH algorithm via elkai
    try:
        max_dist = np.max(distance_matrix[~np.isinf(distance_matrix)]) * 10
        distance_matrix[np.isinf(distance_matrix)] = max_dist
        
        distance_matrix_int = (distance_matrix * 1000).astype(np.int32)
        tour = elkai.solve_int_matrix(distance_matrix_int)
        
        if len(tour) > num_nodes:
            tour = tour[:-1]
            
        return tour
    except Exception as e:
        print(f"Error solving TSP with elkai: {e}")
        return list(range(num_nodes))


def solve_tsp_ortools(coords: np.ndarray) -> list:
    """
    Solve TSP using Google OR-Tools with metaheuristic, running for 5 seconds.
    
    Args:
        coords: Numpy array of node coordinates.
        
    Returns:
        List representing the TSP tour.
    """
    num_nodes = len(coords)
    
    if num_nodes <= 2:
        return list(range(num_nodes))
    
    # Calculate full distance matrix (complete graph)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(coords[i] - coords[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    
    # Convert to distance callback format
    distance_matrix_int = (distance_matrix * 1000).astype(int)
    
    # Create routing model
    manager = pywrapcp.RoutingIndexManager(num_nodes, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix_int[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Setting search parameters with metaheuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 5  # 5 second time limit
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    
    # Get the solution
    if solution:
        tour = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return tour
    else:
        # Fall back to a simple tour if OR-Tools fails
        return list(range(num_nodes))


def solve_tsp_lkh3(coords: np.ndarray, max_candidates: int = 5, alpha: float = None) -> tuple:
    """
    Solve TSP using LKH3 and return both tour and candidate information.
    
    Args:
        coords: Numpy array of node coordinates.
        max_candidates: Maximum number of candidates per node.
        alpha: Alpha value for candidate generation.
        
    Returns:
        Tuple of (tour, candidate_info) where:
        - tour: List representing the TSP tour
        - candidate_info: Dictionary containing candidate set information
    """
    try:
        # Initialize LKH3 wrapper
        wrapper = LKH3Wrapper()
        
        # Solve TSP with candidates
        tour, candidate_info = wrapper.solve_tsp_with_candidates(
            coords, max_candidates=max_candidates, alpha=alpha, cleanup=True
        )
        
        return tour, candidate_info
    except Exception as e:
        print(f"Error solving TSP with LKH3: {e}")
        # Fallback to simple tour
        return list(range(len(coords))), {
            'dimension': len(coords),
            'candidates': {},
            'mst_parents': {}
        }


def solve_tsp_lkh3_parallel_worker(args):
    """Worker function for parallel LKH3 solving"""
    coords, max_candidates, alpha = args
    return solve_tsp_lkh3(coords, max_candidates, alpha)


class TSPDataLoader(PriorDataLoader):
    """
    DataLoader for generating TSP instances.
    All TSP instances in a batch have the same number of nodes.
    Generated TSP instances are random coordinates, and their solutions are solved using Google OR-Tools.
    
    The data structure follows these principles:
    - Each sequence position (seq_len) corresponds to a complete TSP graph
    - Each batch contains multiple TSP instances with the same number of nodes
    - All batches use seq_len_maximum as the fixed number of graphs (sequence length)
    - Different batches can have different numbers of nodes
    - y and target_y both contain the TSP solutions
    """
    def __init__(self, num_steps: int, batch_size: int, eval_pos_seq_len_sampler: callable, 
                 seq_len_maximum: int, device: str, num_nodes_range: tuple = (10, 10),
                 num_processes: int = 16, include_ortools: bool = False, 
                 use_lkh3: bool = True, max_candidates: int = 15, alpha: float = None, **kwargs):
        """
        Parameters:
            num_steps: Number of iterations per epoch.
            batch_size: Number of instances per batch.
            eval_pos_seq_len_sampler: Function to sample the single evaluation position.
            seq_len_maximum: Maximum number of different graphs per sequence, also used as the fixed number of graphs per batch.
            device: Device for computation.
            num_nodes_range: Tuple (min_nodes, max_nodes) indicating the range for the number of nodes in each TSP instance.
            num_processes: Number of processes to use for parallel TSP solving. If None, uses number of CPU cores.
            include_ortools: Whether to include solutions computed using OR-Tools in the batches.
            use_lkh3: Whether to use LKH3 solver instead of elkai for generating candidate sets.
            max_candidates: Maximum number of candidates per node for LKH3.
            alpha: Alpha value for LKH3 candidate generation (optional, uses LKH3 default if None).
            **kwargs: Additional keyword arguments.
        """
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.eval_pos_seq_len_sampler = eval_pos_seq_len_sampler
        self.seq_len_maximum = seq_len_maximum
        self.device = device
        self.num_nodes_range = num_nodes_range
        self.include_ortools = include_ortools
        self.use_lkh3 = use_lkh3
        self.max_candidates = max_candidates
        self.alpha = alpha
        
        # Add flag to track first batch generation
        self._first_batch_generated = False
        
        # Set number of processes for parallel computation
        self.num_processes = num_processes if num_processes is not None else mp.cpu_count()
        print(f"Using {self.num_processes} processes for TSP solving")
        
        if self.use_lkh3:
            if self.alpha is not None:
                print(f"Using LKH3 solver with max_candidates={self.max_candidates}, alpha={self.alpha}")
            else:
                print(f"Using LKH3 solver with max_candidates={self.max_candidates}, alpha=LKH3_default")
        else:
            print("Using elkai solver")
        
        # Use seq_len_maximum as the fixed number of graphs per batch
        self.coord_range = (0, 1)  # Fixed coordinate range
        self.num_features = 2  # x, y coordinates

    def __len__(self):
        return self.num_steps

    def get_test_batch(self) -> Batch:
        """
        Generate a test batch for initializing the model.
        Returns a Batch object with test data where each sequence position is a complete TSP graph.
        """
        # Use a fixed number of nodes for test batch
        current_num_nodes = (self.num_nodes_range[0] + self.num_nodes_range[1]) // 2
        
        # Generate test batch with seq_len_maximum TSP graphs
        if self.include_ortools:
            x, y, candidate_info, ortools_solution = self._generate_batch(current_num_nodes, self.seq_len_maximum, include_ortools=True)
        else:
            x, y, candidate_info = self._generate_batch(current_num_nodes, self.seq_len_maximum, include_ortools=False)
        
        # Sample single evaluation position
        single_eval_pos, _ = self.eval_pos_seq_len_sampler()
        single_eval_pos = min(single_eval_pos, self.seq_len_maximum - 1)
        
        # Return batch with test data
        if self.include_ortools:
            return Batch(x=x, y=y, target_y=y, ortools_solution=ortools_solution, 
                        candidate_info=candidate_info, style=None, single_eval_pos=single_eval_pos)
        else:
            return Batch(x=x, y=y, target_y=y, candidate_info=candidate_info, 
                        style=None, single_eval_pos=single_eval_pos)

    def __iter__(self):
        for _ in range(self.num_steps):
            # Generate a random number of nodes for the current batch
            current_num_nodes = np.random.randint(self.num_nodes_range[0], self.num_nodes_range[1] + 1)
            
            # Generate a batch of TSP graphs using seq_len_maximum as the number of graphs
            if self.include_ortools:
                x, y, candidate_info, ortools_solution = self._generate_batch(current_num_nodes, self.seq_len_maximum, include_ortools=True)
            else:
                x, y, candidate_info = self._generate_batch(current_num_nodes, self.seq_len_maximum, include_ortools=False)
            
            # Sample single evaluation position
            single_eval_pos, _ = self.eval_pos_seq_len_sampler()
            single_eval_pos = min(single_eval_pos, self.seq_len_maximum - 1)
            
            # Yield the batch
            if self.include_ortools:
                yield Batch(x=x, y=y, target_y=y, ortools_solution=ortools_solution, 
                           candidate_info=candidate_info, style=None, single_eval_pos=single_eval_pos)
            else:
                yield Batch(x=x, y=y, target_y=y, candidate_info=candidate_info, 
                           style=None, single_eval_pos=single_eval_pos)
    
    def _generate_coordinates(self, num_nodes, num_instances):
        """Generate random coordinates for TSP instances"""
        return np.random.uniform(
            self.coord_range[0], 
            self.coord_range[1], 
            size=(num_instances, num_nodes, 2)
        )
    
    def _solve_tsp_parallel(self, coords_list, solver_func=solve_tsp_static):
        """Solve multiple TSP instances in parallel"""
        with mp.Pool(processes=self.num_processes) as pool:
            tours = pool.map(solver_func, coords_list)
        return tours
    
    def _solve_tsp_lkh3_parallel(self, coords_list):
        """Solve multiple TSP instances in parallel using LKH3"""
        # Prepare arguments for parallel processing
        args_list = [(coords, self.max_candidates, self.alpha) for coords in coords_list]
        
        with mp.Pool(processes=self.num_processes) as pool:
            results = pool.map(solve_tsp_lkh3_parallel_worker, args_list)
        
        # Separate tours and candidate info
        all_tours = [result[0] for result in results]
        all_candidates = [result[1] for result in results]
        
        return all_tours, all_candidates
    
    def _process_batch_set(self, coords_set, include_ortools=False):
        """Process a set of coordinates and return tours and candidate information"""
        # Flatten the list for parallel processing
        flat_coords = []
        for coords_batch in coords_set:
            flat_coords.extend(coords_batch)
        
        # Solve all instances using the selected solver
        start_time = time.time()
        
        if self.use_lkh3:
            # Use LKH3 solver with candidate generation - NOW PARALLEL!
            all_tours, all_candidates = self._solve_tsp_lkh3_parallel(flat_coords)
        else:
            # Use original elkai solver
            all_tours = self._solve_tsp_parallel(flat_coords)
            all_candidates = [None] * len(all_tours)  # No candidate info for elkai
        
        solve_time = time.time() - start_time
        
        # Solve using OR-Tools if requested
        start_time = time.time()
        ortools_solve_time = 0
        ortools_tours = None
        if include_ortools:
            start_time = time.time()
            ortools_tours = self._solve_tsp_parallel(flat_coords, solver_func=solve_tsp_ortools)
            ortools_time = time.time() - start_time
            print(f"OR-Tools solve time: {ortools_time:.2f}s for {len(flat_coords)} instances")

        ortools_solve_time = time.time() - start_time
        
        # Unflatten the results
        tours_set = []
        candidates_set = []
        ortools_tours_set = []
        idx = 0
        for coords_batch in coords_set:
            batch_tours = all_tours[idx:idx+len(coords_batch)]
            batch_candidates = all_candidates[idx:idx+len(coords_batch)]
            tours_set.append(batch_tours)
            candidates_set.append(batch_candidates)
            
            if include_ortools:
                batch_ortools_tours = ortools_tours[idx:idx+len(coords_batch)]
                ortools_tours_set.append(batch_ortools_tours)
            
            idx += len(coords_batch)
        
        if include_ortools:
            return tours_set, candidates_set, ortools_tours_set, solve_time, ortools_solve_time
        else:
            return tours_set, candidates_set, solve_time
    
    def _generate_batch(self, num_nodes, num_graphs, include_ortools=False):
        """
        Generate a batch of TSP graphs with solutions.
        
        Parameters:
            num_nodes: Number of nodes in each TSP instance (same for all instances in this batch)
            num_graphs: Number of different TSP graphs per batch (sequence length)
            include_ortools: Whether to include solutions computed using OR-Tools with complete graph
            
        Returns:
            x: Tensor of shape (num_graphs, batch_size, num_nodes, 2) representing coordinates
            y: Tensor of shape (num_graphs, batch_size, num_nodes) representing solutions (tours)
            candidate_info: List of candidate information for each instance (if using LKH3)
            ortools_solution: (Optional) Tensor of shape (num_graphs, batch_size, num_nodes) 
                             representing OR-Tools solutions using complete graph
        """
        # Initialize tensors to store results
        x = torch.zeros(num_graphs, self.batch_size, num_nodes, 2)
        y = torch.zeros(num_graphs, self.batch_size, num_nodes, dtype=torch.long)
        
        # Generate all coordinates at once
        coords_set = []
        for g in range(num_graphs):
            coords_batch = self._generate_coordinates(num_nodes, self.batch_size)
            coords_set.append(coords_batch)
            x[g] = torch.tensor(coords_batch, dtype=torch.float32)
        
        # Solve all TSP instances in parallel, optionally including OR-Tools solutions
        if include_ortools:
            tours_set, candidates_set, ortools_tours_set, solve_time, ortools_solve_time = self._process_batch_set(coords_set, include_ortools=True)
            
            # Create tensor for OR-Tools solutions
            ortools_solution = torch.zeros(num_graphs, self.batch_size, num_nodes, dtype=torch.long)
            
            # Store the OR-Tools tours
            for g in range(num_graphs):
                for b in range(self.batch_size):
                    ortools_solution[g, b] = torch.tensor(ortools_tours_set[g][b], dtype=torch.long)
        else:
            tours_set, candidates_set, solve_time = self._process_batch_set(coords_set)
            ortools_solution = None
        
        # Store the tours
        for g in range(num_graphs):
            for b in range(self.batch_size):
                y[g, b] = torch.tensor(tours_set[g][b], dtype=torch.long)
        
        # Flatten candidate information for return
        candidate_info_flat = []
        for g in range(num_graphs):
            for b in range(self.batch_size):
                candidate_info_flat.append(candidates_set[g][b])
        
        # Print debug information for the first batch
        if not self._first_batch_generated:
            self._first_batch_generated = True
            print(f"\n=== First Batch Debug Info ===")
            print(f"x tensor shape: {x.shape}")
            print(f"y tensor shape: {y.shape}")
            
            # Calculate average number of edges from candidate info
            if self.use_lkh3 and candidate_info_flat:
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
                print("No LKH3 candidate information available")
            
            print(f"Number of nodes per instance: {num_nodes}")
            print(f"Batch size: {self.batch_size}")
            print(f"Sequence length (num_graphs): {num_graphs}")
            print(f"Total TSP instances in this batch: {num_graphs * self.batch_size}")
            print("=" * 30 + "\n")
        
        if include_ortools:
            return x, y, candidate_info_flat, ortools_solution, ortools_solve_time
        else:
            return x, y, candidate_info_flat

    def solve_tsp(self, coords: np.ndarray) -> list:
        """
        Solve TSP using Google OR-Tools and return the node visitation order.
        """
        return solve_tsp_static(coords) 