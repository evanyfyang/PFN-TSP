from __future__ import annotations
import numpy as np
import torch
from .prior import PriorDataLoader, Batch
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import multiprocessing as mp
from functools import partial
import time


def solve_tsp_static(coords: np.ndarray) -> list:
    num_nodes = len(coords)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                distance_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
    
    manager = pywrapcp.RoutingIndexManager(num_nodes, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 1000)
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 1  # 添加时间限制
    
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        tour = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            tour.append(node)
            index = solution.Value(routing.NextVar(index))
        return tour
    else:
        return list(range(num_nodes))


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
                 num_processes: int = None, **kwargs):
        """
        Parameters:
            num_steps: Number of iterations per epoch.
            batch_size: Number of instances per batch.
            eval_pos_seq_len_sampler: Function to sample the single evaluation position.
            seq_len_maximum: Maximum number of different graphs per sequence, also used as the fixed number of graphs per batch.
            device: Device for computation.
            num_nodes_range: Tuple (min_nodes, max_nodes) indicating the range for the number of nodes in each TSP instance.
            num_processes: Number of processes to use for parallel TSP solving. If None, uses number of CPU cores.
            **kwargs: Additional keyword arguments.
        """
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.eval_pos_seq_len_sampler = eval_pos_seq_len_sampler
        self.seq_len_maximum = seq_len_maximum
        self.device = device
        self.num_nodes_range = num_nodes_range
        
        # Set number of processes for parallel computation
        self.num_processes = num_processes if num_processes is not None else mp.cpu_count()
        print(f"Using {self.num_processes} processes for TSP solving")
        
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
        # Use a fixed number of nodes for test batch (middle of the range)
        current_num_nodes = (self.num_nodes_range[0] + self.num_nodes_range[1]) // 2
        
        # Generate test batch with seq_len_maximum TSP graphs
        x, y = self._generate_batch(current_num_nodes, self.seq_len_maximum)
        
        # Sample single evaluation position
        single_eval_pos, _ = self.eval_pos_seq_len_sampler()
        single_eval_pos = min(single_eval_pos, self.seq_len_maximum - 1)
        
        # Return batch with test data
        return Batch(x=x, y=y, target_y=y, style=None, single_eval_pos=single_eval_pos)

    def __iter__(self):
        for _ in range(self.num_steps):
            # Generate a random number of nodes for the current batch
            current_num_nodes = np.random.randint(self.num_nodes_range[0], self.num_nodes_range[1] + 1)
            
            # Generate a batch of TSP graphs using seq_len_maximum as the number of graphs
            x, y = self._generate_batch(current_num_nodes, self.seq_len_maximum)
            
            # Sample single evaluation position
            single_eval_pos, _ = self.eval_pos_seq_len_sampler()
            single_eval_pos = min(single_eval_pos, self.seq_len_maximum - 1)
            
            # Yield the batch
            yield Batch(x=x, y=y, target_y=y, style=None, single_eval_pos=single_eval_pos)
    
    def _generate_coordinates(self, num_nodes, num_instances):
        """Generate random coordinates for TSP instances"""
        return np.random.uniform(
            self.coord_range[0], 
            self.coord_range[1], 
            size=(num_instances, num_nodes, 2)
        )
    
    def _solve_tsp_parallel(self, coords_list):
        """Solve multiple TSP instances in parallel"""
        with mp.Pool(processes=self.num_processes) as pool:
            tours = pool.map(solve_tsp_static, coords_list)
        return tours
    
    def _process_batch_set(self, coords_set):
        """Process a set of coordinates and return tours"""
        # Flatten the list for parallel processing
        flat_coords = []
        for coords_batch in coords_set:
            flat_coords.extend(coords_batch)
        
        # Solve all instances in parallel
        start_time = time.time()
        all_tours = self._solve_tsp_parallel(flat_coords)
        solve_time = time.time() - start_time
        
        # Unflatten the results
        tours_set = []
        idx = 0
        for coords_batch in coords_set:
            batch_tours = all_tours[idx:idx+len(coords_batch)]
            tours_set.append(batch_tours)
            idx += len(coords_batch)
        
        return tours_set, solve_time
    
    def _generate_batch(self, num_nodes, num_graphs):
        """
        Generate a batch of TSP graphs with solutions.
        
        Parameters:
            num_nodes: Number of nodes in each TSP instance (same for all instances in this batch)
            num_graphs: Number of different TSP graphs per batch (sequence length)
            
        Returns:
            x: Tensor of shape (num_graphs, batch_size, num_nodes, 2) representing coordinates
            y: Tensor of shape (num_graphs, batch_size, num_nodes) representing solutions (tours)
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
        
        # Solve all TSP instances in parallel
        tours_set, solve_time = self._process_batch_set(coords_set)
        
        # Store the tours
        for g in range(num_graphs):
            for b in range(self.batch_size):
                y[g, b] = torch.tensor(tours_set[g][b], dtype=torch.long)
        
        return x, y

    def solve_tsp(self, coords: np.ndarray) -> list:
        """
        Solve TSP using Google OR-Tools and return the node visitation order.
        """
        return solve_tsp_static(coords) 