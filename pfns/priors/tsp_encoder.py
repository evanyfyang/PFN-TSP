import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_scatter import scatter_mean
import torch_geometric.utils as g_utils
from torch_geometric.nn import global_mean_pool
import math
from ..tsp_nets import Net
from scipy.spatial import Delaunay

class Args:
    """Helper class to provide args for Net initialization"""
    def __init__(self, emb_depth=3, net_units=128, net_act_fn=torch.nn.ReLU(), emb_agg_fn=global_mean_pool, device='cuda', par_depth=3):
        self.emb_depth = emb_depth
        self.net_units = net_units
        self.net_act_fn = net_act_fn
        self.emb_agg_fn = emb_agg_fn
        self.device = device
        self.par_depth = par_depth

class TSPGraphEncoder(nn.Module):
    """
    Encoder for TSP instances using the Net class from tsp_nets.py.
    Takes coordinates and converts them to graph embeddings.
    
    Features:
    - Works with a fixed number of TSP graphs per batch (using seq_len_maximum)
    - Creates a complete graph for each TSP instance 
    - Produces node embeddings and edge information for downstream processing
    """
    def __init__(self, num_features, emsize, max_candidates=50):
        """
        Parameters:
            num_features: Number of features (2 for 2D coordinates)
            emsize: Size of output embeddings
            max_candidates: Maximum number of candidate edges per node
        """
        super().__init__()
        # Store dimensions
        self.num_features = num_features
        self.emsize = emsize
        self.max_candidates = max_candidates
        
        # Initialize Args with appropriate parameters for TSP task
        args = Args(
            emb_depth=6, 
            net_units=emsize, 
            net_act_fn=torch.nn.SiLU(), 
            emb_agg_fn=global_mean_pool,  
            par_depth=3
        )
        
        # Create Net instance
        self.net = Net(args)
    
        
    def forward(self, x, candidate_info=None, gat_pooling=None):
        """
        Forward pass through the GNN encoder.
        
        Args:
            x: Tensor of shape (seq_len, batch, num_nodes, 2) containing node coordinates for TSP graphs
                - seq_len: Number of different TSP problems (fixed as seq_len_maximum)
                - batch: Batch size
                - num_nodes: Number of nodes in each TSP instance
                - 2: X,Y coordinates
            candidate_info: List of candidate information dictionaries from LKH3, one for each (seq_len, batch) pair
                Each dictionary contains:
                - dimension: Number of nodes
                - candidates: Dict mapping node_id to list of (neighbor_id, alpha_value) tuples
                - mst_parents: Dict mapping node_id to parent in MST
            gat_pooling: Optional GAT pooling module for attention-based aggregation
        
        Returns:
            Dictionary containing:
                - node_embeddings: Tensor of shape (seq_len, batch, emsize)
                - edge_info: Tuple of (edge_emb, edge_index, batch_tensor, position_tensor, node_offset_map, edge_counts)
        """
        # Get dimensions
        seq_len, batch_size, num_nodes, _ = x.shape
        
        all_edge_indices = []
        all_edge_attrs = []
        all_batch_ids = []
        all_position_ids = []  
        
        cumulative_nodes = 0
        node_offset_map = {}  
        
        edge_counts = []  
        
        for pos in range(seq_len):
            for b in range(batch_size):
                coords = x[pos, b]  # (num_nodes, 2)
                
                for n in range(num_nodes):
                    node_offset_map[(pos, b, n)] = cumulative_nodes + n
                
                # Use candidate_info if available, otherwise fall back to complete graph
                if candidate_info is not None:
                    # Calculate the index in the flattened candidate_info list
                    candidate_idx = pos * batch_size + b
                    if candidate_idx < len(candidate_info) and candidate_info[candidate_idx] is not None:
                        # Use LKH3 candidate edges and supplement with nearest neighbors if needed
                        edges = set()
                        cand_info = candidate_info[candidate_idx]
                        
                        # Determine target edges per node
                        target_edges_per_node = min(self.max_candidates, num_nodes - 1) if num_nodes > 1 else 0
                        
                        # Build adjacency list from LKH3 candidates
                        node_candidates = {}
                        for node_id, candidates in cand_info['candidates'].items():
                            src_node = node_id - 1  # Convert from 1-based to 0-based indexing
                            if src_node not in node_candidates:
                                node_candidates[src_node] = []
                            
                            for neighbor_id, alpha_value in candidates:
                                dst_node = neighbor_id - 1
                                if dst_node != src_node:  # Avoid self-loops
                                    node_candidates[src_node].append((dst_node, alpha_value))
                        
                        # Calculate distance matrix for finding nearest neighbors if needed
                        dist_matrix = torch.cdist(coords, coords, p=2)  # Euclidean distance
                        
                        # For each node, select up to target_edges_per_node edges
                        for node in range(num_nodes):
                            # Get LKH3 candidates for this node
                            lkh3_candidates = node_candidates.get(node, [])
                            
                            # Sort LKH3 candidates by alpha value (or distance if alpha is same)
                            lkh3_candidates.sort(key=lambda x: x[1])  # Sort by alpha value
                            
                            # Take up to target_edges_per_node from LKH3 candidates
                            selected_neighbors = []
                            for neighbor, alpha in lkh3_candidates[:target_edges_per_node]:
                                selected_neighbors.append(neighbor)
                                edges.add((min(node, neighbor), max(node, neighbor)))
                            
                            # If we need more edges for this node, add nearest neighbors
                            if len(selected_neighbors) < target_edges_per_node:
                                # Get distances from this node to all others
                                distances = dist_matrix[node]
                                # Sort by distance (excluding self and already selected)
                                sorted_indices = torch.argsort(distances)
                                
                                for neighbor_idx in sorted_indices:
                                    neighbor = neighbor_idx.item()
                                    if (neighbor != node and 
                                        neighbor not in selected_neighbors and 
                                        len(selected_neighbors) < target_edges_per_node):
                                        selected_neighbors.append(neighbor)
                                        edges.add((min(node, neighbor), max(node, neighbor)))
                        
                        if len(edges) > 0:
                            edge_index = torch.tensor(list(edges), dtype=torch.long, device=x.device).t().contiguous()
                        else:
                            # Fallback to complete graph if no edges could be created
                            adj_matrix = torch.triu(torch.ones(num_nodes, num_nodes, device=x.device), diagonal=1)
                            edge_index = g_utils.dense_to_sparse(adj_matrix)[0]
                    else:
                        # Fallback to complete graph if candidate_info is missing
                        adj_matrix = torch.triu(torch.ones(num_nodes, num_nodes, device=x.device), diagonal=1)
                        edge_index = g_utils.dense_to_sparse(adj_matrix)[0]
                else:
                    # Fallback to complete graph if no candidate_info provided
                    adj_matrix = torch.triu(torch.ones(num_nodes, num_nodes, device=x.device), diagonal=1)
                    edge_index = g_utils.dense_to_sparse(adj_matrix)[0]

                edge_index += cumulative_nodes
                
                rows, cols = edge_index - cumulative_nodes
                edge_attr = torch.norm(coords[rows] - coords[cols], dim=1).unsqueeze(1)
                edge_index = g_utils.to_undirected(edge_index)
                edge_attr = edge_attr.repeat(2, 1)
                
                all_edge_indices.append(edge_index)
                all_edge_attrs.append(edge_attr)
                
                all_batch_ids.append(torch.full((num_nodes,), b, dtype=torch.long, device=x.device))
                all_position_ids.append(torch.full((num_nodes,), pos, dtype=torch.long, device=x.device))
                
                cumulative_nodes += num_nodes
                edge_counts.append(edge_index.size(1))  
        
        edge_index = torch.cat(all_edge_indices, dim=1)
        edge_attr = torch.cat(all_edge_attrs, dim=0)
        batch_tensor = torch.cat(all_batch_ids, dim=0)
        position_tensor = torch.cat(all_position_ids, dim=0)
        
        # Reshape input for the GNN: (seq_len, batch, num_nodes, 2) -> (seq_len*batch*num_nodes, 2)
        x_flat = x.reshape(-1, self.num_features)
        
        graph_emb, edge_emb = self.net(
            x=x_flat, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            batch=batch_tensor,
            position=position_tensor,  # Pass position information
            gat_pooling=gat_pooling
        )
            
        # Print debug information for the first batch
        if not hasattr(self, '_first_batch_processed'):
            self._first_batch_processed = True
            print(f"\n=== TSPGraphEncoder Debug Info ===")
            print(f"Input shape: {x.shape}")
            
            # Calculate actual edge statistics
            if candidate_info is not None:
                total_candidate_edges = 0
                total_final_edges = 0
                valid_instances = 0
                
                for i, info in enumerate(candidate_info):
                    if info and 'candidates' in info:
                        candidate_edges = sum(len(candidates) for candidates in info['candidates'].values())
                        total_candidate_edges += candidate_edges
                        
                        # Calculate final edges from edge_counts
                        if i < len(edge_counts):
                            # edge_counts includes both directions, so divide by 2
                            final_edges = edge_counts[i] // 2
                            total_final_edges += final_edges
                        
                        valid_instances += 1
                
                if valid_instances > 0:
                    avg_candidate_edges = total_candidate_edges / valid_instances
                    avg_final_edges = total_final_edges / valid_instances
                    print(f"Average candidate edges per graph: {avg_candidate_edges:.1f}")
                    print(f"Average final edges per graph (after supplementing): {avg_final_edges:.1f}")
                    print(f"Target edges per node: {min(self.max_candidates, num_nodes - 1)}")
                    print(f"Average edges per node (final): {avg_final_edges / num_nodes:.1f}")
                else:
                    print("No valid candidate information found")
            else:
                print("No candidate_info provided - using complete graphs")
            
            print(f"Number of nodes per instance: {num_nodes}")
            print("=" * 40 + "\n")
        
        # Return both node embeddings and edge information in a dictionary
        return {
            'node_embeddings': graph_emb,
            'edge_info': (edge_emb, edge_index, batch_tensor, position_tensor, node_offset_map, edge_counts),
        }


class TSPTourEncoder(nn.Module):
    """
    Encoder for TSP tours using edge embeddings.
    Takes tour indices and computes the average edge embedding along the tour.
    """
    def __init__(self, num_features, emsize, max_nodes=100):
        """
        Parameters:
            num_features: Ignored (for compatibility)
            emsize: Size of output embeddings
            max_nodes: Maximum number of nodes in a TSP instance
        """
        super().__init__()
        self.emsize = emsize
        self.max_nodes = max_nodes
        
        # self.backup_embedding = nn.Embedding(max_nodes, emsize)
        
    def forward(self, y, edge_emb=None, edge_index=None, batch=None, position=None, node_offset_map=None, gat_pooling=None):
        """
        Forward pass through the tour encoder.
        
        Args:
            y: Tensor of shape (seq_len, batch, num_nodes) containing tour indices
            edge_emb: Optional tensor of edge embeddings from TSPGraphEncoder
            edge_index: Optional tensor of edge indices from TSPGraphEncoder
            batch: Optional tensor of batch indices for each node
            position: Optional tensor of position indices for each node
            node_offset_map: Optional dictionary mapping (pos, batch, node) to global node indices
            gat_pooling: Optional GAT pooling module for attention-based aggregation
            
        Returns:
            Tensor of shape (seq_len, batch, emsize) containing tour embeddings
        """
        if len(y.shape) == 2:
            y = y.unsqueeze(-1)
            
        seq_len, batch_size, num_nodes = y.shape
        
        tour_embeddings = torch.zeros(seq_len, batch_size, self.emsize, device=y.device)
        
        # Extract edge indices
        src, dst = edge_index
        
        # Create a more efficient edge lookup using sorting
        # Combine src and dst into a single tensor for efficient lookup
        edge_pairs = torch.stack([src, dst], dim=1)  # [num_edges, 2]
        edge_pairs_flipped = torch.stack([dst, src], dim=1)  # [num_edges, 2] (reverse direction)
        
        # Create combined edge tensor with both directions
        all_edge_pairs = torch.cat([edge_pairs, edge_pairs_flipped], dim=0)  # [2*num_edges, 2]
        all_edge_indices = torch.cat([torch.arange(len(src), device=y.device), 
                                     torch.arange(len(src), device=y.device)], dim=0)  # [2*num_edges]
        
        # Sort edges for efficient lookup
        edge_keys = all_edge_pairs[:, 0] * (src.max() + 1) + all_edge_pairs[:, 1]
        sorted_indices = torch.argsort(edge_keys)
        sorted_edge_keys = edge_keys[sorted_indices]
        sorted_edge_indices = all_edge_indices[sorted_indices]
        
        for pos in range(seq_len):
            for b in range(batch_size):
                tour = y[pos, b]  # (num_nodes,)
                
                # Create tour edges (including return to start)
                tour_edges = torch.stack([tour[:-1], tour[1:]], dim=0)  # (2, num_nodes-1)
                final_edge = torch.tensor([[tour[-1]], [tour[0]]], device=y.device)
                tour_edges = torch.cat([tour_edges, final_edge], dim=1)  # (2, num_nodes)
                
                # Convert tour edges to global indices and create lookup keys
                tour_edge_keys = []
                valid_tour_edges = []
                
                for i in range(tour_edges.shape[1]):
                    src_idx = tour_edges[0, i].item()
                    dst_idx = tour_edges[1, i].item()
                    
                    if node_offset_map is not None:
                        if (pos, b, src_idx) in node_offset_map and (pos, b, dst_idx) in node_offset_map:
                            global_src = node_offset_map[(pos, b, src_idx)]
                            global_dst = node_offset_map[(pos, b, dst_idx)]
                            
                            # Create lookup key
                            key = global_src * (src.max() + 1) + global_dst
                            tour_edge_keys.append(key)
                            valid_tour_edges.append((global_src, global_dst))
                
                if tour_edge_keys:
                    # Convert to tensor for efficient lookup
                    tour_keys_tensor = torch.tensor(tour_edge_keys, device=y.device)
                    
                    # Use searchsorted for efficient lookup
                    indices = torch.searchsorted(sorted_edge_keys, tour_keys_tensor)
                    
                    # Collect valid edge embeddings
                    all_tour_edge_embs = []
                    for i, idx in enumerate(indices):
                        if idx < len(sorted_edge_keys) and sorted_edge_keys[idx] == tour_keys_tensor[i]:
                            edge_idx = sorted_edge_indices[idx]
                            all_tour_edge_embs.append(edge_emb[edge_idx])
                
                    # Perform single unified pooling on all collected edge embeddings
                    if len(all_tour_edge_embs) > 0:
                        tour_edge_embs = torch.stack(all_tour_edge_embs, dim=0)  # [total_edges, emsize]
                        
                        # Use GAT pooling for unified tour embedding if available
                        if gat_pooling is not None:
                            batch_indices = torch.zeros(tour_edge_embs.size(0), dtype=torch.long, device=y.device)
                            tour_embedding = gat_pooling(tour_edge_embs, batch_indices)[0]
                        else:
                            tour_embedding = tour_edge_embs.mean(dim=0)
                    else:
                        tour_embedding = torch.zeros(self.emsize, device=y.device)
                else:
                    tour_embedding = torch.zeros(self.emsize, device=y.device)
                
                tour_embeddings[pos, b, :] = tour_embedding
        
        return tour_embeddings


def tsp_graph_encoder_generator(num_features, emsize, max_candidates=50):
    """Generator function for TSP graph encoder"""
    return TSPGraphEncoder(num_features, emsize, max_candidates)

def tsp_tour_encoder_generator(num_features, emsize, max_nodes=100):
    """Generator function for TSP tour encoder"""
    return TSPTourEncoder(num_features, emsize, max_nodes) 