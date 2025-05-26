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
    def __init__(self, num_features, emsize):
        """
        Parameters:
            num_features: Number of features (2 for 2D coordinates)
            emsize: Size of output embeddings
        """
        super().__init__()
        # Store dimensions
        self.num_features = num_features
        self.emsize = emsize
        
        # Initialize Args with appropriate parameters for TSP task
        args = Args(
            emb_depth=3, 
            net_units=emsize, 
            net_act_fn=torch.nn.ReLU(), 
            emb_agg_fn=global_mean_pool,  
            par_depth=3
        )
        
        # Create Net instance
        self.net = Net(args)
    
        
    def forward(self, x, candidate_info=None):
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
                        # Use LKH3 candidate edges
                        edges = set()
                        cand_info = candidate_info[candidate_idx]
                        
                        # Extract edges from candidate information
                        for node_id, candidates in cand_info['candidates'].items():
                            # Convert from 1-based to 0-based indexing
                            src_node = node_id - 1
                            for neighbor_id, alpha_value in candidates:
                                dst_node = neighbor_id - 1
                                # Add undirected edge (both directions)
                                edges.add((min(src_node, dst_node), max(src_node, dst_node)))
                        
                        if len(edges) > 0:
                            edge_index = torch.tensor(list(edges), dtype=torch.long, device=x.device).t().contiguous()
                        else:
                            # Fallback to complete graph if no candidates
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
            position=position_tensor  # Pass position information
        )
            
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
        
    def forward(self, y, edge_emb=None, edge_index=None, batch=None, position=None, node_offset_map=None):
        """
        Forward pass through the tour encoder.
        
        Args:
            y: Tensor of shape (seq_len, batch, num_nodes) containing tour indices
            edge_emb: Optional tensor of edge embeddings from TSPGraphEncoder
            edge_index: Optional tensor of edge indices from TSPGraphEncoder
            batch: Optional tensor of batch indices for each node
            position: Optional tensor of position indices for each node
            node_offset_map: Optional dictionary mapping (pos, batch, node) to global node indices
            
        Returns:
            Tensor of shape (seq_len, batch, emsize) containing tour embeddings
        """
        if len(y.shape) == 2:
            y = y.unsqueeze(-1)
            
        seq_len, batch_size, num_nodes = y.shape
        
        tour_embeddings = torch.zeros(seq_len, batch_size, self.emsize, device=y.device)
        
        for pos in range(seq_len):
            for b in range(batch_size):
                tour = y[pos, b]  # (num_nodes,)
                
                tour_edges = torch.stack([tour[:-1], tour[1:]], dim=0)  # (2, num_nodes-1)
                
                final_edge = torch.tensor([[tour[-1]], [tour[0]]], device=y.device)
                tour_edges = torch.cat([tour_edges, final_edge], dim=1)  # (2, num_nodes)
                
                edge_embs_for_tour = []
                src, dst = edge_index
                
                for i in range(tour_edges.shape[1]):
                    src_idx = tour_edges[0, i].item()
                    dst_idx = tour_edges[1, i].item()
                    
                    if node_offset_map is not None:
                        if (pos, b, src_idx) in node_offset_map and (pos, b, dst_idx) in node_offset_map:
                            global_src = node_offset_map[(pos, b, src_idx)]
                            global_dst = node_offset_map[(pos, b, dst_idx)]
                            
                            edge_mask = ((src == global_src) & (dst == global_dst))
                            if not edge_mask.any():
                                edge_mask = ((src == global_dst) & (dst == global_src))
                        else:
                            edge_mask = torch.zeros_like(src, dtype=torch.bool)
                    else:
                        if position is not None and batch is not None:
                            edge_mask = torch.zeros_like(src, dtype=torch.bool)
                            
                            for e_idx in range(len(src)):
                                src_node, dst_node = src[e_idx], dst[e_idx]
                                src_pos = position[src_node].item()
                                dst_pos = position[dst_node].item()
                                src_batch = batch[src_node].item()
                                dst_batch = batch[dst_node].item()
                                
                                # Check if this edge connects nodes in the current position and batch
                                if (src_pos == pos and dst_pos == pos and 
                                    src_batch == b and dst_batch == b):
                                    src_local_idx = None
                                    dst_local_idx = None
                                    
                                    for n in range(num_nodes):
                                        if (pos, b, n) in node_offset_map:
                                            if node_offset_map[(pos, b, n)] == src_node:
                                                src_local_idx = n
                                            if node_offset_map[(pos, b, n)] == dst_node:
                                                dst_local_idx = n
                                    
                                    if ((src_local_idx == src_idx and dst_local_idx == dst_idx) or
                                        (src_local_idx == dst_idx and dst_local_idx == src_idx)):
                                        edge_mask[e_idx] = True
                                        break
                        else:
                            edge_mask = torch.zeros_like(src, dtype=torch.bool)
                    
                    if edge_mask.any():
                        edge_idx = torch.where(edge_mask)[0][0]
                        edge_embs_for_tour.append(edge_emb[edge_idx])
                    else:
                        edge_embs_for_tour.append(torch.zeros(self.emsize, device=y.device))
                
                tour_edge_embs = torch.stack(edge_embs_for_tour, dim=0)  # (num_nodes, emsize)
                
                tour_embedding = tour_edge_embs.mean(dim=0)
                
                tour_embeddings[pos, b, :] = tour_embedding
        
        return tour_embeddings


def tsp_graph_encoder_generator(num_features, emsize):
    """Generator function for TSP graph encoder"""
    return TSPGraphEncoder(num_features, emsize)

def tsp_tour_encoder_generator(num_features, emsize, max_nodes=100):
    """Generator function for TSP tour encoder"""
    return TSPTourEncoder(num_features, emsize, max_nodes) 