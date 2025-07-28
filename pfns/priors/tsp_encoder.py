import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_scatter import scatter_mean
import torch_geometric.utils as g_utils
from torch_geometric.nn import global_mean_pool
import math
from ..tsp_nets import Net, MultiRelBasisEmbNet
from scipy.spatial import Delaunay
import torch.nn.functional as F
import time

class Args:
    """
    Configuration class for network initialization.
    """
    def __init__(self, emb_depth=6, net_units=128, net_act_fn=torch.nn.ReLU(), emb_agg_fn=global_mean_pool, device='cuda', par_depth=3):
        self.emb_depth = emb_depth
        self.net_units = net_units
        self.net_act_fn = net_act_fn
        self.emb_agg_fn = emb_agg_fn
        self.device = device
        self.par_depth = par_depth

class TSPGraphEncoder(nn.Module):
    """
    Converts coordinates to graph embeddings using vectorized operations.
    """
    def __init__(self, num_features, emsize, max_candidates=5, use_unified_encoding=False, use_shared_basis_film=False, use_instance_hypergraph=False, merge_duplicate_coords=True, num_instances=None, loss_direction_mode='both', edge_type_mode='triple', prediction_mode='dot_product', use_residual_norm=False):
        """
        Initialize TSP graph encoder.
        
        Args:
            edge_type_mode: 'triple' for three edge types (0=graph, 1=tour, 2=center), 
                           'single' for one edge type with additional features (is_solution, is_context)
            use_instance_hypergraph: Use InstanceAwareHypergraphGNN model
        """
        super().__init__()
        self.num_features = num_features
        self.emsize = emsize
        self.max_candidates = max_candidates
        self.use_unified_encoding = use_unified_encoding
        self.use_shared_basis_film = use_shared_basis_film
        self.use_instance_hypergraph = use_instance_hypergraph
        self.merge_duplicate_coords = merge_duplicate_coords
        self.num_instances = num_instances
        self.loss_direction_mode = loss_direction_mode
        self.edge_type_mode = edge_type_mode
        self.prediction_mode = prediction_mode
        
        if loss_direction_mode not in ['both', 'forward']:
            raise ValueError(f"loss_direction_mode must be one of ['both', 'forward'], got {loss_direction_mode}")
        
        if edge_type_mode not in ['single', 'triple']:
            raise ValueError(f"edge_type_mode must be one of ['single', 'triple'], got {edge_type_mode}")
        
        args = Args(
            emb_depth=6, 
            net_units=emsize, 
            net_act_fn=torch.nn.SiLU(), 
            emb_agg_fn=global_mean_pool,  
            par_depth=3
        )
        
        if use_instance_hypergraph:
            if self.num_instances is None:
                raise ValueError("num_instances must be provided when use_instance_hypergraph is True.")
            self.net = Net(args, use_instance_hypergraph=True, num_instances=self.num_instances, num_heads=8, prediction_mode=self.prediction_mode, use_residual_norm=use_residual_norm)
            # No separate edge predictor needed - integrated into the model
        elif use_shared_basis_film:
            if self.num_instances is None:
                raise ValueError("num_instances must be provided when use_shared_basis_film is True.")
            self.net = Net(args, use_shared_basis_film=True, num_relations=3, num_bases=4, num_instances=self.num_instances, num_heads=8)
            self.edge_predictor = nn.Sequential(
                nn.Linear(emsize, emsize),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(emsize, emsize//2),
                nn.GELU(),
                nn.Linear(emsize//2, 1)
            )
        elif use_unified_encoding:
            self.net = Net(args, use_multi_rel_emb_net=True, num_relations=2, num_bases=4)
        else:
            self.net = Net(args, use_multi_rel_emb_net=False)
    
        
    def forward(self, x, y=None, candidate_info=None, gat_pooling=None, single_eval_pos=None):
        """
        Forward pass through the GNN encoder.
        
        Args:
            x: Tensor of shape (seq_len, batch, num_nodes, 2) containing node coordinates for TSP graphs
            y: Tensor of shape (seq_len, batch, num_nodes) containing tour indices (required for unified encoding)
            candidate_info: List of candidate information dictionaries from LKH3
            gat_pooling: Optional GAT pooling module for attention-based aggregation
            single_eval_pos: Position where evaluation starts (for SharedBasisFiLM mode)
        
        Returns:
            Dictionary containing:
                - node_embeddings: Tensor of shape (seq_len, batch, emsize)
                - edge_info: Complete edge information including labels and preprocessed data
        """
        if self.use_instance_hypergraph:
            return self._forward_instance_hypergraph(x, y, candidate_info, single_eval_pos=single_eval_pos)
            
        if self.use_shared_basis_film:
            return self._forward_shared_basis_film(x, y, candidate_info, merge_duplicate_coords=True, single_eval_pos=single_eval_pos)
            
        if self.use_unified_encoding and y is None:
            raise ValueError("Tour information (y) is required when use_unified_encoding=True")
            
        seq_len, batch_size, max_num_nodes, _ = x.shape
        
        all_edge_indices = []
        all_edge_attrs = []
        all_relation_indices = []
        all_batch_ids = []
        all_position_ids = []  
        
        cumulative_nodes = 0
        node_offset_map = {}  
        edge_counts = []  
        all_x_valid = []
        eval_edge_infos = []
        
        for pos in range(seq_len):
            for b in range(batch_size):
                coords = x[pos, b]
                valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
                valid_indices = torch.where(valid_mask)[0]
                num_valid_nodes = len(valid_indices)
                
                if num_valid_nodes == 0:
                    continue
                
                valid_coords = coords[valid_mask]
                
                for j, valid_idx in enumerate(valid_indices):
                    node_offset_map[(pos, b, valid_idx.item())] = cumulative_nodes + j
                
                graph_edges, graph_edge_attrs = self._build_graph_edges(
                    valid_coords, num_valid_nodes, candidate_info, pos, b, batch_size, x.device
                )
                
                if self.use_unified_encoding and y is not None:
                    tour_edges, tour_edge_attrs = self._build_tour_edges(
                        y[pos, b], valid_indices, num_valid_nodes, valid_coords, x.device
                    )
                    
                    if graph_edges.size(1) > 0 and tour_edges.size(1) > 0:
                        edge_index = torch.cat([graph_edges, tour_edges], dim=1)
                        edge_attr = torch.cat([graph_edge_attrs, tour_edge_attrs], dim=0)
                        relation_index = torch.cat([
                            torch.zeros(graph_edges.size(1), dtype=torch.long, device=x.device),
                            torch.ones(tour_edges.size(1), dtype=torch.long, device=x.device)
                        ])
                    elif graph_edges.size(1) > 0:
                        edge_index = graph_edges
                        edge_attr = graph_edge_attrs
                        relation_index = torch.zeros(graph_edges.size(1), dtype=torch.long, device=x.device)
                    elif tour_edges.size(1) > 0:
                        edge_index = tour_edges
                        edge_attr = tour_edge_attrs
                        relation_index = torch.ones(tour_edges.size(1), dtype=torch.long, device=x.device)
                    else:
                        edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
                        edge_attr = torch.empty((0, 1), dtype=torch.float32, device=x.device)
                        relation_index = torch.empty((0,), dtype=torch.long, device=x.device)
                else:
                    edge_index = graph_edges
                    edge_attr = graph_edge_attrs
                    relation_index = torch.zeros(graph_edges.size(1), dtype=torch.long, device=x.device) if graph_edges.size(1) > 0 else torch.empty((0,), dtype=torch.long, device=x.device)

                edge_index += cumulative_nodes
                
                all_edge_indices.append(edge_index)
                all_edge_attrs.append(edge_attr)
                all_relation_indices.append(relation_index)
                
                all_batch_ids.append(torch.full((num_valid_nodes,), b, dtype=torch.long, device=x.device))
                all_position_ids.append(torch.full((num_valid_nodes,), pos, dtype=torch.long, device=x.device))
                
                all_x_valid.append(valid_coords)
                
                is_eval_instance = (single_eval_pos is None) or (pos >= single_eval_pos)
                if is_eval_instance:
                    local_node_offset_map = {}
                    for j, valid_idx in enumerate(valid_indices):
                        local_node_offset_map[(pos, b, valid_idx.item())] = cumulative_nodes + j
                    
                    local_edge_index = edge_index.clone()
                    if edge_index.size(1) > 0:
                        local_edge_index = edge_index - cumulative_nodes
                    
                    eval_edge_infos.append({
                        'pos': pos,
                        'batch': b,
                        'edge_index': local_edge_index,
                        'num_valid_nodes': num_valid_nodes,
                        'valid_indices': valid_indices,
                        'node_offset_map': local_node_offset_map
                    })
                
                cumulative_nodes += num_valid_nodes
                edge_counts.append(edge_index.size(1))  
        
        edge_index = torch.cat(all_edge_indices, dim=1) if all_edge_indices else torch.empty((2, 0), dtype=torch.long, device=x.device)
        edge_attr = torch.cat(all_edge_attrs, dim=0) if all_edge_attrs else torch.empty((0, 1), dtype=torch.float32, device=x.device)
        relation_index = torch.cat(all_relation_indices, dim=0) if all_relation_indices else torch.empty((0,), dtype=torch.long, device=x.device)
        batch_tensor = torch.cat(all_batch_ids, dim=0) if all_batch_ids else torch.empty((0,), dtype=torch.long, device=x.device)
        position_tensor = torch.cat(all_position_ids, dim=0) if all_position_ids else torch.empty((0,), dtype=torch.long, device=x.device)
        
        x_valid_flat = torch.cat(all_x_valid, dim=0) if all_x_valid else torch.empty((0, self.num_features), device=x.device)
        
        if self.use_unified_encoding:
            graph_emb, edge_emb = self.net.infer(
                x=x_valid_flat, 
                edge_index=edge_index, 
                edge_attr=edge_attr, 
                batch=batch_tensor,
                position=position_tensor,
                emb_net=lambda x, ei, ea: self.net.emb_net(x, ei, ea, relation_index),
                gat_pooling=gat_pooling
            )
        else:
            graph_emb, edge_emb = self.net(
                x=x_valid_flat, 
                edge_index=edge_index, 
                edge_attr=edge_attr, 
                batch=batch_tensor,
                position=position_tensor,
                gat_pooling=gat_pooling
            )
            
        edge_info = {
            'embeddings': edge_emb,
            'indices': edge_index,
            'batch': batch_tensor,
            'position': position_tensor,
            'node_offset_map': node_offset_map,
            'edge_counts': edge_counts,
            'eval_infos': eval_edge_infos
        }
        
        final_output = {
            'node_embeddings': graph_emb,
            'edge_info': edge_info,
        }
        
        return final_output

    def _build_graph_edges(self, valid_coords, num_valid_nodes, candidate_info, pos, b, batch_size, device):
        def _create_complete_graph(num_nodes, device):
            if num_nodes <= 1:
                return torch.empty((2, 0), dtype=torch.long, device=device)
            
            nodes = torch.arange(num_nodes, device=device)
            src_nodes = nodes.unsqueeze(1).expand(-1, num_nodes).flatten()
            dst_nodes = nodes.unsqueeze(0).expand(num_nodes, -1).flatten()
            
            non_self_mask = src_nodes != dst_nodes
            edge_index = torch.stack([src_nodes[non_self_mask], dst_nodes[non_self_mask]], dim=0)
            return edge_index

        def _vectorized_knn_lkh3_edges_optimized(valid_coords, num_valid_nodes, candidate_info, pos, b, batch_size, device, max_candidates):
            if num_valid_nodes <= 1:
                return torch.empty((2, 0), dtype=torch.long, device=device)
            
            target_edges_per_node = min(max_candidates, num_valid_nodes - 1)
            
            dist_matrix = torch.cdist(valid_coords, valid_coords, p=2)
            k_neighbors = min(target_edges_per_node + 1, num_valid_nodes)  # +1 for self
            _, nearest_indices = torch.topk(-dist_matrix, k_neighbors, dim=1)
            
            node_candidates = {}
            if candidate_info is not None:
                candidate_idx = pos * batch_size + b
                if candidate_idx < len(candidate_info) and candidate_info[candidate_idx] is not None:
                    cand_info = candidate_info[candidate_idx]
                    
                    for node_id, candidates in cand_info['candidates'].items():
                        src_node = node_id - 1
                        if src_node < num_valid_nodes:
                            valid_candidates = []
                            for neighbor_id, alpha_value in candidates:
                                dst_node = neighbor_id - 1
                                if dst_node < num_valid_nodes and dst_node != src_node:
                                    valid_candidates.append((dst_node, alpha_value))
                            if valid_candidates:
                                node_candidates[src_node] = valid_candidates
            
            edge_pairs_set = set()
            
            for node in range(num_valid_nodes):
                lkh3_candidates = node_candidates.get(node, [])
                lkh3_candidates.sort(key=lambda x: x[1])  
                
                added_for_node = 0
                
                for neighbor, alpha in lkh3_candidates:
                    if added_for_node >= target_edges_per_node:
                        break
                    if neighbor != node:
                        edge_pair = (min(node, neighbor), max(node, neighbor))
                        if edge_pair not in edge_pairs_set:
                            edge_pairs_set.add(edge_pair)
                        added_for_node += 1
                
                if added_for_node < target_edges_per_node:
                    neighbors = nearest_indices[node]
                    for neighbor_tensor in neighbors:
                        if added_for_node >= target_edges_per_node:
                            break
                        neighbor_idx = neighbor_tensor.item()
                        if neighbor_idx != node:
                            edge_pair = (min(node, neighbor_idx), max(node, neighbor_idx))
                            if edge_pair not in edge_pairs_set:
                                edge_pairs_set.add(edge_pair)
                            added_for_node += 1
            
            if edge_pairs_set:
                edge_pairs = list(edge_pairs_set)
                min_nodes, max_nodes = zip(*edge_pairs)
                
                src_nodes = list(min_nodes) + list(max_nodes)
                dst_nodes = list(max_nodes) + list(min_nodes)
                
                edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long, device=device)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            
            return edge_index

        # Use optimized vectorized implementation
        if candidate_info is not None or num_valid_nodes > 1:
            edge_index = _vectorized_knn_lkh3_edges_optimized(
                valid_coords, num_valid_nodes, candidate_info, pos, b, batch_size, device, self.max_candidates
            )
        else:
            edge_index = _create_complete_graph(num_valid_nodes, device)
        
        # Fallback to complete graph if no edges generated
        if edge_index.size(1) == 0 and num_valid_nodes > 1:
            edge_index = _create_complete_graph(num_valid_nodes, device)
        
        # OPTIMIZATION: Vectorized edge attribute computation
        if edge_index.size(1) > 0:
            rows, cols = edge_index
            edge_attr = torch.norm(valid_coords[rows] - valid_coords[cols], dim=1).unsqueeze(1)
        else:
            edge_attr = torch.empty((0, 1), dtype=torch.float32, device=device)
        
        return edge_index, edge_attr

    def _build_tour_edges(self, tour, valid_indices, num_valid_nodes, valid_coords, device):
        """
        Build bidirectional tour edges from Hamiltonian cycle.
        """
        valid_tour_mask = (tour != -1)
        if not valid_tour_mask.any() or num_valid_nodes < 2:
            return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0, 1), dtype=torch.float32, device=device)
        
        valid_tour = tour[valid_tour_mask]
        tour_length = len(valid_tour)
        
        if tour_length < 2:
            return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0, 1), dtype=torch.float32, device=device)
        
        max_node_idx = max(valid_indices.max().item() + 1, valid_tour.max().item() + 1)
        node_mapping = torch.full((max_node_idx,), -1, dtype=torch.long, device=device)
        node_mapping[valid_indices] = torch.arange(num_valid_nodes, device=device)
        
        curr_nodes = valid_tour
        next_nodes = torch.cat([valid_tour[1:], valid_tour[0:1]], dim=0)
        
        valid_curr_mask = curr_nodes < max_node_idx
        valid_next_mask = next_nodes < max_node_idx
        
        curr_mapped = torch.full_like(curr_nodes, -1)
        next_mapped = torch.full_like(next_nodes, -1)
        
        curr_mapped[valid_curr_mask] = node_mapping[curr_nodes[valid_curr_mask]]
        next_mapped[valid_next_mask] = node_mapping[next_nodes[valid_next_mask]]
        
        valid_edge_mask = (curr_mapped >= 0) & (next_mapped >= 0)
        if not valid_edge_mask.any():
            return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0, 1), dtype=torch.float32, device=device)
        
        curr_valid = curr_mapped[valid_edge_mask]
        next_valid = next_mapped[valid_edge_mask]
        
        forward_edges = torch.stack([curr_valid, next_valid], dim=0)
        backward_edges = torch.stack([next_valid, curr_valid], dim=0)
        edge_index = torch.cat([forward_edges, backward_edges], dim=1)
        
        if edge_index.size(1) > 0:
            rows, cols = edge_index
            edge_attr = torch.norm(valid_coords[rows] - valid_coords[cols], dim=1).unsqueeze(1)
        else:
            edge_attr = torch.empty((0, 1), dtype=torch.float32, device=device)
        
        return edge_index, edge_attr

    def _forward_shared_basis_film(self, x, y, candidate_info, merge_duplicate_coords=True, single_eval_pos=None):
        """
        SharedBasisFiLM forward pass with coordinate merging across instances.
        Fixed: Now merges coordinates within each batch independently, not globally.
        """
        seq_len, batch_size, max_num_nodes, _ = x.shape
        
        if merge_duplicate_coords:
            
            # Store results for all batches
            all_merged_coords = []
            all_node_offset_maps = {}
            all_eval_edge_infos = []
            all_eval_predictions = []
            all_eval_edge_counts = []
            
            eval_start_pos = single_eval_pos or seq_len - 1
            total_nodes_all_batches = 0
            total_merged_all_batches = 0
            
            for batch_idx in range(batch_size):
                
                # Process each batch independently
                unique_coords_batch = []
                coord_to_global_idx_batch = {}
                global_to_originals_batch = {}
                instance_node_mappings_batch = {}
                instance_center_nodes_batch = []
                
                inst_id = 0
                total_nodes_batch = 0
                merged_count_batch = 0
                
                # Step 1: Build coordinate mapping for this batch
                for pos in range(seq_len):
                    coords = x[pos, batch_idx]
                    valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
                    valid_indices = torch.where(valid_mask)[0]
                    num_valid_nodes = len(valid_indices)
                    
                    if num_valid_nodes == 0:
                        continue
                    
                    valid_coords = coords[valid_mask]
                    instance_mapping = {}
                    # Use 1e8 for quantization to achieve 1e-8 precision threshold for coordinate merging
                    coords_quantized = (valid_coords * 1e8).round().long()
                    total_nodes_batch += num_valid_nodes
                        
                    for local_idx in range(num_valid_nodes):
                        coord_key = tuple(coords_quantized[local_idx].cpu().tolist())
                            
                        if coord_key not in coord_to_global_idx_batch:
                            global_idx = len(unique_coords_batch)
                            unique_coords_batch.append(valid_coords[local_idx])
                            coord_to_global_idx_batch[coord_key] = global_idx
                            global_to_originals_batch[global_idx] = []
                        else:
                            global_idx = coord_to_global_idx_batch[coord_key]
                            merged_count_batch += 1
                            
                        orig_node_idx = valid_indices[local_idx].item()
                        all_node_offset_maps[(pos, batch_idx, orig_node_idx)] = global_idx
                        global_to_originals_batch[global_idx].append((pos, batch_idx, orig_node_idx))
                        instance_mapping[local_idx] = global_idx
                    
                    instance_node_mappings_batch[inst_id] = instance_mapping
                    inst_id += 1
                
                # Step 2: Add instance center nodes
                center_coords_list_batch = []
                inst_id = 0
                for pos in range(seq_len):
                    coords = x[pos, batch_idx]
                    valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
                    num_valid_nodes = valid_mask.sum().item()
                    
                    if num_valid_nodes == 0:
                        continue
                    
                    valid_coords = coords[valid_mask]
                    center_coord = valid_coords.mean(dim=0)
                    center_coords_list_batch.append(center_coord)
                    
                    center_key = tuple((center_coord * 1e6).round().long().cpu().tolist())
                    if center_key not in coord_to_global_idx_batch:
                        center_global_idx = len(unique_coords_batch) + len(center_coords_list_batch) - 1
                        coord_to_global_idx_batch[center_key] = center_global_idx
                    else:
                        center_global_idx = coord_to_global_idx_batch[center_key]
                    
                    instance_center_nodes_batch.append(center_global_idx)
                    inst_id += 1
                
                # Step 3: Create merged coordinates for this batch
                if len(unique_coords_batch) > 0:
                    merged_coords_batch = torch.stack(unique_coords_batch, dim=0)
                else:
                    merged_coords_batch = torch.empty((0, 2), device=x.device)
                
                if center_coords_list_batch:
                    center_coords_tensor_batch = torch.stack(center_coords_list_batch, dim=0)
                    merged_coords_batch = torch.cat([merged_coords_batch, center_coords_tensor_batch], dim=0)
                
                # Step 4: Build edges for this batch
                edge_list = []
                edge_attr_list = []
                type_indices_list = []
                inst_indices_list = []
                
                inst_id = 0
                for pos in range(seq_len):
                    coords = x[pos, batch_idx]
                    valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
                    valid_indices = torch.where(valid_mask)[0]
                    num_valid_nodes = len(valid_indices)
                    
                    if num_valid_nodes == 0:
                        continue
                    
                    valid_coords = coords[valid_mask]
                    instance_mapping = instance_node_mappings_batch[inst_id]
                    
                    # Build graph edges
                    local_graph_edges, graph_edge_attrs = self._build_graph_edges(
                        valid_coords, num_valid_nodes, candidate_info, pos, batch_idx, batch_size, x.device
                    )
                    
                    if local_graph_edges.size(1) > 0:
                        mapping_tensor = torch.zeros(max(instance_mapping.keys()) + 1, dtype=torch.long, device=x.device)
                        for local_idx, global_idx in instance_mapping.items():
                            mapping_tensor[local_idx] = global_idx
                        
                        local_indices = local_graph_edges.view(-1)
                        global_indices = mapping_tensor[local_indices].view(2, -1)
                        
                        edge_list.append(global_indices)
                        edge_attr_list.append(graph_edge_attrs)
                        type_indices_list.append(torch.zeros(global_indices.size(1), device=x.device, dtype=torch.long))
                        inst_indices_list.append(torch.full((global_indices.size(1),), inst_id, device=x.device, dtype=torch.long))
                    
                    # Build tour edges if y is provided
                    if y is not None:
                        local_tour_edges, tour_edge_attrs = self._build_tour_edges(
                            y[pos, batch_idx], valid_indices, num_valid_nodes, valid_coords, x.device
                        )
                        
                        if local_tour_edges.size(1) > 0:
                            local_indices = local_tour_edges.view(-1)
                            global_indices = mapping_tensor[local_indices].view(2, -1)
                            
                            edge_list.append(global_indices)
                            edge_attr_list.append(tour_edge_attrs)
                            type_indices_list.append(torch.ones(global_indices.size(1), device=x.device, dtype=torch.long))
                            inst_indices_list.append(torch.full((global_indices.size(1),), inst_id, device=x.device, dtype=torch.long))
                    
                    # Add center edges
                    center_global_idx = instance_center_nodes_batch[inst_id]
                    if len(instance_mapping) > 0:
                        node_indices = torch.tensor(list(instance_mapping.values()), device=x.device, dtype=torch.long)
                        center_indices = torch.full_like(node_indices, center_global_idx)
                        
                        forward_edges = torch.stack([node_indices, center_indices], dim=0)
                        backward_edges = torch.stack([center_indices, node_indices], dim=0)
                        center_edges = torch.cat([forward_edges, backward_edges], dim=1)
                        
                        rows, cols = center_edges[0], center_edges[1]
                        center_edge_attrs = torch.norm(
                            merged_coords_batch[rows] - merged_coords_batch[cols], dim=1
                        ).unsqueeze(1)
                        
                        edge_list.append(center_edges)
                        edge_attr_list.append(center_edge_attrs)
                        type_indices_list.append(torch.full((center_edges.size(1),), 2, device=x.device, dtype=torch.long))
                        inst_indices_list.append(torch.full((center_edges.size(1),), inst_id, device=x.device, dtype=torch.long))
                    
                    inst_id += 1
                
                # Step 5: Merge edges for this batch
                if edge_list:
                    merged_edges_batch = torch.cat(edge_list, dim=1)
                    merged_edge_attrs_batch = torch.cat(edge_attr_list, dim=0)
                    merged_type_indices_batch = torch.cat(type_indices_list, dim=0)
                    merged_inst_indices_batch = torch.cat(inst_indices_list, dim=0)
                else:
                    merged_edges_batch = torch.empty((2, 0), dtype=torch.long, device=x.device)
                    merged_edge_attrs_batch = torch.empty((0, 1), device=x.device)
                    merged_type_indices_batch = torch.empty((0,), dtype=torch.long, device=x.device)
                    merged_inst_indices_batch = torch.empty((0,), dtype=torch.long, device=x.device)
                
                # STEP 2: GNN推理阶段（每个batch）
                # Step 6: Run GNN inference for this batch
                node_embs_batch, edge_embs_batch = self.net.infer(
                    x=merged_coords_batch,
                    edge_index=merged_edges_batch,
                    edge_attr=merged_edge_attrs_batch,
                    batch=torch.zeros(merged_coords_batch.size(0), device=x.device, dtype=torch.long),
                    position=torch.zeros(merged_coords_batch.size(0), device=x.device, dtype=torch.long),
                    emb_net=self.net.emb_net,
                    use_shared_basis_film=True,
                    type_index=merged_type_indices_batch,
                    inst_index=merged_inst_indices_batch,
                    instance_nodes=instance_center_nodes_batch
                )
                
                # STEP 3: 后处理阶段（每个batch）
                
                # Step 7: Process evaluation instances for this batch
                inst_id = 0
                for pos in range(seq_len):
                    coords = x[pos, batch_idx]
                    valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
                    num_valid_nodes = valid_mask.sum().item()
                    
                    if num_valid_nodes == 0:
                        continue
                    
                    if pos < eval_start_pos:
                        inst_id += 1
                        continue
                    
                    # Get edges for this instance
                    edge_mask = (merged_type_indices_batch == 0) & (merged_inst_indices_batch == inst_id)
                    instance_edges_global = merged_edges_batch[:, edge_mask]
                    instance_edge_embs = edge_embs_batch[edge_mask]
                    
                    # Apply forward filtering if needed
                    if self.loss_direction_mode == 'forward':
                        forward_edge_mask = torch.ones(instance_edges_global.size(1), dtype=torch.bool, device=x.device)
                        
                        for edge_idx in range(instance_edges_global.size(1)):
                            u_global, v_global = instance_edges_global[0, edge_idx].item(), instance_edges_global[1, edge_idx].item()
                            
                            u_originals = []
                            v_originals = []
                            
                            if u_global in global_to_originals_batch:
                                for orig_pos, orig_batch, orig_node in global_to_originals_batch[u_global]:
                                    if orig_pos == pos and orig_batch == batch_idx:
                                        u_originals.append(orig_node)
                            
                            if v_global in global_to_originals_batch:
                                for orig_pos, orig_batch, orig_node in global_to_originals_batch[v_global]:
                                    if orig_pos == pos and orig_batch == batch_idx:
                                        v_originals.append(orig_node)
                            
                            keep_edge = False
                            for u_orig in u_originals:
                                for v_orig in v_originals:
                                    if u_orig < v_orig:
                                        keep_edge = True
                                        break
                                if keep_edge:
                                    break
                            
                            forward_edge_mask[edge_idx] = keep_edge
                        
                        instance_edges_global = instance_edges_global[:, forward_edge_mask]
                        instance_edge_embs = instance_edge_embs[forward_edge_mask]
                    
                    # Predict edges for this instance
                    edge_preds = self.edge_predictor(instance_edge_embs).squeeze(-1) if instance_edge_embs.size(0) > 0 else torch.empty(0, device=x.device)
                    
                    all_eval_predictions.append(edge_preds)
                    all_eval_edge_counts.append(instance_edges_global.size(1))
                    
                    # Create eval_edge_info
                    original_valid_indices = torch.where(valid_mask)[0]
                    all_eval_edge_infos.append({
                        'pos': pos,
                        'batch': batch_idx,
                        'edge_index': instance_edges_global,
                        'num_valid_nodes': num_valid_nodes,
                        'valid_indices': original_valid_indices,
                        'node_offset_map': {},
                        'global_to_originals': global_to_originals_batch,
                        'instance_mapping': instance_node_mappings_batch.get(inst_id, {}),
                        'is_shared_basis_film': True
                    })
                    
                    inst_id += 1
                
                total_nodes_all_batches += total_nodes_batch
                total_merged_all_batches += merged_count_batch
            
            # STEP 3: 最终后处理阶段
            
            # Create final output tensors
            max_edges = max(all_eval_edge_counts) if all_eval_edge_counts else 1
            seq_eval_len = seq_len - eval_start_pos
            edge_values_padded = torch.zeros(seq_eval_len, batch_size, max_edges, device=x.device)
            
            if all_eval_predictions:
                pred_idx = 0
                for pos in range(eval_start_pos, seq_len):
                    for batch_idx in range(batch_size):
                        for eval_info in all_eval_edge_infos:
                            if eval_info['pos'] == pos and eval_info['batch'] == batch_idx:
                                if pred_idx < len(all_eval_predictions):
                                    pred_size = all_eval_predictions[pred_idx].size(0)
                                    edge_values_padded[pos - eval_start_pos, batch_idx, :pred_size] = all_eval_predictions[pred_idx]
                                    pred_idx += 1
                                break
            
            edge_info = {
                'embeddings': None,
                'indices': None,
                'batch': None,
                'position': None,
                'node_offset_map': all_node_offset_maps,
                'edge_counts': all_eval_edge_counts,
                'eval_infos': all_eval_edge_infos
            }
            
            final_output = {
                'node_embeddings': torch.zeros(seq_len, batch_size, self.emsize, device=x.device),
                'edge_info': edge_info,
                'direct_predictions': True,
                'edge_predictions': edge_values_padded
            }
            
            return final_output
        
        else:
            # No merging case - simplified version
            return self._forward_shared_basis_film_no_merge(x, y, candidate_info, single_eval_pos)
    
    def _forward_shared_basis_film_no_merge(self, x, y, candidate_info, single_eval_pos=None):
        """
        SharedBasisFiLM without coordinate merging.
        Supports both triple edge types (0=graph, 1=tour, 2=center) and single edge type with features.
        Each instance is processed independently.
        """
        seq_len, batch_size, max_num_nodes, _ = x.shape
        eval_start_pos = single_eval_pos or seq_len - 1
        
        all_eval_predictions = []
        all_eval_edge_counts = []
        all_eval_edge_infos = []
        node_offset_map = {}
        
        for batch_idx in range(batch_size):
            batch_predictions = []
            batch_edge_counts = []
            batch_eval_infos = []
            
            for pos in range(seq_len):
                coords = x[pos, batch_idx]
                valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
                valid_indices = torch.where(valid_mask)[0]
                num_valid_nodes = len(valid_indices)
                
                if num_valid_nodes == 0:
                    continue
                
                valid_coords = coords[valid_mask]
                
                # Create node offset mapping (local indices for this instance)
                for j, valid_idx in enumerate(valid_indices):
                    node_offset_map[(pos, batch_idx, valid_idx.item())] = j
                
                if pos < eval_start_pos:
                    continue  # Skip training instances
                
                # Determine if this is a context instance
                is_context = pos < single_eval_pos if single_eval_pos is not None else False
                
                # Add instance center node for both modes
                center_coord = valid_coords.mean(dim=0)
                extended_coords = torch.cat([valid_coords, center_coord.unsqueeze(0)], dim=0)
                center_node_idx = num_valid_nodes  # Index of the center node
                instance_nodes = [center_node_idx]  # List containing center node index
                
                # Build graph edges for this instance
                graph_edges, graph_edge_attrs = self._build_graph_edges(
                    valid_coords, num_valid_nodes, candidate_info, pos, batch_idx, batch_size, x.device
                )
                
                if self.edge_type_mode == 'triple':
                    # Triple edge mode: separate edge types for graph, tour, and center
                
                    # Build tour edges if available
                    tour_edges = torch.empty((2, 0), dtype=torch.long, device=x.device)
                    tour_edge_attrs = torch.empty((0, 1), device=x.device)
                
                    if y is not None:
                        tour_edges, tour_edge_attrs = self._build_tour_edges(
                            y[pos, batch_idx], valid_indices, num_valid_nodes, valid_coords, x.device
                        )
                
                    # Build center edges (type 2) - connections between center node and all other nodes
                    center_edges = torch.empty((2, 0), dtype=torch.long, device=x.device)
                    center_edge_attrs = torch.empty((0, 1), device=x.device)
                    
                    if num_valid_nodes > 0:
                        # Create bidirectional edges between center and all nodes
                        node_indices = torch.arange(num_valid_nodes, device=x.device, dtype=torch.long)
                        center_indices = torch.full_like(node_indices, center_node_idx)
                        
                        forward_edges = torch.stack([node_indices, center_indices], dim=0)
                        backward_edges = torch.stack([center_indices, node_indices], dim=0)
                        center_edges = torch.cat([forward_edges, backward_edges], dim=1)
                        
                        # Calculate edge attributes for center edges
                        rows, cols = center_edges[0], center_edges[1]
                        center_edge_attrs = torch.norm(
                            extended_coords[rows] - extended_coords[cols], dim=1
                        ).unsqueeze(1)
                    
                    # Combine all edges with proper type distinction
                    edge_list = []
                    edge_attr_list = []
                    type_indices_list = []
                    
                    # Add graph edges (type 0)
                    if graph_edges.size(1) > 0:
                        edge_list.append(graph_edges)
                        edge_attr_list.append(graph_edge_attrs)
                        type_indices_list.append(torch.zeros(graph_edges.size(1), device=x.device, dtype=torch.long))
                    
                    # Add tour edges (type 1)
                    if tour_edges.size(1) > 0:
                        edge_list.append(tour_edges)
                        edge_attr_list.append(tour_edge_attrs)
                        type_indices_list.append(torch.ones(tour_edges.size(1), device=x.device, dtype=torch.long))
                    
                    # Add center edges (type 2)
                    if center_edges.size(1) > 0:
                        edge_list.append(center_edges)
                        edge_attr_list.append(center_edge_attrs)
                        type_indices_list.append(torch.full((center_edges.size(1),), 2, device=x.device, dtype=torch.long))
                    
                    # Combine all edges
                    if edge_list:
                        all_edges = torch.cat(edge_list, dim=1)
                        all_edge_attrs = torch.cat(edge_attr_list, dim=0)
                        # Use single edge type (type 0) for ALL edges including center edges
                        type_indices = torch.zeros(all_edges.size(1), device=x.device, dtype=torch.long)
                    else:
                        all_edges = torch.empty((2, 0), dtype=torch.long, device=x.device)
                        all_edge_attrs = torch.empty((0, 3), device=x.device)
                        type_indices = torch.empty((0,), dtype=torch.long, device=x.device)
                
                else:  # self.edge_type_mode == 'single'
                    # Single edge mode: combine all edges with additional features and center node connections
                    
                    # Build tour edges if available for feature extraction
                    tour_edges = torch.empty((2, 0), dtype=torch.long, device=x.device)
                    if y is not None and is_context:
                        tour_edges, _ = self._build_tour_edges(
                            y[pos, batch_idx], valid_indices, num_valid_nodes, valid_coords, x.device
                        )
                    
                    # Create tour edge set for solution labeling
                    tour_edge_set = set()
                    if tour_edges.size(1) > 0:
                        for i in range(tour_edges.size(1)):
                            u, v = tour_edges[0, i].item(), tour_edges[1, i].item()
                            tour_edge_set.add((min(u, v), max(u, v)))
                    
                    # Build center edges - connections between center node and all other nodes
                    center_edges = torch.empty((2, 0), dtype=torch.long, device=x.device)
                    center_edge_attrs = torch.empty((0, 3), device=x.device)
                    
                    if num_valid_nodes > 0:
                        # Create bidirectional edges between center and all nodes
                        node_indices = torch.arange(num_valid_nodes, device=x.device, dtype=torch.long)
                        center_indices = torch.full_like(node_indices, center_node_idx)
                        
                        forward_edges = torch.stack([node_indices, center_indices], dim=0)
                        backward_edges = torch.stack([center_indices, node_indices], dim=0)
                        center_edges = torch.cat([forward_edges, backward_edges], dim=1)
                        
                        # Calculate 3D edge attributes for center edges
                        num_center_edges = center_edges.size(1)
                        center_edge_attrs = torch.zeros((num_center_edges, 3), device=x.device)
                        
                        # Feature 0: distance
                        rows, cols = center_edges[0], center_edges[1]
                        distances = torch.norm(extended_coords[rows] - extended_coords[cols], dim=1)
                        center_edge_attrs[:, 0] = distances
                        
                        # Feature 1: is_solution (center edges are never solution edges)
                        center_edge_attrs[:, 1] = 0.0
                        
                        # Feature 2: is_context
                        center_edge_attrs[:, 2] = 1.0 if is_context else 0.0
                    
                    # Combine all edges with enhanced attributes
                    edge_list = []
                    edge_attr_list = []
                    
                    # Add graph edges with 3D features
                    if graph_edges.size(1) > 0:
                        num_graph_edges = graph_edges.size(1)
                        graph_edge_attrs_3d = torch.zeros((num_graph_edges, 3), device=x.device)
                        
                        # Feature 0: distance (same as before)
                        rows, cols = graph_edges[0], graph_edges[1]
                        distances = torch.norm(valid_coords[rows] - valid_coords[cols], dim=1)
                        graph_edge_attrs_3d[:, 0] = distances
                        
                        # Feature 1: is_solution (check against tour edges)
                        is_solution_flags = torch.zeros(num_graph_edges, device=x.device)
                        for i in range(num_graph_edges):
                            u, v = graph_edges[0, i].item(), graph_edges[1, i].item()
                            edge_key = (min(u, v), max(u, v))
                            if edge_key in tour_edge_set:
                                is_solution_flags[i] = 1.0
                        graph_edge_attrs_3d[:, 1] = is_solution_flags
                        
                        # Feature 2: is_context
                        graph_edge_attrs_3d[:, 2] = 1.0 if is_context else 0.0
                        
                        edge_list.append(graph_edges)
                        edge_attr_list.append(graph_edge_attrs_3d)
                    
                    # Add center edges with 3D features
                    if center_edges.size(1) > 0:
                        edge_list.append(center_edges)
                        edge_attr_list.append(center_edge_attrs)
                    
                    # Combine all edges with single edge type (type 0)
                    if edge_list:
                        all_edges = torch.cat(edge_list, dim=1)
                        all_edge_attrs = torch.cat(edge_attr_list, dim=0)
                        # Use single edge type (type 0) for ALL edges including center edges
                        type_indices = torch.zeros(all_edges.size(1), device=x.device, dtype=torch.long)
                    else:
                        all_edges = torch.empty((2, 0), dtype=torch.long, device=x.device)
                        all_edge_attrs = torch.empty((0, 3), device=x.device)
                        type_indices = torch.empty((0,), dtype=torch.long, device=x.device)
                
                # Apply forward filtering if needed
                if self.loss_direction_mode == 'forward' and all_edges.size(1) > 0:
                    if self.edge_type_mode == 'triple':
                        # Filter only graph and tour edges, keep center edges as they are
                        forward_mask = torch.ones(all_edges.size(1), dtype=torch.bool, device=x.device)
                        
                        # Apply filtering to non-center edges only
                        non_center_mask = type_indices != 2
                        if non_center_mask.any():
                            edge_forward_condition = all_edges[0] < all_edges[1]
                            forward_mask = forward_mask & (~non_center_mask | edge_forward_condition)
                        
                    all_edges = all_edges[:, forward_mask]
                    all_edge_attrs = all_edge_attrs[forward_mask]
                    type_indices = type_indices[forward_mask]
                else:
                    # For single edge mode, apply filtering but exclude center edges
                    forward_mask = torch.ones(all_edges.size(1), dtype=torch.bool, device=x.device)
                    
                    # Identify center edges (edges involving center_node_idx)
                    center_edge_mask = (all_edges[0] == center_node_idx) | (all_edges[1] == center_node_idx)
                    
                    # Apply forward filtering only to non-center edges
                    non_center_mask = ~center_edge_mask
                    if non_center_mask.any():
                        edge_forward_condition = all_edges[0] < all_edges[1]
                        forward_mask = forward_mask & (~non_center_mask | edge_forward_condition)
                    
                    all_edges = all_edges[:, forward_mask]
                    all_edge_attrs = all_edge_attrs[forward_mask]
                    type_indices = type_indices[forward_mask]
                
                num_edges = all_edges.size(1)
                total_nodes = len(extended_coords)  # Use actual number of nodes
                
                if num_edges > 0:
                    # Run GNN inference for this single instance
                    # No inst_index needed - all edges use same instance ID (0)
                    unified_inst_indices = torch.zeros(num_edges, dtype=torch.long, device=x.device)
                    
                    node_embs, edge_embs = self.net.infer(
                        x=extended_coords,
                        edge_index=all_edges,
                        edge_attr=all_edge_attrs,
                        batch=torch.zeros(total_nodes, device=x.device, dtype=torch.long),
                        position=torch.zeros(total_nodes, device=x.device, dtype=torch.long),
                        emb_net=self.net.emb_net,
                        use_shared_basis_film=True,
                        type_index=type_indices,
                        inst_index=unified_inst_indices,
                        instance_nodes=instance_nodes
                    )
                    
                    # Filter edges for prediction based on edge_type_mode
                    if self.edge_type_mode == 'triple':
                        # Filter out center edges for prediction (only predict graph edges type 0)
                        graph_edge_mask = type_indices == 0
                        if graph_edge_mask.any():
                            graph_edge_embs = edge_embs[graph_edge_mask]
                            edge_preds = self.edge_predictor(graph_edge_embs).squeeze(-1)
                            graph_edges_only = all_edges[:, graph_edge_mask]
                        else:
                            edge_preds = torch.empty(0, device=x.device)
                            graph_edges_only = torch.empty((2, 0), dtype=torch.long, device=x.device)
                    else:
                        # Single edge mode: filter out center edges for prediction (only predict graph edges)
                        center_edge_mask = (all_edges[0] == center_node_idx) | (all_edges[1] == center_node_idx)
                        graph_edge_mask = ~center_edge_mask
                        
                        if graph_edge_mask.any():
                            graph_edge_embs = edge_embs[graph_edge_mask]
                            edge_preds = self.edge_predictor(graph_edge_embs).squeeze(-1)
                            graph_edges_only = all_edges[:, graph_edge_mask]
                        else:
                            edge_preds = torch.empty(0, device=x.device)
                            graph_edges_only = torch.empty((2, 0), dtype=torch.long, device=x.device)
                    
                    batch_predictions.append(edge_preds)
                    batch_edge_counts.append(graph_edges_only.size(1))  # Count prediction edges
                    
                    # Create eval info
                    eval_info = {
                        'pos': pos,
                        'batch': batch_idx,
                        'edge_index': graph_edges_only,  # Prediction edges for evaluation
                        'num_valid_nodes': num_valid_nodes,
                        'valid_indices': valid_indices,
                        'node_offset_map': {},
                        'is_shared_basis_film': True,
                        'is_no_merge': True,
                        'edge_type_mode': self.edge_type_mode
                    }
                    
                    if self.edge_type_mode == 'triple':
                        eval_info.update({
                            'has_center_node': True,
                            'center_node_idx': center_node_idx,
                            'edge_features_dim': 1  # Standard distance features
                        })
                    else:
                        eval_info.update({
                            'has_center_node': True,  # Single mode also has center node
                            'center_node_idx': center_node_idx,
                            'edge_features_dim': 3  # [distance, is_solution, is_context]
                        })
                    
                    batch_eval_infos.append(eval_info)
                else:
                    # No edges for this instance
                    batch_predictions.append(torch.empty(0, device=x.device))
                    batch_edge_counts.append(0)
                    
                    eval_info = {
                        'pos': pos,
                        'batch': batch_idx,
                        'edge_index': torch.empty((2, 0), dtype=torch.long, device=x.device),
                        'num_valid_nodes': num_valid_nodes,
                        'valid_indices': valid_indices,
                        'node_offset_map': {},
                        'is_shared_basis_film': True,
                        'is_no_merge': True,
                        'edge_type_mode': self.edge_type_mode
                    }
                    
                    if self.edge_type_mode == 'triple':
                        eval_info.update({
                            'has_center_node': True,
                            'center_node_idx': num_valid_nodes if num_valid_nodes > 0 else 0,
                            'edge_features_dim': 1  # Standard distance features
                        })
                    else:
                        eval_info.update({
                            'has_center_node': True,  # Single mode also has center node
                            'center_node_idx': num_valid_nodes if num_valid_nodes > 0 else 0,
                            'edge_features_dim': 3  # [distance, is_solution, is_context]
                        })
                    
                    batch_eval_infos.append(eval_info)
            
            all_eval_predictions.extend(batch_predictions)
            all_eval_edge_counts.extend(batch_edge_counts)
            all_eval_edge_infos.extend(batch_eval_infos)
        
        # Create output tensors
        seq_eval_len = seq_len - eval_start_pos
        max_edges = max(all_eval_edge_counts) if all_eval_edge_counts else 1
        edge_values_padded = torch.zeros(seq_eval_len, batch_size, max_edges, device=x.device)
        
        if all_eval_predictions:
            pred_idx = 0
            for pos in range(eval_start_pos, seq_len):
                for batch_idx in range(batch_size):
                    if pred_idx < len(all_eval_predictions):
                        pred_size = all_eval_predictions[pred_idx].size(0)
                        if pred_size > 0:
                            edge_values_padded[pos - eval_start_pos, batch_idx, :pred_size] = all_eval_predictions[pred_idx]
                        pred_idx += 1
        
        edge_info = {
            'embeddings': None,
            'indices': None,
            'batch': None,
            'position': None,
            'node_offset_map': node_offset_map,
            'edge_counts': all_eval_edge_counts,
            'eval_infos': all_eval_edge_infos
        }
        
        return {
            'node_embeddings': torch.zeros(seq_len, batch_size, self.emsize, device=x.device),
            'edge_info': edge_info,
            'direct_predictions': True,
            'edge_predictions': edge_values_padded
        }

    def _forward_instance_hypergraph(self, x, y, candidate_info, single_eval_pos=None):
        """
        InstanceAwareHypergraphGNN forward pass.
        OPTIMIZED: Per-instance coordinate deduplication and efficient edge building.
        """
        seq_len, batch_size, max_num_nodes, _ = x.shape
        eval_start_pos = single_eval_pos or seq_len - 1
        
        # Build unified graph with all instances, center nodes, and coordinate merging
        if self.merge_duplicate_coords:
            time_a = time.time()
            
            # OPTIMIZATION: Per-instance coordinate deduplication
            all_unique_coords = []
            global_coord_offset = 0
            instance_mapping = {}
            node_offset_map = {}
            eval_infos = []
            global_to_originals = {}
            
            instance_id = 0
            total_nodes = 0
            merged_count = 0
            
            for pos in range(seq_len):
                for batch_idx in range(batch_size):
                    coords = x[pos, batch_idx]
                    valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
                    valid_indices = torch.where(valid_mask)[0]
                    num_valid_nodes = len(valid_indices)
                    
                    if num_valid_nodes == 0:
                        continue
                    
                    valid_coords = coords[valid_mask]
                    total_nodes += num_valid_nodes
                    
                    # OPTIMIZATION: Per-instance coordinate deduplication
                    coords_quantized = (valid_coords * 1e8).round().long()
                    coord_keys = [tuple(coord.cpu().tolist()) for coord in coords_quantized]
                    
                    # Local deduplication for this instance
                    local_unique_coords = []
                    local_coord_to_idx = {}
                    city_node_indices = []
                    
                    for local_idx, coord_key in enumerate(coord_keys):
                        if coord_key not in local_coord_to_idx:
                            local_coord_to_idx[coord_key] = len(local_unique_coords)
                            local_unique_coords.append(valid_coords[local_idx])
                            merged_count += 1
                        
                        global_idx = global_coord_offset + local_coord_to_idx[coord_key]
                        city_node_indices.append(global_idx)
                        
                        orig_node_idx = valid_indices[local_idx].item()
                        node_offset_map[(pos, batch_idx, orig_node_idx)] = global_idx
                        
                        if global_idx not in global_to_originals:
                            global_to_originals[global_idx] = []
                        global_to_originals[global_idx].append((pos, batch_idx, orig_node_idx))
                    
                    all_unique_coords.extend(local_unique_coords)
                    global_coord_offset += len(local_unique_coords)
                    
                    instance_mapping[instance_id] = city_node_indices.copy()
                    
                    if pos >= eval_start_pos:
                        eval_infos.append({
                            'pos': pos,
                            'batch': batch_idx,
                            'instance_id': instance_id,
                            'num_valid_nodes': num_valid_nodes,
                            'valid_indices': valid_indices,
                            'node_offset_map': {k: v for k, v in node_offset_map.items() 
                                              if k[0] == pos and k[1] == batch_idx},
                            'city_node_indices': city_node_indices
                        })
                    
                    instance_id += 1
            
            # OPTIMIZATION: Add center nodes efficiently
            center_node_start_idx = len(all_unique_coords)
            instance_id = 0
            
            for pos in range(seq_len):
                for batch_idx in range(batch_size):
                    coords = x[pos, batch_idx]
                    valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
                    num_valid_nodes = len(valid_indices := torch.where(valid_mask)[0])
                    
                    if num_valid_nodes == 0:
                        continue
                    
                    valid_coords = coords[valid_mask]
                    center_coord = valid_coords.mean(dim=0)
                    center_node_idx = center_node_start_idx + instance_id
                    all_unique_coords.append(center_coord)
                    
                    instance_mapping[instance_id].append(center_node_idx)
                    
                    # Update eval info with center node
                    if pos >= eval_start_pos:
                        for eval_info in eval_infos:
                            if eval_info['instance_id'] == instance_id:
                                eval_info['center_node_idx'] = center_node_idx
                                break
                    
                    instance_id += 1
            
            # OPTIMIZATION: Efficient edge building with pre-allocation (全向量化重构)
            merged_coords = torch.stack(all_unique_coords, dim=0) if all_unique_coords else torch.empty((0, 2), device=x.device)

            # 1. 批量收集所有instance的边
            all_u, all_v, all_instance_id, all_edge_type = [], [], [], []
            for instance_id in range(len(instance_mapping)):
                city_node_indices = instance_mapping[instance_id][:-1]
                center_node_idx = instance_mapping[instance_id][-1]
                num_nodes = len(city_node_indices)
                if num_nodes == 0:
                    continue
                # 构建图边（KNN/candidate）
                valid_coords = merged_coords[city_node_indices]
                local_graph_edges, _ = self._build_graph_edges(
                    valid_coords, num_nodes, candidate_info, instance_id // batch_size, instance_id % batch_size, batch_size, x.device
                )
                if local_graph_edges.size(1) > 0:
                    # 映射到全局索引
                    city_indices_tensor = torch.tensor(city_node_indices, device=x.device, dtype=torch.long)
                    global_graph_edges = city_indices_tensor[local_graph_edges]
                    all_u.append(global_graph_edges[0])
                    all_v.append(global_graph_edges[1])
                    all_instance_id.append(torch.full((global_graph_edges.size(1),), instance_id, device=x.device, dtype=torch.long))
                    all_edge_type.append(torch.zeros(global_graph_edges.size(1), device=x.device, dtype=torch.long))
                # 构建中心边
                if num_nodes > 0:
                    city_indices = torch.tensor(city_node_indices, device=x.device, dtype=torch.long)
                    center_indices = torch.full_like(city_indices, center_node_idx)
                    all_u.append(city_indices)
                    all_v.append(center_indices)
                    all_instance_id.append(torch.full((city_indices.size(0),), instance_id, device=x.device, dtype=torch.long))
                    all_edge_type.append(torch.ones(city_indices.size(0), device=x.device, dtype=torch.long))
            if all_u:
                all_u = torch.cat(all_u)
                all_v = torch.cat(all_v)
                all_instance_id = torch.cat(all_instance_id)
                all_edge_type = torch.cat(all_edge_type)
            else:
                all_u = torch.empty(0, dtype=torch.long, device=x.device)
                all_v = torch.empty(0, dtype=torch.long, device=x.device)
                all_instance_id = torch.empty(0, dtype=torch.long, device=x.device)
                all_edge_type = torch.empty(0, dtype=torch.long, device=x.device)
            # 2. 全局有向边去重
            edge_pairs = torch.stack([all_u, all_v], dim=0)  # [2, E]，有向边
            unique_edges, inverse_indices = torch.unique(edge_pairs.t(), dim=0, return_inverse=True)
            merged_edges = unique_edges.t()  # [2, num_unique_edges]
            num_unique_edges = merged_edges.size(1)
            num_instances_total = len(instance_mapping)
            # 3. 批量计算边属性
            edge_attr = torch.norm(merged_coords[merged_edges[0]] - merged_coords[merged_edges[1]], dim=1, keepdim=True)
            # 4. instance到边、边到instance的映射
            # 对于每个instance，找到其所有边在unique_edges中的索引
            instance_to_edges = {}
            edge_to_instances = {}
            for idx, (inst_id, inv_idx) in enumerate(zip(all_instance_id.cpu().tolist(), inverse_indices.cpu().tolist())):
                if inst_id not in instance_to_edges:
                    instance_to_edges[inst_id] = set()
                instance_to_edges[inst_id].add(inv_idx)
                if inv_idx not in edge_to_instances:
                    edge_to_instances[inv_idx] = set()
                edge_to_instances[inv_idx].add(inst_id)
            for k in instance_to_edges:
                instance_to_edges[k] = list(instance_to_edges[k])
            for k in edge_to_instances:
                edge_to_instances[k] = list(edge_to_instances[k])
            merged_edge_attrs = edge_attr
            # --- s_uvi_matrix generation ---
            # s_uvi_matrix: [num_unique_edges, num_instances], 1 if edge is solution in instance, else 0
            s_uvi_matrix = torch.zeros((num_unique_edges, num_instances_total), dtype=torch.float32, device=x.device)
            # For each instance, mark solution edges
            for eval_info in eval_infos:
                instance_id = eval_info['instance_id']
                city_node_indices = eval_info['city_node_indices']
                valid_indices = eval_info['valid_indices']
                pos = eval_info['pos']
                batch = eval_info['batch']
                # Get tour for this instance
                if y is not None and valid_indices is not None and len(valid_indices) > 1:
                    tour = y[pos, batch, :len(valid_indices)]
                    valid_tour_mask = (tour != -1)
                    valid_tour = tour[valid_tour_mask]
                    if len(valid_tour) > 1:
                        # Map local node idx to global idx
                        local_to_global = {}
                        for local_idx, global_idx in enumerate(city_node_indices):
                            if local_idx < len(valid_indices):
                                local_to_global[local_idx] = global_idx
                        # Build tour edges (有向)
                        curr_nodes = valid_tour
                        next_nodes = torch.cat([valid_tour[1:], valid_tour[0:1]], dim=0)
                        for u_local, v_local in zip(curr_nodes.tolist(), next_nodes.tolist()):
                            if u_local in local_to_global and v_local in local_to_global:
                                u_global = local_to_global[u_local]
                                v_global = local_to_global[v_local]
                                # Find global edge idx
                                edge_tensor = torch.tensor([u_global, v_global], device=x.device)
                                # Find in unique_edges
                                match = (unique_edges == edge_tensor).all(dim=1)
                                idxs = torch.where(match)[0]
                                if len(idxs) > 0:
                                    s_uvi_matrix[idxs[0], instance_id] = 1.0
        
        else:
            # OPTIMIZATION: Without coordinate merging - streamlined for speed
            all_nodes = []
            edge_list = []
            edge_attr_list = []
            instance_mapping = {}
            node_offset_map = {}
            eval_infos = []
            
            # ENHANCEMENT: Edge tracking for no-merge case
            edge_to_instances = {}
            instance_to_edges = {}
            
            global_node_idx = 0
            instance_id = 0
            
            for pos in range(seq_len):
                for batch_idx in range(batch_size):
                    coords = x[pos, batch_idx]
                    valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
                    num_valid_nodes = len(valid_indices := torch.where(valid_mask)[0])
                    
                    if num_valid_nodes == 0:
                        continue
                    
                    valid_coords = coords[valid_mask]
                    
                    # Add nodes efficiently
                    city_node_indices = list(range(global_node_idx, global_node_idx + num_valid_nodes))
                    all_nodes.append(valid_coords)
                    
                    center_coord = valid_coords.mean(dim=0)
                    center_node_idx = global_node_idx + num_valid_nodes
                    all_nodes.append(center_coord.unsqueeze(0))
                    
                    instance_mapping[instance_id] = city_node_indices + [center_node_idx]
                    
                    # Initialize instance edge tracking
                    instance_to_edges[instance_id] = []
                    
                    # OPTIMIZATION: Fast node offset mapping
                    for j, valid_idx in enumerate(valid_indices):
                        node_offset_map[(pos, batch_idx, valid_idx.item())] = global_node_idx + j
                    
                    # OPTIMIZATION: Fast edge building with tracking
                    graph_edges, graph_distances = self._build_graph_edges(
                        valid_coords, num_valid_nodes, candidate_info, pos, batch_idx, batch_size, x.device
                    )
                    
                    if graph_edges.size(1) > 0:
                        global_graph_edges = graph_edges + global_node_idx
                        edge_list.append(global_graph_edges)
                        
                        # Track graph edges for this instance
                        for edge_idx in range(global_graph_edges.size(1)):
                            edge_u, edge_v = global_graph_edges[0, edge_idx].item(), global_graph_edges[1, edge_idx].item()
                            edge_key = (min(edge_u, edge_v), max(edge_u, edge_v))
                            edge_to_instances[edge_key] = [instance_id]
                            instance_to_edges[instance_id].append(edge_key)
                        
                        # OPTIMIZATION: Vectorized edge attributes
                        num_graph_edges = graph_edges.size(1)
                        graph_edge_attrs = torch.zeros((num_graph_edges, 2), device=x.device)
                        graph_edge_attrs[:, 0] = graph_distances.squeeze(-1)
                        graph_edge_attrs[:, 1] = 0.0
                        edge_attr_list.append(graph_edge_attrs)
                    
                    # OPTIMIZATION: Fast center edges with tracking
                    if num_valid_nodes > 0:
                        city_indices = torch.arange(global_node_idx, global_node_idx + num_valid_nodes, device=x.device)
                        center_indices = torch.full_like(city_indices, center_node_idx)
                        
                        forward_edges = torch.stack([city_indices, center_indices], dim=0)
                        backward_edges = torch.stack([center_indices, city_indices], dim=0)
                        center_edges = torch.cat([forward_edges, backward_edges], dim=1)
                        
                        edge_list.append(center_edges)
                        
                        # Track center edges for this instance
                        for edge_idx in range(center_edges.size(1)):
                            edge_u, edge_v = center_edges[0, edge_idx].item(), center_edges[1, edge_idx].item()
                            edge_key = (min(edge_u, edge_v), max(edge_u, edge_v))
                            edge_to_instances[edge_key] = [instance_id]
                            instance_to_edges[instance_id].append(edge_key)
                        
                        num_center_edges = center_edges.size(1)
                        center_edge_attrs = torch.zeros((num_center_edges, 2), device=x.device)
                        center_edge_attrs[:, 0] = 0.0
                        center_edge_attrs[:, 1] = 1.0
                        edge_attr_list.append(center_edge_attrs)
                    
                    # Store evaluation info
                    if pos >= eval_start_pos:
                        eval_infos.append({
                            'pos': pos,
                            'batch': batch_idx,
                            'instance_id': instance_id,
                            'num_valid_nodes': num_valid_nodes,
                            'valid_indices': valid_indices,
                            'node_offset_map': {k: v for k, v in node_offset_map.items() 
                                              if k[0] == pos and k[1] == batch_idx},
                            'city_node_indices': city_node_indices,
                            'center_node_idx': center_node_idx
                        })
                    
                    global_node_idx += num_valid_nodes + 1
                    instance_id += 1
            
            merged_coords = torch.cat(all_nodes, dim=0) if all_nodes else torch.empty((0, 2), device=x.device)
        
            # Single concatenation at the end
            if edge_list:
                merged_edges = torch.cat(edge_list, dim=1)
                merged_edge_attrs = torch.cat(edge_attr_list, dim=0)
            else:
                merged_edges = torch.empty((2, 0), dtype=torch.long, device=x.device)
                merged_edge_attrs = torch.empty((0, 2), device=x.device)
        
        time_b = time.time()
        if merged_coords.size(0) > 0 and merged_edges.size(1) > 0:
            node_embs, edge_embs, z_instances, all_predictions = self.net.infer(
                x=merged_coords,
                edge_index=merged_edges,
                edge_attr=merged_edge_attrs,
                batch=torch.zeros(merged_coords.size(0), device=x.device, dtype=torch.long),
                emb_net=self.net.emb_net,
                use_instance_hypergraph=True,
                instance_mapping=instance_mapping,
                s_uvi_matrix=s_uvi_matrix,
                single_eval_pos=single_eval_pos,
                instance_to_edges=instance_to_edges,
                edge_to_instances=edge_to_instances
            )
        else:
            node_embs = torch.zeros(seq_len, batch_size, self.emsize, device=x.device)
            edge_embs = None
            z_instances = {}
            all_predictions = torch.empty(0, device=x.device)
        
        # OPTIMIZATION: Fast edge filtering and prediction generation
        predictions_list = []
        edge_counts = []
        
        if len(eval_infos) > 0 and merged_edges.size(1) > 0:
            # OPTIMIZATION: Vectorized edge filtering
            src_nodes = merged_edges[0]
            dst_nodes = merged_edges[1]
            
            for eval_info in eval_infos:
                instance_id = eval_info['instance_id']
                city_node_indices = eval_info['city_node_indices']
                
                if len(city_node_indices) > 0:
                    # OPTIMIZATION: Efficient edge masking using isin
                    city_node_tensor = torch.tensor(city_node_indices, device=x.device, dtype=torch.long)
                    src_in_city = torch.isin(src_nodes, city_node_tensor)
                    dst_in_city = torch.isin(dst_nodes, city_node_tensor)
                    instance_edge_mask = src_in_city & dst_in_city
                    instance_edges = merged_edges[:, instance_edge_mask]
                    
                    # Fix: Extract predictions for this specific instance
                    if instance_edge_mask.any():
                        instance_preds = all_predictions[instance_edge_mask, instance_id]  # [num_edges_for_instance]
                    else:
                        instance_preds = torch.empty(0, device=x.device)
                    
                    predictions_list.append(instance_preds)
                    edge_counts.append(instance_edges.size(1))
                    eval_info['edge_index'] = instance_edges
                else:
                    predictions_list.append(torch.empty(0, device=x.device))
                    edge_counts.append(0)
                    eval_info['edge_index'] = torch.empty((2, 0), dtype=torch.long, device=x.device)
        else:
            for eval_info in eval_infos:
                predictions_list.append(torch.empty(0, device=x.device))
                edge_counts.append(0)
                eval_info['edge_index'] = torch.empty((2, 0), dtype=torch.long, device=x.device)
        
        # OPTIMIZATION: Fast output tensor creation
        seq_eval_len = seq_len - eval_start_pos
        max_edges = max(edge_counts) if edge_counts else 1
        edge_values_padded = torch.zeros(seq_eval_len, batch_size, max_edges, device=x.device)
        
        # OPTIMIZATION: Vectorized prediction filling
        pred_idx = 0
        for pos in range(eval_start_pos, seq_len):
            for batch_idx in range(batch_size):
                if pred_idx < len(predictions_list):
                    pred_size = predictions_list[pred_idx].size(0)
                    if pred_size > 0:
                        edge_values_padded[pos - eval_start_pos, batch_idx, :pred_size] = predictions_list[pred_idx]
                    pred_idx += 1
        
        edge_info = {
            'embeddings': edge_embs,
            'indices': merged_edges,
            'batch': None,
            'position': None,
            'node_offset_map': node_offset_map,
            'edge_counts': edge_counts,
            'eval_infos': eval_infos,
            'instance_mapping': instance_mapping,
            'z_instances': z_instances,
            'all_predictions': all_predictions,
            'coordinate_merging_enabled': self.merge_duplicate_coords,
            'global_to_originals': global_to_originals if self.merge_duplicate_coords else {},
            'edge_to_instances': edge_to_instances,  
            'instance_to_edges': instance_to_edges,  
            's_uvi_matrix': s_uvi_matrix  # [num_edges, num_instances]
        }
        
        final_output = {
            'node_embeddings': node_embs,
            'edge_info': edge_info,
            'direct_predictions': True,
            'edge_predictions': edge_values_padded
        }
        
        time_c = time.time()
        # print(f"Time taken: {time_b - time_a:.2f}s, {time_c - time_b:.2f}s")
        return final_output


class TSPTourEncoder(nn.Module):
    """
    TSP tour encoder using edge embeddings.
    """
    def __init__(self, num_features, emsize, max_nodes=100):
        """
        Initialize TSP tour encoder.
        """
        super().__init__()
        self.emsize = emsize
        self.max_nodes = max_nodes
        
    def forward(self, y, edge_emb=None, edge_index=None, batch=None, position=None, node_offset_map=None, gat_pooling=None):
        """
        Encode TSP tours using edge embeddings.
        """
        if len(y.shape) == 2:
            y = y.unsqueeze(-1)
            
        seq_len, batch_size, num_nodes = y.shape
        
        tour_embeddings = torch.zeros(seq_len, batch_size, self.emsize, device=y.device)
        
        if edge_emb is None or edge_index is None or edge_index.size(1) == 0:
            return tour_embeddings
        
        src, dst = edge_index
        
        edge_pairs = torch.stack([src, dst], dim=1)
        edge_pairs_flipped = torch.stack([dst, src], dim=1)
        
        all_edge_pairs = torch.cat([edge_pairs, edge_pairs_flipped], dim=0)
        all_edge_indices = torch.cat([torch.arange(len(src), device=y.device), 
                                     torch.arange(len(src), device=y.device)], dim=0)
        
        if len(src) > 0:
            edge_keys = all_edge_pairs[:, 0] * (src.max() + 1) + all_edge_pairs[:, 1]
            sorted_indices = torch.argsort(edge_keys)
            sorted_edge_keys = edge_keys[sorted_indices]
            sorted_edge_indices = all_edge_indices[sorted_indices]
        else:
            sorted_edge_keys = torch.empty(0, dtype=torch.long, device=y.device)
            sorted_edge_indices = torch.empty(0, dtype=torch.long, device=y.device)
        
        for pos in range(seq_len):
            for b in range(batch_size):
                tour = y[pos, b]
                
                valid_tour_mask = (tour != -1)
                if not valid_tour_mask.any():
                    continue
                
                valid_tour = tour[valid_tour_mask]
                tour_length = len(valid_tour)
                
                if tour_length < 2:
                    continue
                
                tour_edges = torch.stack([valid_tour[:-1], valid_tour[1:]], dim=0)
                final_edge = torch.tensor([[valid_tour[-1]], [valid_tour[0]]], device=y.device)
                tour_edges = torch.cat([tour_edges, final_edge], dim=1)
                
                tour_edge_keys = []
                valid_tour_edges = []
                
                with torch.no_grad():
                    tour_edges_cpu = tour_edges.t().cpu().numpy()
                    
                    for src_idx, dst_idx in tour_edges_cpu:
                        src_idx, dst_idx = int(src_idx), int(dst_idx)
                        
                        if (node_offset_map is not None and 
                            (pos, b, src_idx) in node_offset_map and 
                            (pos, b, dst_idx) in node_offset_map):
                            
                            global_src = node_offset_map[(pos, b, src_idx)]
                            global_dst = node_offset_map[(pos, b, dst_idx)]
                                
                            if len(src) > 0:
                                key = global_src * (src.max() + 1) + global_dst
                                tour_edge_keys.append(key)
                                valid_tour_edges.append((global_src, global_dst))
                
                if tour_edge_keys and len(sorted_edge_keys) > 0:
                    tour_keys_tensor = torch.tensor(tour_edge_keys, device=y.device)
                    
                    indices = torch.searchsorted(sorted_edge_keys, tour_keys_tensor)
                    
                    all_tour_edge_embs = []
                    for i, idx in enumerate(indices):
                        if idx < len(sorted_edge_keys) and sorted_edge_keys[idx] == tour_keys_tensor[i]:
                            edge_idx = sorted_edge_indices[idx]
                            all_tour_edge_embs.append(edge_emb[edge_idx])
                
                    if len(all_tour_edge_embs) > 0:
                        tour_edge_embs = torch.stack(all_tour_edge_embs, dim=0)
                        
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


def tsp_graph_encoder_generator(num_features, emsize, max_candidates=50, use_unified_encoding=False, use_shared_basis_film=False, use_instance_hypergraph=False, merge_duplicate_coords=True, num_instances=None, loss_direction_mode='both', edge_type_mode='triple', prediction_mode='dot_product', use_residual_norm=False):
    """
    Create TSP graph encoder.
    """
    return TSPGraphEncoder(
        num_features, emsize, max_candidates, use_unified_encoding, 
        use_shared_basis_film, use_instance_hypergraph, merge_duplicate_coords, num_instances=num_instances,
        loss_direction_mode=loss_direction_mode, edge_type_mode=edge_type_mode, prediction_mode=prediction_mode,
        use_residual_norm=use_residual_norm
    )

def tsp_tour_encoder_generator(num_features, emsize, max_nodes=100):
    """
    Create TSP tour encoder.
    """
    return TSPTourEncoder(num_features, emsize, max_nodes) 