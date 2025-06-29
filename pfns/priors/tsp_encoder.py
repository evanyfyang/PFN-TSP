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

class Args:
    """
    Configuration class for network initialization.
    """
    def __init__(self, emb_depth=3, net_units=128, net_act_fn=torch.nn.ReLU(), emb_agg_fn=global_mean_pool, device='cuda', par_depth=3):
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
    def __init__(self, num_features, emsize, max_candidates=5, use_unified_encoding=False, use_shared_basis_film=False, merge_duplicate_coords=True, num_instances=None, loss_direction_mode='both'):
        """
        Initialize TSP graph encoder.
        """
        super().__init__()
        self.num_features = num_features
        self.emsize = emsize
        self.max_candidates = max_candidates
        self.use_unified_encoding = use_unified_encoding
        self.use_shared_basis_film = use_shared_basis_film
        self.merge_duplicate_coords = merge_duplicate_coords
        self.num_instances = num_instances
        self.loss_direction_mode = loss_direction_mode
        
        if loss_direction_mode not in ['both', 'forward']:
            raise ValueError(f"loss_direction_mode must be one of ['both', 'forward'], got {loss_direction_mode}")
        
        args = Args(
            emb_depth=6, 
            net_units=emsize, 
            net_act_fn=torch.nn.SiLU(), 
            emb_agg_fn=global_mean_pool,  
            par_depth=3
        )
        
        if use_shared_basis_film:
            if self.num_instances is None:
                raise ValueError("num_instances must be provided when use_shared_basis_film is True.")
            self.net = Net(args, use_shared_basis_film=True, num_relations=3, num_bases=4, num_instances=self.num_instances, num_heads=8)
            self.edge_predictor = nn.Sequential(
                nn.Linear(emsize, emsize),
                nn.GELU(),
                nn.Linear(emsize, 1)
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
                    
                    eval_edge_infos.append({
                        'pos': pos,
                        'batch': b,
                        'edge_index': edge_index.clone(),
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
        
        return {
            'node_embeddings': graph_emb,
                        'edge_info': edge_info,
        }

    def _build_graph_edges(self, valid_coords, num_valid_nodes, candidate_info, pos, b, batch_size, device):
        """
        Build bidirectional graph edges with LKH3 candidate optimization.
        """
        def _create_complete_graph(num_nodes, device):
            if num_nodes <= 1:
                return torch.empty((2, 0), dtype=torch.long, device=device)
            
            nodes = torch.arange(num_nodes, device=device)
            src_nodes = nodes.unsqueeze(1).expand(-1, num_nodes).flatten()
            dst_nodes = nodes.unsqueeze(0).expand(num_nodes, -1).flatten()
            
            non_self_mask = src_nodes != dst_nodes
            edge_index = torch.stack([src_nodes[non_self_mask], dst_nodes[non_self_mask]], dim=0)
            return edge_index
        if candidate_info is not None:
            candidate_idx = pos * batch_size + b
            if candidate_idx < len(candidate_info) and candidate_info[candidate_idx] is not None:
                cand_info = candidate_info[candidate_idx]
                
                target_edges_per_node = min(self.max_candidates, num_valid_nodes - 1) if num_valid_nodes > 1 else 0
                
                edge_src_list = []
                edge_dst_list = []
                
                node_candidates = {}
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
                                
                need_nearest_neighbors = any(
                    len(node_candidates.get(node, [])) < target_edges_per_node 
                    for node in range(num_valid_nodes)
                )
                
                if need_nearest_neighbors:
                    dist_matrix = torch.cdist(valid_coords, valid_coords, p=2)
                for node in range(num_valid_nodes):
                    lkh3_candidates = node_candidates.get(node, [])
                    lkh3_candidates.sort(key=lambda x: x[1])
                    
                    selected_neighbors = set()
                    
                    for neighbor, alpha in lkh3_candidates[:target_edges_per_node]:
                        selected_neighbors.add(neighbor)
                        edge_src_list.extend([node, neighbor])
                        edge_dst_list.extend([neighbor, node])
                    
                    if len(selected_neighbors) < target_edges_per_node:
                        distances = dist_matrix[node]
                        sorted_indices = torch.argsort(distances)
                        
                        remaining_needed = target_edges_per_node - len(selected_neighbors)
                        candidates_mask = (sorted_indices != node)
                        
                        valid_candidates = sorted_indices[candidates_mask]
                        for neighbor in valid_candidates[:remaining_needed * 2]:
                            neighbor_idx = neighbor.item()
                            if neighbor_idx not in selected_neighbors and len(selected_neighbors) < target_edges_per_node:
                                selected_neighbors.add(neighbor_idx)
                                edge_src_list.extend([node, neighbor_idx])
                                edge_dst_list.extend([neighbor_idx, node])
                                
                if edge_src_list:
                    edge_index = torch.tensor([edge_src_list, edge_dst_list], dtype=torch.long, device=device)
                else:
                    edge_index = _create_complete_graph(num_valid_nodes, device)
            else:
                edge_index = _create_complete_graph(num_valid_nodes, device)
        else:
            edge_index = _create_complete_graph(num_valid_nodes, device)
        
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
        """
        seq_len, batch_size, max_num_nodes, _ = x.shape
        
        if merge_duplicate_coords:
            unique_coords = []
            coord_to_global_idx = {}
            node_offset_map = {}
            global_to_originals = {}
            instance_node_mappings = {}
            
            all_edges = []
            all_edge_attrs = []
            all_type_indices = []
            all_inst_indices = []
            instance_center_nodes = []
            inst_id = 0
            
            for pos in range(seq_len):
                for b in range(batch_size):
                    coords = x[pos, b]
                    valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
                    valid_indices = torch.where(valid_mask)[0]
                    num_valid_nodes = len(valid_indices)
                    
                    if num_valid_nodes == 0:
                        continue
                    
                    valid_coords = coords[valid_mask]
                    instance_mapping = {}
                    
                    coords_quantized = (valid_coords * 1e6).round().long()
                    
                    for local_idx in range(num_valid_nodes):
                        coord_key = tuple(coords_quantized[local_idx].cpu().tolist())
                        
                        if coord_key not in coord_to_global_idx:
                            global_idx = len(unique_coords)
                            unique_coords.append(valid_coords[local_idx])
                            coord_to_global_idx[coord_key] = global_idx
                            global_to_originals[global_idx] = []
                        else:
                            global_idx = coord_to_global_idx[coord_key]
                        
                        orig_node_idx = valid_indices[local_idx].item()
                        node_offset_map[(pos, b, orig_node_idx)] = global_idx
                        global_to_originals[global_idx].append((pos, b, orig_node_idx))
                        instance_mapping[local_idx] = global_idx
                    
                    instance_node_mappings[inst_id] = instance_mapping
                    inst_id += 1
            
            if len(unique_coords) > 0:
                merged_coords = torch.stack(unique_coords, dim=0)
            else:
                merged_coords = torch.empty((0, 2), device=x.device)
            
            inst_id = 0
            center_coords_list = []
            for pos in range(seq_len):
                for b in range(batch_size):
                    coords = x[pos, b]
                    valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
                    num_valid_nodes = valid_mask.sum().item()
                    
                    if num_valid_nodes == 0:
                        continue
                    
                    valid_coords = coords[valid_mask]
                    center_coord = valid_coords.mean(dim=0)
                    center_coords_list.append(center_coord)
                    
                    center_key = tuple((center_coord * 1e6).round().long().cpu().tolist())
                    
                    if center_key not in coord_to_global_idx:
                        center_global_idx = len(merged_coords) + len(center_coords_list) - 1
                        coord_to_global_idx[center_key] = center_global_idx
                    else:
                        center_global_idx = coord_to_global_idx[center_key]
                    
                    instance_center_nodes.append(center_global_idx)
                    inst_id += 1
            
            if center_coords_list:
                center_coords_tensor = torch.stack(center_coords_list, dim=0)
                merged_coords = torch.cat([merged_coords, center_coords_tensor], dim=0)
        else:
            all_coords = []
            all_edges = []
            all_edge_attrs = []
            all_type_indices = []
            all_inst_indices = []
            instance_center_nodes = []
            node_offset_map = {}
            instance_node_mappings = {}
            inst_id = 0
            
            coords_list = []
            center_coords_list = []
            
            for pos in range(seq_len):
                for b in range(batch_size):
                    coords = x[pos, b]
                    valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
                    valid_indices = torch.where(valid_mask)[0]
                    num_valid_nodes = len(valid_indices)
                    
                    if num_valid_nodes == 0:
                        continue
                    
                    valid_coords = coords[valid_mask]
                    coords_list.append(valid_coords)
                    
                    center_coord = valid_coords.mean(dim=0)
                    center_coords_list.append(center_coord)
                    
                    start_node_idx = sum(len(c) for c in coords_list[:-1])
                    center_node_idx = len(coords_list) - 1 + sum(len(c) for c in coords_list)
                    instance_center_nodes.append(center_node_idx)
                    
                    instance_mapping = {}
                    for i, valid_idx in enumerate(valid_indices):
                        global_idx = start_node_idx + i
                        node_offset_map[(pos, b, valid_idx.item())] = global_idx
                        instance_mapping[i] = global_idx
                    
                    instance_node_mappings[inst_id] = instance_mapping
                    inst_id += 1
            
            if coords_list:
                all_coords = torch.cat(coords_list, dim=0)
                center_coords = torch.stack(center_coords_list, dim=0)
                merged_coords = torch.cat([all_coords, center_coords], dim=0)
            else:
                merged_coords = torch.empty((0, 2), device=x.device)
        
        inst_id = 0
        edge_list = []
        edge_attr_list = []
        type_indices_list = []
        inst_indices_list = []
        
        for pos in range(seq_len):
            for b in range(batch_size):
                coords = x[pos, b]
                valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
                valid_indices = torch.where(valid_mask)[0]
                num_valid_nodes = len(valid_indices)
                
                if num_valid_nodes == 0:
                    continue
                
                valid_coords = coords[valid_mask]
                instance_mapping = instance_node_mappings[inst_id]
                
                local_graph_edges, graph_edge_attrs = self._build_graph_edges(
                    valid_coords, num_valid_nodes, candidate_info, pos, b, batch_size, x.device
                )
                
                if local_graph_edges.size(1) > 0:
                    global_graph_edges = torch.zeros_like(local_graph_edges)
                    
                    local_indices = local_graph_edges.view(-1)
                    mapping_tensor = torch.zeros(max(instance_mapping.keys()) + 1, dtype=torch.long, device=x.device)
                    
                    for local_idx, global_idx in instance_mapping.items():
                        mapping_tensor[local_idx] = global_idx
                    
                    global_indices = mapping_tensor[local_indices].view(2, -1)
                    global_graph_edges = global_indices
                    
                    edge_list.append(global_graph_edges)
                    edge_attr_list.append(graph_edge_attrs)
                    type_indices_list.append(torch.zeros(global_graph_edges.size(1), device=x.device, dtype=torch.long))
                    inst_indices_list.append(torch.full((global_graph_edges.size(1),), inst_id, device=x.device, dtype=torch.long))

                if y is not None:
                    local_tour_edges, tour_edge_attrs = self._build_tour_edges(
                        y[pos, b], valid_indices, num_valid_nodes, valid_coords, x.device
                    )
                    
                    if local_tour_edges.size(1) > 0:
                        global_tour_edges = torch.zeros_like(local_tour_edges)
                        local_indices = local_tour_edges.view(-1)
                        global_indices = mapping_tensor[local_indices].view(2, -1)
                        global_tour_edges = global_indices
                        
                        edge_list.append(global_tour_edges)
                        edge_attr_list.append(tour_edge_attrs)
                        type_indices_list.append(torch.ones(global_tour_edges.size(1), device=x.device, dtype=torch.long))
                        inst_indices_list.append(torch.full((global_tour_edges.size(1),), inst_id, device=x.device, dtype=torch.long))

                center_global_idx = instance_center_nodes[inst_id]
                
                if len(instance_mapping) > 0:
                    node_indices = torch.tensor(list(instance_mapping.values()), device=x.device, dtype=torch.long)
                    center_indices = torch.full_like(node_indices, center_global_idx)
                    
                    forward_edges = torch.stack([node_indices, center_indices], dim=0)
                    backward_edges = torch.stack([center_indices, node_indices], dim=0)
                    center_edges = torch.cat([forward_edges, backward_edges], dim=1)
                    
                    num_center_edges = center_edges.size(1)
                    rows, cols = center_edges[0], center_edges[1]
                    center_edge_attrs = torch.norm(
                        merged_coords[rows] - merged_coords[cols], dim=1
                    ).unsqueeze(1)
                    
                    edge_list.append(center_edges)
                    edge_attr_list.append(center_edge_attrs)
                    type_indices_list.append(torch.full((num_center_edges,), 2, device=x.device, dtype=torch.long))
                    inst_indices_list.append(torch.full((num_center_edges,), inst_id, device=x.device, dtype=torch.long))
                
                inst_id += 1
        
        if edge_list:
            merged_edges = torch.cat(edge_list, dim=1)
            merged_edge_attrs = torch.cat(edge_attr_list, dim=0)
            merged_type_indices = torch.cat(type_indices_list, dim=0)
            merged_inst_indices = torch.cat(inst_indices_list, dim=0)
        else:
            merged_edges = torch.empty((2, 0), dtype=torch.long, device=x.device)
            merged_edge_attrs = torch.empty((0, 1), device=x.device)
            merged_type_indices = torch.empty((0,), dtype=torch.long, device=x.device)
            merged_inst_indices = torch.empty((0,), dtype=torch.long, device=x.device)
        
        node_embs, edge_embs = self.net.infer(
            x=merged_coords,
            edge_index=merged_edges,
            edge_attr=merged_edge_attrs,
            batch=torch.zeros(merged_coords.size(0), device=x.device, dtype=torch.long),
            position=torch.zeros(merged_coords.size(0), device=x.device, dtype=torch.long),
            emb_net=self.net.emb_net,
            use_shared_basis_film=True,
            type_index=merged_type_indices,
            inst_index=merged_inst_indices,
            instance_nodes=instance_center_nodes
        )
        
        eval_start_pos = single_eval_pos or seq_len - 1
        eval_edge_index_list = []
        eval_edge_counts = []
        eval_predictions = []
        eval_edge_infos = []
        
        instance_to_pos_batch = {}
        inst_id = 0
        for pos in range(seq_len):
            for b in range(batch_size):
                coords = x[pos, b]
                valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
                num_valid_nodes = valid_mask.sum().item()
                if num_valid_nodes > 0:
                    instance_to_pos_batch[inst_id] = (pos, b, num_valid_nodes)
                    inst_id += 1
        
        for inst_id in range(len(instance_to_pos_batch)):
            pos, b, num_valid_nodes = instance_to_pos_batch[inst_id]
            
            if pos < eval_start_pos:
                continue
                
            edge_mask = (merged_type_indices == 0) & (merged_inst_indices == inst_id)
            instance_edges_global = merged_edges[:, edge_mask]
            instance_edge_embs = edge_embs[edge_mask]
            
            if self.loss_direction_mode == 'forward':
                forward_mask = instance_edges_global[0] < instance_edges_global[1]
                instance_edges_global = instance_edges_global[:, forward_mask]
                instance_edge_embs = instance_edge_embs[forward_mask]
            
            edge_preds = self.edge_predictor(instance_edge_embs).squeeze(-1) if instance_edge_embs.size(0) > 0 else torch.empty(0, device=x.device)
            
            eval_edge_index_list.append(instance_edges_global.t())
            eval_edge_counts.append(instance_edges_global.size(1))
            eval_predictions.append(edge_preds)
            
            coords = x[pos, b]
            valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
            original_valid_indices = torch.where(valid_mask)[0]
            
            eval_edge_infos.append({
                'pos': pos,
                'batch': b,
                'edge_index': instance_edges_global,
                'num_valid_nodes': num_valid_nodes,
                'valid_indices': original_valid_indices,
                'node_offset_map': {},
                'global_to_originals': global_to_originals,
                'instance_mapping': instance_node_mappings.get(inst_id, {}),
                'is_shared_basis_film': True
            })
        
        max_edges = max(eval_edge_counts) if eval_edge_counts else 1
        padded_predictions = [F.pad(preds, (0, max_edges - preds.size(0))) for preds in eval_predictions]
        seq_eval_len = seq_len - eval_start_pos
        edge_values_padded = torch.zeros(seq_eval_len, batch_size, max_edges, device=x.device)
        
        if padded_predictions:
            pred_idx = 0
            for pos in range(eval_start_pos, seq_len):
                for b in range(batch_size):
                    found = False
                    for eval_info in eval_edge_infos:
                        if eval_info['pos'] == pos and eval_info['batch'] == b:
                            if pred_idx < len(padded_predictions):
                                edge_values_padded[pos - eval_start_pos, b, :padded_predictions[pred_idx].size(0)] = padded_predictions[pred_idx]
                                pred_idx += 1
                            found = True
                            break
                    if not found:
                        pass
        
        edge_info = {
            'embeddings': None,
            'indices': None,
            'batch': None,
            'position': None,
            'node_offset_map': node_offset_map,
            'edge_counts': eval_edge_counts,
            'eval_infos': eval_edge_infos
        }
        
        return {
            'node_embeddings': torch.zeros(seq_len, batch_size, self.emsize, device=x.device),
            'edge_info': edge_info,
            'direct_predictions': True,
            'edge_predictions': edge_values_padded
        }


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


def tsp_graph_encoder_generator(num_features, emsize, max_candidates=50, use_unified_encoding=False, use_shared_basis_film=False, merge_duplicate_coords=True, num_instances=None, loss_direction_mode='both'):
    """
    Create TSP graph encoder.
    """
    return TSPGraphEncoder(
        num_features, emsize, max_candidates, use_unified_encoding, 
        use_shared_basis_film, merge_duplicate_coords, num_instances=num_instances,
        loss_direction_mode=loss_direction_mode
    )

def tsp_tour_encoder_generator(num_features, emsize, max_nodes=100):
    """
    Create TSP tour encoder.
    """
    return TSPTourEncoder(num_features, emsize, max_nodes) 