from __future__ import annotations

import itertools
import time
import yaml
import inspect
from contextlib import nullcontext
from tqdm import tqdm
import typing as tp

import torch
from torch import nn
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

from . import utils
from .priors import prior
from . import priors
from .transformer import TransformerModel
from .bar_distribution import BarDistribution, FullSupportBarDistribution, get_bucket_limits, get_custom_bar_dist
from .utils import get_cosine_schedule_with_warmup, get_openai_lr, StoreDictKeyPair, get_weighted_single_eval_pos_sampler, get_uniform_single_eval_pos_sampler
from . import positional_encodings
from .utils import init_dist, bool_mask_to_att_mask
from .priors.tsp_data_loader import TSPDataLoader
from .priors.tsp_encoder import tsp_graph_encoder_generator, tsp_tour_encoder_generator
from torch.autograd import profiler

class TSPAttentionCriterion(nn.Module):
    """
    TSP attention criterion that computes edge labels dynamically from targets.
    Supports configurable loss direction modes while always creating bidirectional edges for GNN.
    """
    def __init__(self, loss_direction_mode='both'):
        """
        Initialize TSP attention criterion.
        
        Args:
            loss_direction_mode: How to handle edge directions in loss calculation:
                - 'both': Use both directions (u->v and v->u) for bidirectional learning
                - 'forward': Use only forward/canonical direction (u->v where u < v)
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_direction_mode = loss_direction_mode
        
        if loss_direction_mode not in ['both', 'forward']:
            raise ValueError(f"loss_direction_mode must be one of ['both', 'forward'], got {loss_direction_mode}")

    def forward(self, output, targets, edge_info, single_eval_pos):
        """
        Enhanced forward pass with support for different TSP encoding modes.
        Handles standard, SharedBasisFiLM, and InstanceAwareHypergraphGNN modes efficiently.
        """
        # Store edge_info for use in loss computation functions
        self._current_edge_info = edge_info
        
        # Get tensor dimensions for proper loss tensor initialization
        seq_len = output.size(0)
        batch_size = output.size(1)
        
        # Check if we have evaluation infos (for structured modes like SharedBasisFiLM and InstanceAwareHypergraphGNN)
        eval_infos = edge_info.get('eval_infos', []) if edge_info else []
        
        if eval_infos:
            # Use optimized batch loss computation for structured modes
            losses_tensor = self._optimized_batch_loss_computation(output, targets, eval_infos, single_eval_pos)
            
            if losses_tensor.numel() == 1:
                losses_tensor = losses_tensor.expand(seq_len, batch_size)
            elif losses_tensor.size() != (seq_len, batch_size):
                if losses_tensor.numel() == seq_len * batch_size:
                    losses_tensor = losses_tensor.view(seq_len, batch_size)
            else:
                losses_tensor = losses_tensor
        
            return losses_tensor
        else:
            if edge_info and len(edge_info) >= 3:
                if isinstance(edge_info, dict):
                    edge_index_list = edge_info.get('indices', [])
                    node_offset_map = edge_info.get('node_offset_map', {})
                    edge_counts = edge_info.get('edge_counts', [])
                else:
                    edge_index_list, node_offset_map, edge_counts = edge_info[:3]
                
                eval_infos_converted = self._fast_tuple_conversion(
                    edge_index_list, node_offset_map, edge_counts,
                    targets.size(0), targets.size(1), single_eval_pos, targets, output.device
                )
                
                losses_tensor = self._compute_vectorized_loss(output, eval_infos_converted, targets, single_eval_pos)
                
                if losses_tensor.size() != (seq_len, batch_size):
                    if losses_tensor.numel() == seq_len * batch_size:
                        losses_tensor = losses_tensor.view(seq_len, batch_size)
                    else:
                        losses_tensor = torch.zeros(seq_len, batch_size, device=output.device, requires_grad=True)
                
                return losses_tensor
            else:
                return torch.zeros(seq_len, batch_size, device=output.device, requires_grad=True)

    def _fast_tuple_conversion(self, edge_index_list, node_offset_map, edge_counts, 
                              seq_len, batch_size, single_eval_pos, targets, device):
        """
        Fast conversion from tuple format to eval_infos.
        """
        eval_infos = []
        
        num_nodes = targets.size(-1)
        valid_indices_tensor = torch.arange(num_nodes, device=device)
        
        idx = 0
        for pos_idx in range(seq_len):
            for batch_idx in range(batch_size):
                if idx < len(edge_index_list) and edge_index_list[idx].size(0) > 0:
                    edge_tensor = edge_index_list[idx]
                    if edge_tensor.size(1) == 2:
                        edge_tensor = edge_tensor.t()
                    
                    pos_key = pos_idx + (single_eval_pos or 0)
                    local_node_offset_map = {
                        k: v for k, v in node_offset_map.items() 
                        if k[0] == pos_key and k[1] == batch_idx
                    }
                    
                    eval_infos.append({
                        'pos': pos_key,
                        'batch': batch_idx,
                        'edge_index': edge_tensor,
                        'num_valid_nodes': num_nodes,
                        'valid_indices': valid_indices_tensor,
                        'node_offset_map': local_node_offset_map
                    })
                idx += 1
        
        return eval_infos

    def _compute_vectorized_loss(self, output, eval_infos, targets, single_eval_pos):
        if not eval_infos:
            seq_len = output.size(0)
            batch_size = output.size(1)
            return torch.zeros(seq_len, batch_size, device=output.device, requires_grad=True)
        
        return self._optimized_batch_loss_computation(output, targets, eval_infos, single_eval_pos)

    def _optimized_batch_loss_computation(self, output, targets, eval_infos, single_eval_pos):
        if not eval_infos:
            seq_len = output.size(0) 
            batch_size = output.size(1)
            return torch.zeros(seq_len, batch_size, device=output.device, requires_grad=True)
        
        seq_len = output.size(0) 
        batch_size = output.size(1)
        
        loss_tensor = torch.zeros(seq_len, batch_size, device=output.device, requires_grad=True)
        
        edge_info_mappings = {}
        if hasattr(self, '_current_edge_info') and self._current_edge_info is not None:
            edge_info_mappings = {
                'edge_to_instances': self._current_edge_info.get('edge_to_instances', {}),
                'instance_to_edges': self._current_edge_info.get('instance_to_edges', {}),
                'instance_mapping': self._current_edge_info.get('instance_mapping', {}),
                'coordinate_merging_enabled': self._current_edge_info.get('coordinate_merging_enabled', False)
            }
        
        tour_edges_cache = {}
        
        pos_batch_map = {}
        for eval_info in eval_infos:
            pos = eval_info['pos']
            batch_idx = eval_info['batch']
            
            eval_pos_in_output = pos - single_eval_pos
            targets_pos = pos - (single_eval_pos or 0)
            
            if (eval_pos_in_output < 0 or eval_pos_in_output >= seq_len or 
                targets_pos < 0 or targets_pos >= targets.size(0) or 
                batch_idx >= batch_size):
                continue
            
            if eval_pos_in_output not in pos_batch_map:
                pos_batch_map[eval_pos_in_output] = {}
            pos_batch_map[eval_pos_in_output][batch_idx] = (eval_info, targets_pos)
        
        loss_values = torch.zeros(seq_len, batch_size, device=output.device)
        
        for output_pos, batch_infos in pos_batch_map.items():
            
            for batch_idx, (eval_info, targets_pos) in batch_infos.items():
                edge_index = eval_info.get('edge_index')
                num_valid_nodes = eval_info.get('num_valid_nodes', 0)
                valid_indices = eval_info.get('valid_indices', [])
                
                if edge_index is None or edge_index.size(1) == 0 or num_valid_nodes == 0:
                    continue
                
                tour_cache_key = (targets_pos, batch_idx, num_valid_nodes)
                if tour_cache_key not in tour_edges_cache:
                    target_tour = targets[targets_pos, batch_idx, :num_valid_nodes]
                    valid_tour_mask = target_tour != -1
                    
                    if valid_tour_mask.any():
                        valid_tour = target_tour[valid_tour_mask]
                        tour_edges_set = self._create_tour_edges_vectorized(valid_tour)
                        tour_edges_cache[tour_cache_key] = tour_edges_set
                    else:
                        tour_edges_cache[tour_cache_key] = set()
                
                tour_edges_set = tour_edges_cache[tour_cache_key]
                if not tour_edges_set:
                    continue
                
                num_edges = edge_index.size(1)
                edge_predictions = output[output_pos, batch_idx, :num_edges]
                
                if edge_predictions.numel() == 0:
                    continue
                
                enhanced_eval_info = eval_info.copy()
                enhanced_eval_info.update(edge_info_mappings)
                
                try:
                    if enhanced_eval_info.get('coordinate_merging_enabled', False):
                        edge_labels, edge_weights = self._vectorized_instance_hypergraph_labeling(
                            edge_index, tour_edges_set, enhanced_eval_info, num_edges, output.device
                        )
                    elif enhanced_eval_info.get('is_shared_basis_film', False):
                        edge_labels, edge_weights = self._vectorized_shared_basis_film_labeling(
                            edge_index, tour_edges_set, enhanced_eval_info, num_edges, output.device
                        )
                    else:
                        edge_labels, edge_weights = self._vectorized_standard_labeling(
                            edge_index, tour_edges_set, valid_indices, num_edges, output.device
                        )
                    
                    if edge_labels.numel() > 0:
                        min_size = min(edge_predictions.size(0), edge_labels.size(0))
                        if min_size > 0:
                            pred_subset = edge_predictions[:min_size]
                            label_subset = edge_labels[:min_size]
                            weight_subset = edge_weights[:min_size]
                            
                            bce_loss = F.binary_cross_entropy_with_logits(
                                pred_subset, label_subset, weight=weight_subset, reduction='sum'
                            )
                            
                            loss_values[output_pos, batch_idx] = bce_loss/weight_subset.sum()
                
                except Exception as e:
                    continue
        
        return loss_tensor + loss_values

    def _vectorized_create_edge_labels(self, edge_index, tour, valid_indices, num_valid_nodes, device, eval_info=None):
        if edge_index.size(1) == 0:
            return torch.empty(0, device=device), torch.empty(0, device=device)
        
        num_edges = edge_index.size(1)
        edge_labels = torch.zeros(num_edges, device=device)
        edge_weights = torch.zeros(num_edges, device=device)
        
        valid_tour_mask = (tour != -1)
        if not valid_tour_mask.any() or num_valid_nodes < 2:
            edge_weights.fill_(0.1)  
            return edge_labels, edge_weights
        
        valid_tour = tour[valid_tour_mask]
        if len(valid_tour) < 2:
            edge_weights.fill_(0.1)
            return edge_labels, edge_weights
        
        tour_edges_set = self._create_tour_edges_vectorized(valid_tour)
        
        is_shared_basis_film = eval_info is not None and eval_info.get('is_shared_basis_film', False)
        is_instance_hypergraph = eval_info is not None and eval_info.get('instance_id') is not None
        
        if is_shared_basis_film:
            edge_labels, edge_weights = self._vectorized_shared_basis_film_labeling(
                edge_index, tour_edges_set, eval_info, num_edges, device
            )
        elif is_instance_hypergraph:
            edge_labels, edge_weights = self._vectorized_instance_hypergraph_labeling(
                edge_index, tour_edges_set, eval_info, num_edges, device
            )
        else:
            edge_labels, edge_weights = self._vectorized_standard_labeling(
                edge_index, tour_edges_set, valid_indices, num_edges, device
            )
        
        return edge_labels, edge_weights

    def _create_tour_edges_vectorized(self, valid_tour):
        tour_edges_set = set()
        valid_tour_len = len(valid_tour)
        
        current_nodes = valid_tour
        next_nodes = torch.cat([valid_tour[1:], valid_tour[0:1]])
        
        current_cpu = current_nodes.cpu().numpy()
        next_cpu = next_nodes.cpu().numpy()
        
        for i in range(valid_tour_len):
            n1, n2 = int(current_cpu[i]), int(next_cpu[i])
            
            if self.loss_direction_mode == 'both':
                tour_edges_set.add((n1, n2))
                tour_edges_set.add((n2, n1))
            else:  # 'forward' mode
                if n1 > n2:
                    n1, n2 = n2, n1
                tour_edges_set.add((n1, n2))
        
        return tour_edges_set

    def _vectorized_instance_hypergraph_labeling(self, edge_index, tour_edges_set, eval_info, num_edges, device):
        edge_labels = torch.zeros(num_edges, device=device, dtype=torch.float32)
        edge_weights = torch.full((num_edges,), 0.1, device=device, dtype=torch.float32)
        
        if eval_info is None:
            return edge_labels, edge_weights
        
        city_node_indices = eval_info.get('city_node_indices', [])
        if city_node_indices:
            return self._process_city_node_indices_labeling_fixed(
                edge_index, tour_edges_set, city_node_indices, num_edges, device, eval_info
            )
        
        num_valid_nodes = eval_info.get('num_valid_nodes', 0)
        if num_valid_nodes > 0:
            valid_indices = torch.arange(num_valid_nodes, device=device)
            return self._vectorized_standard_labeling(edge_index, tour_edges_set, valid_indices, num_edges, device)
        else:
            return edge_labels, edge_weights
    
    def _process_city_node_indices_labeling_fixed(self, edge_index, tour_edges_set, city_node_indices, num_edges, device, eval_info):
        edge_labels = torch.zeros(num_edges, device=device, dtype=torch.float32)
        edge_weights = torch.full((num_edges,), 0.1, device=device, dtype=torch.float32)
        
        valid_indices = eval_info.get('valid_indices', [])
        if valid_indices is None or len(valid_indices) == 0:
            return edge_labels, edge_weights
        
        if hasattr(self, '_current_edge_info') and self._current_edge_info is not None:
            global_to_originals = self._current_edge_info.get('global_to_originals', {})
            pos = eval_info.get('pos')
            batch = eval_info.get('batch')
            
            global_to_local_orig = {}
            for global_idx in city_node_indices:
                if global_idx in global_to_originals:
                    for orig_pos, orig_batch, orig_node in global_to_originals[global_idx]:
                        if orig_pos == pos and orig_batch == batch:
                            global_to_local_orig[global_idx] = orig_node
                            break
            
            if len(global_to_local_orig) == 0:
                for local_idx, global_idx in enumerate(city_node_indices):
                    if local_idx < len(valid_indices):
                        orig_node = valid_indices[local_idx].item()
                        global_to_local_orig[global_idx] = orig_node
            
            tour_edges_global = set()
            for orig_u, orig_v in tour_edges_set:
                global_u = None
                global_v = None
                
                for global_idx, local_orig in global_to_local_orig.items():
                    if local_orig == orig_u:
                        global_u = global_idx
                    if local_orig == orig_v:
                        global_v = global_idx
                
                if global_u is not None and global_v is not None:
                    tour_edges_global.add((global_u, global_v))
            
            if len(tour_edges_global) > 10:

                unique_tour_edges = set()
                for u, v in tour_edges_global:
                    canonical_edge = (min(u, v), max(u, v))
                    unique_tour_edges.add(canonical_edge)
                
                if len(unique_tour_edges) > 5:
                    tour_edges_global = set()
                    for local_idx, global_idx in enumerate(city_node_indices):
                        if local_idx < len(valid_indices):
                            orig_node = valid_indices[local_idx].item()
                            for orig_u, orig_v in tour_edges_set:
                                if orig_u == orig_node:
                                    for local_idx2, global_idx2 in enumerate(city_node_indices):
                                        if local_idx2 < len(valid_indices) and valid_indices[local_idx2].item() == orig_v:
                                            tour_edges_global.add((global_idx, global_idx2))
                                            break
        else:
            orig_to_global = {}
            for local_idx, global_idx in enumerate(city_node_indices):
                if local_idx < len(valid_indices):
                    orig_node = valid_indices[local_idx].item()
                    orig_to_global[orig_node] = global_idx
            
            tour_edges_global = set()
            for u_orig, v_orig in tour_edges_set:
                if u_orig in orig_to_global and v_orig in orig_to_global:
                    u_global = orig_to_global[u_orig]
                    v_global = orig_to_global[v_orig]
                    tour_edges_global.add((u_global, v_global))
        
        # 检查每条边是否为tour边
        tour_edges_found = 0
        
        for edge_idx in range(min(num_edges, edge_index.size(1))):
            u_global = edge_index[0, edge_idx].item()
            v_global = edge_index[1, edge_idx].item()
            
            if (u_global, v_global) in tour_edges_global:
                edge_labels[edge_idx] = 1.0
                edge_weights[edge_idx] = 1.0
                tour_edges_found += 1
            else:
                edge_labels[edge_idx] = 0.0
                edge_weights[edge_idx] = 0.1
        
        return edge_labels, edge_weights

    def _vectorized_standard_labeling(self, edge_index, tour_edges_set, valid_indices, num_edges, device):
        """
        Vectorized labeling for standard mode.
        """
        edge_labels = torch.zeros(num_edges, device=device)
        edge_weights = torch.zeros(num_edges, device=device)
        
        if valid_indices is None:
            edge_weights.fill_(0.1)
            return edge_labels, edge_weights
        
        u_indices = edge_index[0].cpu().numpy()
        v_indices = edge_index[1].cpu().numpy()
        valid_indices_cpu = valid_indices.cpu().numpy()
        
        global_to_local = {}
        for local_idx, global_idx in enumerate(valid_indices_cpu):
            global_to_local[global_idx] = local_idx
        
        for i in range(num_edges):
            u_global = int(u_indices[i])
            v_global = int(v_indices[i])
                
            if u_global in global_to_local and v_global in global_to_local:
                u_local = global_to_local[u_global]
                v_local = global_to_local[v_global]
                
                if (u_local, v_local) in tour_edges_set:
                    edge_labels[i] = 1.0
                    edge_weights[i] = 1.0
                else:
                    edge_labels[i] = 0.0
                    edge_weights[i] = 0.1
            else:
                edge_labels[i] = 0.0
                edge_weights[i] = 0.1
        
        return edge_labels, edge_weights

    def _vectorized_shared_basis_film_labeling(self, edge_index, tour_edges_set, eval_info, num_edges, device):
        """
        OPTIMIZATION 7: Optimized SharedBasisFiLM edge mapping with caching.
        """
        edge_labels = torch.zeros(num_edges, device=device)
        edge_weights = torch.zeros(num_edges, device=device)
        
        global_to_originals = eval_info['global_to_originals']
        pos = eval_info['pos']
        batch = eval_info['batch']
        
        # Pre-filter global_to_originals for this position/batch
        relevant_originals = {}
        for global_idx, originals_list in global_to_originals.items():
            relevant_originals[global_idx] = [
                orig_node for orig_pos, orig_batch, orig_node in originals_list
                if orig_pos == pos and orig_batch == batch
            ]
        
        # Process edges with optimized lookup
        u_indices = edge_index[0].cpu().numpy()
        v_indices = edge_index[1].cpu().numpy()
        
        for i in range(num_edges):
            u_idx, v_idx = int(u_indices[i]), int(v_indices[i])
            
            u_originals = relevant_originals.get(u_idx, [])
            v_originals = relevant_originals.get(v_idx, [])
                
            # Check if any combination forms a tour edge
            found_tour_edge = False
            for u_orig in u_originals:
                for v_orig in v_originals:
                    if (u_orig, v_orig) in tour_edges_set:
                        found_tour_edge = True
                        break
                if found_tour_edge:
                    break
                
            if found_tour_edge:
                edge_labels[i] = 1.0
                edge_weights[i] = 1.0
            else:
                edge_labels[i] = 0.0
                edge_weights[i] = 0.1
        
        return edge_labels, edge_weights

class Losses():
    gaussian = nn.GaussianNLLLoss(full=True, reduction='none')
    mse = nn.MSELoss(reduction='none')
    ce = lambda num_classes: nn.CrossEntropyLoss(reduction='none', weight=torch.ones(num_classes))
    bce = nn.BCEWithLogitsLoss(reduction='none')
    get_BarDistribution = BarDistribution
    
class TrainingResult(tp.NamedTuple):
    # the mean loss in the last epoch across dataset sizes (single_eval_pos's)
    total_loss: tp.Optional[float]
    # the mean loss in the last epoch for each dataset size (single_eval_pos's)
    total_positional_losses: tp.Optional[tp.List[float]]
    # the trained model
    model: nn.Module
    # the dataloader used for training
    data_loader: tp.Optional[torch.utils.data.DataLoader]


def train(priordataloader_class_or_get_batch: prior.PriorDataLoader | callable, criterion, encoder_generator, emsize=200, nhid=200, nlayers=6, nhead=2, dropout=0.0,
          epochs=10, steps_per_epoch=100, batch_size=200, seq_len=10, lr=None, weight_decay=0.0, warmup_epochs=10, input_normalization=False,
          y_encoder_generator=None, pos_encoder_generator=None, decoder_dict={}, extra_prior_kwargs_dict={}, scheduler=get_cosine_schedule_with_warmup,
          load_weights_from_this_state_dict=None, validation_period=10, single_eval_pos_gen=None, gpu_device='cuda:0',
          aggregate_k_gradients=1, verbose=True, style_encoder_generator=None, epoch_callback=None, step_callback=None, continue_model=None,
          initializer=None, initialize_with_model=None, train_mixed_precision=False, efficient_eval_masking=True, border_decoder=None
          , num_global_att_tokens=0, progress_bar=False, use_residual_norm=False, **model_extra_args):
    device: str = gpu_device if torch.cuda.is_available() else 'cpu:0'
    print(f'Using {device} device')
    using_dist, rank, device = init_dist(device)
    single_eval_pos_gen = single_eval_pos_gen if callable(single_eval_pos_gen) else lambda: single_eval_pos_gen

    if inspect.isclass(priordataloader_class_or_get_batch) and issubclass(priordataloader_class_or_get_batch, prior.PriorDataLoader):
        priordataloader_class = priordataloader_class_or_get_batch
    else:
        priordataloader_class = priors.utils.get_batch_to_dataloader(priordataloader_class_or_get_batch)
        

    def eval_pos_seq_len_sampler():
        single_eval_pos = single_eval_pos_gen()
        return single_eval_pos, seq_len
    dl = priordataloader_class(num_steps=steps_per_epoch,
                               batch_size=batch_size,
                               eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
                               seq_len_maximum=seq_len,
                               device=device,
                               num_processes=8,
                               **extra_prior_kwargs_dict)

    test_batch: prior.Batch = dl.get_test_batch()
    style_def = test_batch.style
    style_encoder = style_encoder_generator(style_def.shape[1], emsize) if (style_def is not None) else None
    pos_encoder = (pos_encoder_generator or positional_encodings.NoPositionalEncoding)(emsize, seq_len * 2)
    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif isinstance(criterion, BarDistribution) or "BarDistribution" in criterion.__class__.__name__: # TODO remove this fix (only for dev)
        n_out = criterion.num_bars
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = criterion.weight.shape[0]
    else:
        n_out = 1

    if continue_model:
        model = continue_model
    else:
        decoder_dict = decoder_dict if decoder_dict else {'standard': (None, n_out)}

        decoder_once_dict = {}
        if test_batch.mean_prediction is not None:
            decoder_once_dict['mean_prediction'] = decoder_dict['standard']

        encoder = encoder_generator(dl.num_features, emsize, use_residual_norm=use_residual_norm)
        model = TransformerModel(encoder=encoder
                                 , nhead=nhead
                                 , ninp=emsize
                                 , nhid=nhid
                                 , nlayers=nlayers
                                 , dropout=dropout
                                 , style_encoder=style_encoder
                                 , y_encoder=y_encoder_generator(1, emsize) if y_encoder_generator is not None else None
                                 , input_normalization=input_normalization
                                 , pos_encoder=pos_encoder
                                 , decoder_dict=decoder_dict
                                 , init_method=initializer
                                 , efficient_eval_masking=efficient_eval_masking
                                 , decoder_once_dict=decoder_once_dict
                                 , num_global_att_tokens=num_global_att_tokens
                                 , **model_extra_args
                                 )
    model.criterion = criterion
    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)
    if initialize_with_model is not None:
        model.init_from_small_model(initialize_with_model)



    model.to(device)
    if using_dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank],
                                                          output_device=rank,
                                                          broadcast_buffers=False,
                                                          find_unused_parameters=test_batch.mean_prediction is not None)
        dl.model = model.module # use local model, should not use multi-gpu functionality..
    else:
        dl.model = model

    # learning rate
    if lr is None:
        lr = get_openai_lr(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler(optimizer, warmup_epochs, epochs if epochs is not None else 100) # when training for fixed time lr schedule takes 100 steps

    scaler = GradScaler() if train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    def train_epoch():
        model.train()  # Turn on the train mode
        total_loss = 0.
        total_positional_losses = 0.
        total_positional_losses_recorded = 0
        nan_steps = 0
        ignore_steps = 0
        before_get_batch = time.time()
        assert len(dl) % aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'
        tqdm_iter = tqdm(range(len(dl)), desc='Training Epoch') if progress_bar else None

        for batch, full_data in enumerate(dl):
            data = (full_data.style.to(device) if full_data.style is not None else None, full_data.x.to(device), full_data.y.to(device))
            targets = full_data.target_y.to(device)
            single_eval_pos = full_data.single_eval_pos
            candidate_info = getattr(full_data, 'candidate_info', None)  # Extract candidate_info from batch
            
            def get_metrics():
                return total_loss / steps_per_epoch, (
                        total_positional_losses / total_positional_losses_recorded).tolist(), \
                       time_to_get_batch, forward_time, step_time, nan_steps.cpu().item() / (batch + 1), \
                       ignore_steps.cpu().item() / (batch + 1)

            tqdm_iter.update() if tqdm_iter is not None else None
            if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1):
                cm = model.no_sync()
            else:
                cm = nullcontext()
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                before_forward = time.time()
                
                metrics_to_log = {}
                with autocast(device.split(':')[0], enabled=scaler is not None):
                    output, edge_info = model(tuple(e.to(device) if torch.is_tensor(e) else e for e in data),
                                single_eval_pos=single_eval_pos, only_return_standard_out=False, candidate_info=candidate_info)
                    
                    forward_time = time.time() - before_forward
                    before_loss = time.time()

                    if single_eval_pos is not None:
                        targets = targets[single_eval_pos:]
                
                    losses = criterion(output, targets, edge_info, single_eval_pos)
                    
                    # Handle losses tensor shape - it should already be [seq_len, batch_size]
                    if losses.dim() == 2 and losses.size() == (output.shape[0], output.shape[1]):
                        # Losses are already in correct shape [seq_len, batch_size]
                        pass
                    elif losses.dim() == 1 and losses.numel() == output.shape[0] * output.shape[1]:
                        # Reshape from flattened to [seq_len, batch_size]
                        losses = losses.view(output.shape[0], output.shape[1])
                    else:
                        # Fallback: create zero tensor with correct shape
                        losses = torch.zeros(output.shape[0], output.shape[1], device=output.device, requires_grad=True)
                                                              
                    loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)

                    # If loss is a zero tensor with no grad_fn (e.g. from a batch with no valid targets),
                    # connect it to the computation graph to prevent a crash in backward().
                    if not loss.requires_grad:
                        loss = loss + 0.0 * output.sum()
                        
                    loss_scaled = loss / aggregate_k_gradients
                    loss_time = time.time() - before_loss

                # Scale the loss if using mixed precision
                if scaler: 
                    loss_scaled = scaler.scale(loss_scaled)

                loss_scaled.backward()
                
                loss_backward_time = time.time() - before_loss

                if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                    if scaler:
                        # Unscale gradients before clipping
                        scaler.unscale_(optimizer)
                        
                        # Check for invalid gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        
                        # Step the optimizer
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                    optimizer.zero_grad()

                step_time = time.time() - before_forward

                if not torch.isnan(loss):
                    total_loss += loss.cpu().detach().item()
                    total_positional_losses += losses.mean(1).cpu().detach() if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), seq_len)*\
                        utils.torch_nanmean(losses[:seq_len-single_eval_pos].mean(0)).cpu().detach()

                    total_positional_losses_recorded += torch.ones(seq_len) if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), seq_len)

                    metrics_to_log = {**metrics_to_log, **{f"loss": loss, "single_eval_pos": single_eval_pos}}
                    if step_callback is not None and rank == 0:
                        step_callback(metrics_to_log)
                    nan_steps += nan_share
                    ignore_steps += (targets == -100).float().mean()
            # except Exception as e:
            #     print("Invalid step encountered, skipping...")
            #     print(e)
            #     raise(e)

            if tqdm_iter:
                tqdm_iter.set_postfix({'data_time': time_to_get_batch, 'step_time': step_time, 'mean_loss': total_loss / (batch+1)})

            before_get_batch = time.time()
        return get_metrics()

    total_loss = float('inf')
    total_positional_losses = float('inf')
    try:
        # Initially test the epoch callback function
        if epoch_callback is not None and rank == 0:
            epoch_callback(model, 1, data_loader=dl, scheduler=scheduler)
        for epoch in (range(1, epochs + 1) if epochs is not None else itertools.count(1)):
            epoch_start_time = time.time()
            try:
                total_loss, total_positional_losses, time_to_get_batch, forward_time, step_time, nan_share, ignore_share =\
                    train_epoch()
            except Exception as e:
                print("Invalid epoch encountered, skipping...")
                print(e)
                raise (e)
            if hasattr(dl, 'validate') and epoch % validation_period == 0:
                with torch.no_grad():
                    val_score = dl.validate(model)
            
            else:
                val_score = None

            if verbose:
                print('-' * 89)
                print(
                    f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.2f} | '
                    f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                    f' data time {time_to_get_batch:5.2f} step time {step_time:5.2f}'
                    f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}'
                    + (f'val score {val_score}' if val_score is not None else ''))
                print('-' * 89)

            # stepping with wallclock time based scheduler
            if epoch_callback is not None and rank == 0:
                epoch_callback(model, epoch, data_loader=dl, scheduler=scheduler)
            scheduler.step()
    except KeyboardInterrupt:
        pass

    if rank == 0: # trivially true for non-parallel training
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            dl = None
        return TrainingResult(total_loss, total_positional_losses, model.to('cpu'), dl)

def _parse_args(config_parser, parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def train_tsp(
    emsize=200, 
    nhid=200, 
    nlayers=6, 
    nhead=2, 
    dropout=0.0,
    epochs=10, 
    steps_per_epoch=100, 
    batch_size=32, 
    seq_len=20, 
    lr=None, 
    weight_decay=0.0, 
    warmup_epochs=0,
    num_nodes_range=(10, 20),
    gpu_device=None,
    max_candidates=15,
    priordataloader_class=None,
    use_unified_encoding=False,
    use_shared_basis_film=False,
    use_instance_hypergraph=False,
    merge_duplicate_coords=True,
    loss_direction_mode='both',
    edge_type_mode='triple',
    prediction_mode='dot_product',
    use_residual_norm=False,
    **extra_args
):
    """
    Train a Transformer model for TSP instances using GNN for node encoding.
    Uses the original train() function with custom encoders and loss function.
    Always creates bidirectional edges for optimal GNN performance.
    
    Args:
        emsize: Embedding size
        nhid: Hidden dimension in transformer
        nlayers: Number of transformer layers
        nhead: Number of attention heads
        dropout: Dropout rate
        epochs: Number of training epochs
        steps_per_epoch: Number of steps per epoch
        batch_size: Batch size
        seq_len: Maximum sequence length
        lr: Learning rate (if None, uses OpenAI schedule)
        weight_decay: Weight decay for optimizer
        warmup_epochs: Number of warmup epochs for learning rate
        num_nodes_range: Range of nodes in TSP instances (min, max)
        gpu_device: Device to use for computation (defaults to cuda if available)
        max_candidates: Maximum number of candidates per node for LKH3
        priordataloader_class: Custom dataloader class (defaults to TSPDataLoader)
        use_unified_encoding: If True, uses unified encoding that combines graph and tour information
        use_shared_basis_film: If True, uses SharedBasisFiLMAttentionGNN for merged large graph processing
        merge_duplicate_coords: If True and use_shared_basis_film=True, merge nodes with identical coordinates
        loss_direction_mode: How to handle edge directions in loss calculation. Options:
            - 'both': Use both directions (u->v and v->u) for bidirectional learning - DEFAULT
            - 'forward': Use only forward/canonical direction (u->v where u < v)
        edge_type_mode: Edge type mode for no-merge SharedBasisFiLM. Options:
            - 'triple': Use three edge types (0=graph, 1=tour, 2=center) - DEFAULT
            - 'single': Use single edge type with additional features (is_solution, is_context)
        **extra_args: Additional arguments for train function
        
    Returns:
        TrainingResult object
    """
    device = gpu_device if gpu_device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Validate direction mode
    if loss_direction_mode not in ['both', 'forward']:
        raise ValueError(f"loss_direction_mode must be one of ['both', 'forward'], got {loss_direction_mode}")
    
    # Use provided dataloader class or default to TSPDataLoader
    if priordataloader_class is None:
        priordataloader_class = TSPDataLoader
    
    # Create single_eval_pos sampler
    single_eval_pos_sampler = get_uniform_single_eval_pos_sampler(seq_len, min_len=3)
    
    # Create custom loss for edge prediction with direction control
    tsp_criterion = TSPAttentionCriterion(loss_direction_mode=loss_direction_mode)
    
    # Prepare extra_prior_kwargs_dict
    default_kwargs = {
        'num_nodes_range': num_nodes_range,
        'max_candidates': max_candidates
    }
    
    # Merge with any additional kwargs passed in
    if 'extra_prior_kwargs_dict' in extra_args:
        default_kwargs.update(extra_args['extra_prior_kwargs_dict'])
        extra_args = {k: v for k, v in extra_args.items() if k != 'extra_prior_kwargs_dict'}
    
    # Define valid model parameters
    valid_model_params = {
        'input_normalization', 'init_method', 'pre_norm', 'activation',
        'recompute_attn', 'num_global_att_tokens', 'full_attention',
        'all_layers_same_init', 'efficient_eval_masking', 'decoder_once_dict',
        'return_all_outputs', 'save_trainingset_representations'
    }
    
    # Filter out parameters that should not be passed to TransformerModel
    model_extra_args = {k: v for k, v in extra_args.items() 
                       if k in valid_model_params}
    
    # Create encoder generator (always creates bidirectional edges)
    num_instances = batch_size * seq_len
    encoder_generator = lambda num_features, emsize, **kwargs: tsp_graph_encoder_generator(
        num_features, emsize, 
        max_candidates=max_candidates, 
        use_unified_encoding=use_unified_encoding,
        use_shared_basis_film=use_shared_basis_film,
        use_instance_hypergraph=use_instance_hypergraph,
        merge_duplicate_coords=merge_duplicate_coords,
        num_instances=num_instances,
        loss_direction_mode=loss_direction_mode,
        edge_type_mode=edge_type_mode,
        prediction_mode=prediction_mode,
        use_residual_norm=use_residual_norm
    )
    
    # Determine y_encoder_generator based on encoding setting
    if use_instance_hypergraph:
        # InstanceAwareHypergraphGNN mode does direct edge prediction, no separate y_encoder needed
        y_encoder_generator = None
    elif use_shared_basis_film:
        # SharedBasisFiLM mode does direct edge prediction, no separate y_encoder needed
        y_encoder_generator = None
    elif use_unified_encoding:
        # When using unified encoding, we don't need a separate y_encoder
        y_encoder_generator = None
    else:
        # Use separate tour encoder
        y_encoder_generator = lambda num_features, emsize: tsp_tour_encoder_generator(
            num_features, emsize, max_nodes=max(num_nodes_range)
        )
    
    # Use train() function with the custom components
    result = train(
        priordataloader_class_or_get_batch=priordataloader_class,
        criterion=tsp_criterion,
        encoder_generator=encoder_generator,
        y_encoder_generator=y_encoder_generator,
        emsize=emsize,
        nhid=nhid,
        nlayers=nlayers,
        nhead=nhead,
        dropout=dropout,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        extra_prior_kwargs_dict=default_kwargs,
        single_eval_pos_gen=single_eval_pos_sampler,
        gpu_device=device,
        progress_bar=extra_args.get('progress_bar', True),
        use_residual_norm=use_residual_norm,
        **model_extra_args
    )
    
    return result