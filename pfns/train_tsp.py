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
        Compute loss for TSP edge prediction by dynamically creating edge labels.
        
        Args:
            output: Model predictions [seq_len, batch_size, max_edges]
            targets: Target tours [seq_len, batch_size, num_nodes] 
            edge_info: Edge information with structural data in eval_infos (unified dict format)
            single_eval_pos: Position where evaluation starts
            
        Returns:
            losses: Loss tensor [seq_len, batch_size]
        """
        seq_len, batch_size, _ = output.shape
        losses = torch.zeros((seq_len, batch_size), device=output.device)
        
        # Handle unified dict format with eval_infos
        eval_infos = edge_info.get('eval_infos', [])
        
        if not eval_infos:
            return losses
        
        # Process each evaluation instance and dynamically compute labels
        for eval_info in eval_infos:
            pos = eval_info['pos']
            batch = eval_info['batch']
            edge_index = eval_info['edge_index']
            num_valid_nodes = eval_info['num_valid_nodes']
            valid_indices = eval_info['valid_indices']
            
            # Calculate position in output tensor
            if single_eval_pos is not None:
                output_pos = pos - single_eval_pos
            else:
                output_pos = pos
            
            # Skip if outside output range
            if output_pos < 0 or output_pos >= seq_len:
                continue
            
            # Get the target tour for this instance
            tour = targets[output_pos, batch]
            
            # Dynamically compute edge labels and weights
            edge_labels, edge_weights = self._create_edge_labels(
                edge_index, tour, valid_indices, num_valid_nodes, output.device, eval_info
            )
            
            # Apply loss calculation
            if edge_labels is not None and edge_weights is not None and len(edge_labels) > 0:
                num_edges = len(edge_labels)
                
                # Get model predictions for this instance
                pred_logits = output[output_pos, batch, :num_edges]
                
                # Compute weighted BCE loss
                if len(pred_logits) == len(edge_labels):
                    loss = (self.bce(pred_logits, edge_labels) * edge_weights).sum()
                    
                    # Normalize by number of valid nodes instead of total edges
                    if num_valid_nodes > 0:
                        loss = loss / num_valid_nodes
                        losses[output_pos, batch] = loss
        
        return losses

    def _create_edge_labels(self, edge_index, tour, valid_indices, num_valid_nodes, device, eval_info=None):
        """
        Create ground truth edge labels and weights for loss calculation.
        Always handles bidirectional edges correctly based on loss_direction_mode.
        Supports both standard mode and merged coordinate mode (SharedBasisFiLM).
        
        Args:
            edge_index: Edge indices tensor [2, num_edges] - LOCAL indices for standard mode, GLOBAL indices for SharedBasisFiLM
            tour: Tour sequence tensor [num_nodes] (with padding) - ORIGINAL indices
            valid_indices: Valid node indices tensor - ORIGINAL indices  
            num_valid_nodes: Number of valid nodes
            device: Computing device
            eval_info: Additional evaluation info (contains mapping information for SharedBasisFiLM mode)
            
        Returns:
            edge_labels: Binary labels for each edge [num_edges]
            edge_weights: Weights for each edge in loss calculation [num_edges]
        """
        if edge_index.size(1) == 0:
            return torch.empty(0, device=device), torch.empty(0, device=device)
        
        num_edges = edge_index.size(1)
        edge_labels = torch.zeros(num_edges, device=device)
        edge_weights = torch.zeros(num_edges, device=device)
        
        # Filter out padding values (-1) from tour
        valid_tour_mask = (tour != -1)
        if not valid_tour_mask.any() or num_valid_nodes < 2:
            # No valid tour, return all negative labels with reduced weight
            edge_weights.fill_(0.25)
            return edge_labels, edge_weights
        
        valid_tour = tour[valid_tour_mask]
        if len(valid_tour) < 2:
            # Too few valid nodes for a tour
            edge_weights.fill_(0.25)
            return edge_labels, edge_weights
        
        # Create tour edges set based on loss_direction_mode (using ORIGINAL indices)
        tour_edges = set()
        valid_tour_len = len(valid_tour)
        for i in range(valid_tour_len):
            n1, n2 = valid_tour[i].item(), valid_tour[(i + 1) % valid_tour_len].item()
            
            if self.loss_direction_mode == 'both':
                # Add both directions for bidirectional loss
                tour_edges.add((n1, n2))
                tour_edges.add((n2, n1))
            else:  # 'forward' mode
                # Add only canonical direction (smaller index first)
                if n1 > n2:
                    n1, n2 = n2, n1
                tour_edges.add((n1, n2))
        
        # Check if we're in SharedBasisFiLM mode with merged coordinates
        is_shared_basis_film = eval_info is not None and eval_info.get('is_shared_basis_film', False)
        
        # Label each edge
        for i in range(num_edges):
            u_idx, v_idx = edge_index[0, i].item(), edge_index[1, i].item()
            
            if is_shared_basis_film:
                # SharedBasisFiLM mode: edge_index contains GLOBAL indices
                # Need to map global indices to original indices
                global_to_originals = eval_info['global_to_originals']
                instance_mapping = eval_info['instance_mapping']
                pos = eval_info['pos']
                batch = eval_info['batch']
                
                # Find original nodes that correspond to these global indices
                u_originals = []
                v_originals = []
                
                # Get original nodes for global index u_idx
                if u_idx in global_to_originals:
                    for orig_pos, orig_batch, orig_node in global_to_originals[u_idx]:
                        if orig_pos == pos and orig_batch == batch:
                            u_originals.append(orig_node)
                
                # Get original nodes for global index v_idx  
                if v_idx in global_to_originals:
                    for orig_pos, orig_batch, orig_node in global_to_originals[v_idx]:
                        if orig_pos == pos and orig_batch == batch:
                            v_originals.append(orig_node)
                
                # Check if any combination of original nodes forms a tour edge
                found_tour_edge = False
                for u_orig in u_originals:
                    for v_orig in v_originals:
                        edge_tuple = (u_orig, v_orig)
                        if edge_tuple in tour_edges:
                            found_tour_edge = True
                            break
                    if found_tour_edge:
                        break
                
                if found_tour_edge:
                    edge_labels[i] = 1.0  # Positive label for tour edges
                    edge_weights[i] = 1.0  # Full weight for tour edges
                else:
                    edge_labels[i] = 0.0  # Negative label for non-tour edges
                    edge_weights[i] = 0.25  # Reduced weight for non-tour edges
            else:
                # Standard mode: edge_index contains LOCAL indices
                # Map local indices to original indices
                if u_idx < len(valid_indices) and v_idx < len(valid_indices):
                    u_original = valid_indices[u_idx].item()
                    v_original = valid_indices[v_idx].item()
                    
                    edge_tuple = (u_original, v_original)
                    
                    if edge_tuple in tour_edges:
                        edge_labels[i] = 1.0  # Positive label for tour edges
                        edge_weights[i] = 1.0  # Full weight for tour edges
                    else:
                        edge_labels[i] = 0.0  # Negative label for non-tour edges
                        edge_weights[i] = 0.25  # Reduced weight for non-tour edges
                else:
                    # Invalid local indices, treat as negative with reduced weight
                    edge_labels[i] = 0.0
                    edge_weights[i] = 0.25
        
        # Debug: print label statistics for first few batches
        if not hasattr(self, '_debug_label_count'):
            self._debug_label_count = 0
        
        if self._debug_label_count < 1:  # Only print for first 3 calls
            positive_labels = (edge_labels == 1.0).sum().item()
            total_edges = len(edge_labels)
            print(f"Debug _create_edge_labels #{self._debug_label_count}: {positive_labels}/{total_edges} positive labels")
            print(f"  tour_edges: {len(tour_edges)}, valid_indices: {valid_indices.tolist()}")
            print(f"  Sample edges: {[(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(min(5, num_edges))]}")
            self._debug_label_count += 1
        
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
          initializer=None, initialize_with_model=None, train_mixed_precision=True, efficient_eval_masking=True, border_decoder=None
          , num_global_att_tokens=0, progress_bar=False, **model_extra_args):
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
    print(f'Style definition of first 3 examples: {style_def[:3] if style_def is not None else None}')
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

        encoder = encoder_generator(dl.num_features, emsize)
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

    print(f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters")

    try:
        for (k, v), (k2, v2) in zip(model.state_dict().items(), initialize_with_model.state_dict().items()):
            print(k, ((v - v2) / v).abs().mean(), v.shape)
    except Exception:
        pass

    model.to(device)
    if using_dist:
        print("Distributed training")
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
        print(f"Using OpenAI max lr of {lr}.")
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
                    losses = losses.view(-1, output.shape[1]) 
                                                              
                    loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)

                    # If loss is a zero tensor with no grad_fn (e.g. from a batch with no valid targets),
                    # connect it to the computation graph to prevent a crash in backward().
                    if not loss.requires_grad:
                        loss = loss + 0.0 * output.sum()
                        
                    loss_scaled = loss / aggregate_k_gradients
                    loss_time = time.time() - before_loss

                if scaler: loss_scaled = scaler.scale(loss_scaled)

                loss_scaled.backward()
                
                loss_backward_time = time.time() - before_loss

                if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                    if scaler: scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
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
    merge_duplicate_coords=True,
    loss_direction_mode='both',
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
        **extra_args: Additional arguments for train function
        
    Returns:
        TrainingResult object
    """
    device = gpu_device if gpu_device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Validate direction mode
    if loss_direction_mode not in ['both', 'forward']:
        raise ValueError(f"loss_direction_mode must be one of ['both', 'forward'], got {loss_direction_mode}")
    
    # Log configuration
    print(f"Training TSP model on {device} with {emsize} embedding size")
    print(f"Edge configuration:")
    print(f"  - Always create bidirectional edges: True (optimal for GNN)")
    print(f"  - Loss direction mode: {loss_direction_mode}")
    
    if loss_direction_mode == 'both':
        print("  ✓ Optimal configuration: bidirectional edge creation + bidirectional loss calculation")
    else:
        print("  ✓ Standard configuration: bidirectional edge creation + forward loss calculation")
    
    # Use provided dataloader class or default to TSPDataLoader
    if priordataloader_class is None:
        priordataloader_class = TSPDataLoader
    
    print(f"Using dataloader: {priordataloader_class.__name__}")
    
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
    encoder_generator = lambda num_features, emsize: tsp_graph_encoder_generator(
        num_features, emsize, 
        max_candidates=max_candidates, 
        use_unified_encoding=use_unified_encoding,
        use_shared_basis_film=use_shared_basis_film,
        merge_duplicate_coords=merge_duplicate_coords,
        num_instances=num_instances,
        loss_direction_mode=loss_direction_mode
    )
    
    # Determine y_encoder_generator based on encoding setting
    if use_shared_basis_film:
        # SharedBasisFiLM mode does direct edge prediction, no separate y_encoder needed
        y_encoder_generator = None
        print(f"Using SharedBasisFiLM mode - merging all instances into single large graph with direct edge prediction")
    elif use_unified_encoding:
        # When using unified encoding, we don't need a separate y_encoder
        y_encoder_generator = None
        print(f"Using unified encoding - graph and tour information will be processed together")
    else:
        # Use separate tour encoder
        y_encoder_generator = lambda num_features, emsize: tsp_tour_encoder_generator(
            num_features, emsize, max_nodes=max(num_nodes_range)
        )
        print(f"Using separate encoders - graph encoder and tour encoder")
    
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
        **model_extra_args
    )
    
    return result