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
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, targets, edge_info, single_eval_pos):
        edge_index_list, node_offset_map, edge_counts = edge_info
        seq_len, batch_size, num_nodes = targets.shape
        
        losses = torch.zeros((seq_len, batch_size), device=output.device)
        
        reversed_node_map = {value: key for key, value in node_offset_map.items()}
        
        for i in range(seq_len):
            for j in range(batch_size):
                idx = (single_eval_pos + i)*batch_size + j
                num_edges = edge_counts[idx]
                edges = edge_index_list[idx].cpu().tolist()
                edge_labels = torch.zeros(num_edges, device=output.device)
                
                tour = targets[i, j].cpu().tolist()
                tour_edges = set()
                tour_len = len(tour)
                for k in range(tour_len):
                    n1, n2 = tour[k], tour[(k + 1) % tour_len]
                    if n1 > n2:
                        n1, n2 = n2, n1
                    tour_edges.add((n1, n2))
                
                # sorted_edges = [(min(reversed_node_map[edge[0]][-1], reversed_node_map[edge[1]][-1]),
                #                  max(reversed_node_map[edge[0]][-1], reversed_node_map[edge[1]][-1]))
                #                 for edge in edges]
                weights = torch.zeros_like(edge_labels)

                for e_idx, (node0, node1) in enumerate(edges):
                    u, v = reversed_node_map[node0][-1], reversed_node_map[node1][-1]
                    if ((u,v) in tour_edges) or ((v,u) in tour_edges):
                        edge_labels[e_idx] = 1.0 
                        weights[e_idx] = 1.0
                    else:
                        edge_labels[e_idx] = 0.0
                        weights[e_idx] = 0.25

                loss = (self.bce(output[i, j, :num_edges], edge_labels) * weights).sum() / num_nodes
                losses[i, j] = loss
        
        return losses
        
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
                                 , y_encoder=y_encoder_generator(1, emsize)
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
        tqdm_iter = tqdm(range(len(dl)), desc='Training Epoch') if rank==0 and progress_bar else None

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
                try:
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
                except Exception as e:
                    print("Invalid step encountered, skipping...")
                    print(e)
                    raise(e)

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
    **extra_args
):
    """
    Train a Transformer model for TSP instances using GNN for node encoding.
    Uses the original train() function with custom encoders and loss function.
    
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
        **extra_args: Additional arguments for train function
        
    Returns:
        TrainingResult object
    """
    device = gpu_device if gpu_device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Training TSP model on {device} with {emsize} embedding size")
    
    # Use provided dataloader class or default to TSPDataLoader
    if priordataloader_class is None:
        priordataloader_class = TSPDataLoader
    
    print(f"Using dataloader: {priordataloader_class.__name__}")
    
    # Create single_eval_pos sampler
    single_eval_pos_sampler = get_uniform_single_eval_pos_sampler(seq_len, min_len=3)
    
    # Create custom loss for BMM with edge attention
    
        
    # Create the custom TSP criterion
    tsp_criterion = TSPAttentionCriterion()
    
    # Prepare extra_prior_kwargs_dict
    default_kwargs = {
        'num_nodes_range': num_nodes_range,
        'max_candidates': max_candidates
    }
    
    # Merge with any additional kwargs passed in
    if 'extra_prior_kwargs_dict' in extra_args:
        default_kwargs.update(extra_args['extra_prior_kwargs_dict'])
        extra_args = {k: v for k, v in extra_args.items() if k != 'extra_prior_kwargs_dict'}
    
    # Use train() function with the custom components
    result = train(
        priordataloader_class_or_get_batch=priordataloader_class,
        criterion=tsp_criterion,
        encoder_generator=tsp_graph_encoder_generator,
        y_encoder_generator=lambda num_features, emsize: tsp_tour_encoder_generator(num_features, emsize, max_nodes=max(num_nodes_range)),
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
        **extra_args
    )
    
    return result