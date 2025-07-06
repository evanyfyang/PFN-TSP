import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, TransformerEncoder
import torch.nn.functional as F
from .layer import TransformerEncoderLayer, _get_activation_fn
from .utils import SeqBN, bool_mask_to_att_mask
import time

# Attention-based pooling for edge embeddings
class GATPooling(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Linear transformation for attention features
        self.attention_linear = nn.Linear(hidden_size, hidden_size)
        
        # Compute scalar attention scores
        self.score_linear = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, batch_indices):
        """
        x: [num_edges, hidden_size] - edge embeddings
        batch_indices: [num_edges] - batch assignment for each edge
        """
        if x.size(0) == 0:
            return torch.zeros(1, self.hidden_size, device=x.device)
            
        batch_size = batch_indices.max().item() + 1 if batch_indices.numel() > 0 else 1
        
        # Compute attention features
        attention_features = self.attention_linear(x)  # [num_edges, hidden_size]
        attention_features = torch.relu(attention_features)
        attention_features = self.dropout(attention_features)
        
        # Compute scalar attention scores for each edge
        attention_scores = self.score_linear(attention_features)  # [num_edges, 1]
        attention_scores = attention_scores.squeeze(-1)  # [num_edges]
        
        # Compute softmax weights and weighted sum for each batch separately
        pooled_outputs = []
        for b in range(batch_size):
            batch_mask = (batch_indices == b)
            if batch_mask.sum() > 0:
                # Get current batch edge embeddings and attention scores
                batch_embeddings = x[batch_mask]  # [batch_edges, hidden_size]
                batch_scores = attention_scores[batch_mask]  # [batch_edges]
                
                # Compute softmax weights within batch
                batch_weights = F.softmax(batch_scores, dim=0)  # [batch_edges]
                
                # Weighted sum to get graph representation
                pooled = torch.sum(batch_embeddings * batch_weights.unsqueeze(-1), dim=0)  # [hidden_size]
                pooled_outputs.append(pooled)
            else:
                pooled_outputs.append(torch.zeros(self.hidden_size, device=x.device))
        
        return torch.stack(pooled_outputs, dim=0)  # [batch_size, hidden_size]

class TransformerModel(nn.Module):
    def __init__(self, encoder, ninp, nhead, nhid, nlayers, dropout=0.0, style_encoder=None, y_encoder=None,
                 pos_encoder=None, decoder_dict=None, input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False, num_global_att_tokens=0, full_attention=False,
                 all_layers_same_init=False, efficient_eval_masking=True, decoder_once_dict=None, return_all_outputs=False,
                 save_trainingset_representations=False):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layer_creator = lambda: TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation=activation,
                                                                pre_norm=pre_norm, recompute_attn=recompute_attn,
                                                                save_trainingset_representations=save_trainingset_representations)
        self.transformer_encoder = TransformerEncoder(encoder_layer_creator(), nlayers)\
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        self.ninp = ninp
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.pos_encoder = pos_encoder
        self.return_all_outputs = return_all_outputs

        # Add GAT pooling for edge embeddings
        self.gat_pooling = GATPooling(hidden_size=ninp, dropout=dropout)
        self.edge_concat_mlp = nn.Sequential(nn.Linear(2*ninp, ninp), nn.GELU(), nn.Linear(ninp, 1))

        self.edge_hard_mlp = nn.Sequential(nn.Linear(ninp, ninp), nn.GELU(), nn.Linear(ninp, 1))

        def make_decoder_dict(decoder_description_dict):
            if decoder_description_dict is None or len(decoder_description_dict) == 0:
                return None
            initialized_decoder_dict = {}
            for decoder_key in decoder_description_dict:
                decoder_model, decoder_n_out = decoder_description_dict[decoder_key]
                if decoder_model is None:
                    initialized_decoder_dict[decoder_key] = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, decoder_n_out))
                else:
                    initialized_decoder_dict[decoder_key] = decoder_model(ninp, nhid, decoder_n_out)
                print('Initialized decoder for', decoder_key, 'with', decoder_description_dict[decoder_key], ' and nout', decoder_n_out)
            return torch.nn.ModuleDict(initialized_decoder_dict)

        self.decoder_dict = make_decoder_dict(decoder_dict)
        self.decoder_dict_once = make_decoder_dict(decoder_once_dict)

        # N(0,1) is the initialization as the default of nn.Embedding
        self.decoder_dict_once_embeddings = torch.nn.Parameter(torch.randn((len(self.decoder_dict_once), 1, ninp))) if self.decoder_dict_once is not None else None
            #nn.Embedding(len(self.decoder_dict.keys()), nhid)
        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.style_encoder = style_encoder
        self.init_method = init_method
        if num_global_att_tokens is not None:
            assert not full_attention
        self.global_att_embeddings = nn.Embedding(num_global_att_tokens, ninp) if num_global_att_tokens else None
        self.full_attention = full_attention
        self.efficient_eval_masking = efficient_eval_masking

        self.nhid = nhid

        self.edge_mlp = nn.Sequential(nn.Linear(nhid, nhid), nn.GELU())

        self.init_weights()

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('efficient_eval_masking', False)
        if not hasattr(self, 'decoder_dict_once'):
            self.__dict__.setdefault('decoder_dict_once', None)
        if hasattr(self, 'decoder') and not hasattr(self, 'decoder_dict'):
            self.add_module('decoder_dict', nn.ModuleDict({'standard': self.decoder}))
        self.__dict__.setdefault('return_all_outputs', False)

        def add_approximate_false(module):
            if isinstance(module, nn.GELU):
                module.__dict__.setdefault('approximate', 'none')

        self.apply(add_approximate_false)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        train_size = sz-query_size
        mask = torch.zeros(sz,sz) == 0
        mask[:,train_size:].zero_()
        mask |= torch.eye(sz) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_query_matrix(num_global_att_tokens, seq_len, num_query_tokens):
        train_size = seq_len + num_global_att_tokens - num_query_tokens
        sz = seq_len + num_global_att_tokens
        mask = torch.zeros(num_query_tokens, sz) == 0
        mask[:,train_size:].zero_()
        mask[:,train_size:] |= torch.eye(num_query_tokens) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_trainset_matrix(num_global_att_tokens, seq_len, num_query_tokens):
        train_size = seq_len + num_global_att_tokens - num_query_tokens
        trainset_size = seq_len - num_query_tokens
        mask = torch.zeros(trainset_size, num_global_att_tokens) == 0
        #mask[:,num_global_att_tokens:].zero_()
        #mask[:,num_global_att_tokens:] |= torch.eye(trainset_size) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_globaltokens_matrix(num_global_att_tokens, seq_len, num_query_tokens):
        mask = torch.zeros(num_global_att_tokens, num_global_att_tokens+seq_len-num_query_tokens) == 0
        return bool_mask_to_att_mask(mask)

    def init_weights(self):
        initrange = 1.
        # if isinstance(self.encoder,EmbeddingEncoder):
        #    self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        if self.init_method is not None:
            self.apply(self.init_method)
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
            for attn in attns:
                nn.init.zeros_(attn.out_proj.weight)
                nn.init.zeros_(attn.out_proj.bias)

    def forward(self, *args, **kwargs):
        """
        This will perform a forward-pass (possibly recording gradients) of the model.
        We have multiple interfaces we support with this model:

        model(train_x, train_y, test_x, src_mask=None, style=None, only_return_standard_out=True)
        model((x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
        model((style,x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
        """
        if len(args) == 3:
            # case model(train_x, train_y, test_x, src_mask=None, style=None, only_return_standard_out=True)
            assert all(kwarg in {'src_mask', 'style', 'only_return_standard_out', 'candidate_info'} for kwarg in kwargs.keys()), \
                f"Unrecognized keyword argument in kwargs: {set(kwargs.keys()) - {'src_mask', 'style', 'only_return_standard_out', 'candidate_info'}}"
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=0)
            style = kwargs.pop('style', None)
            return self._forward((style, x, args[1]), single_eval_pos=len(args[0]), **kwargs)
        elif len(args) == 1 and isinstance(args, tuple):
            # case model((x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
            # case model((style,x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
            assert all(kwarg in {'src_mask', 'single_eval_pos', 'only_return_standard_out', 'candidate_info'} for kwarg in kwargs.keys()), \
                f"Unrecognized keyword argument in kwargs: {set(kwargs.keys()) - {'src_mask', 'single_eval_pos', 'only_return_standard_out', 'candidate_info'}}"
            return self._forward(*args, **kwargs)

    def _forward(self, src, src_mask=None, single_eval_pos=None, only_return_standard_out=True, candidate_info=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'

        if len(src) == 2: # (x,y) and no style
            src = (None,) + src

        style_src, x_src, y_src = src

        if single_eval_pos is None:
            single_eval_pos = x_src.shape[0]

        # Check if encoder supports unified encoding (has use_unified_encoding attribute)
        encoder_supports_unified = hasattr(self.encoder, 'use_unified_encoding') and self.encoder.use_unified_encoding
        
        # Pass candidate_info and y_src to encoder if it supports them
        if hasattr(self.encoder, 'forward') and 'candidate_info' in self.encoder.forward.__code__.co_varnames:
            encoder_kwargs = {'candidate_info': candidate_info}
            
            # Add y parameter if encoder supports unified encoding
            if encoder_supports_unified and 'y' in self.encoder.forward.__code__.co_varnames:
                encoder_kwargs['y'] = y_src
                
            if 'gat_pooling' in self.encoder.forward.__code__.co_varnames:
                encoder_kwargs['gat_pooling'] = self.gat_pooling
            
            # Add single_eval_pos for SharedBasisFiLM mode
            if 'single_eval_pos' in self.encoder.forward.__code__.co_varnames:
                encoder_kwargs['single_eval_pos'] = single_eval_pos
                
            x_encoder_output = self.encoder(x_src, **encoder_kwargs)
        else:
            x_encoder_output = self.encoder(x_src)

        edge_info = None
        
        if isinstance(x_encoder_output, dict) and 'node_embeddings' in x_encoder_output:
            edge_info = x_encoder_output.get('edge_info')
            x_encoded = x_encoder_output['node_embeddings']
            
            # Check for direct predictions mode (SharedBasisFiLM)
            if x_encoder_output.get('direct_predictions', False):
                # For SharedBasisFiLM mode, encoder has already done edge prediction
                # Just return the pre-computed edge predictions and edge_info
                edge_predictions = x_encoder_output.get('edge_predictions')
                if edge_predictions is not None:
                    return edge_predictions, edge_info
                else:
                    # Fallback if no edge predictions provided
                    seq_eval_len = x_src.shape[0] - single_eval_pos
                    batch_size = x_src.shape[1]
                    dummy_predictions = torch.zeros(seq_eval_len, batch_size, 1, device=x_src.device)
                    return dummy_predictions, edge_info
        else:
            x_encoded = x_encoder_output

        if self.decoder_dict_once is not None:
            x_encoded = torch.cat([x_encoded, self.decoder_dict_once_embeddings.repeat(1, x_encoded.shape[1], 1)], dim=0)

        # Use y_encoder only if not using unified encoding
        if y_src is not None and self.y_encoder is not None and not encoder_supports_unified:
            y_shape_adjusted = y_src.unsqueeze(-1) if len(y_src.shape) < len(x_encoded.shape) else y_src
            if edge_info is not None:
                if len(edge_info) == 7:
                    edge_emb, edge_index, batch, position_tensor, node_offset_map, edge_counts, _ = [edge_info[key] for key in edge_info]
                else:
                    edge_emb, edge_index, batch, position_tensor, node_offset_map, edge_counts = edge_info
                y_encoded = self.y_encoder(
                    y_shape_adjusted,
                    edge_emb=edge_emb,
                    edge_index=edge_index,
                    batch=batch,
                    position=position_tensor,
                    node_offset_map=node_offset_map,
                    gat_pooling=self.gat_pooling
                )
            else:
                y_encoded = self.y_encoder(y_shape_adjusted)
        else:
            y_encoded = None

        if self.style_encoder:
            assert style_src is not None, 'style_src must be given if style_encoder is used'
            style_src = self.style_encoder(style_src).unsqueeze(0)
        else:
            style_src = torch.tensor([], device=x_encoded.device)
        global_src = torch.tensor([], device=x_encoded.device) if self.global_att_embeddings is None else \
            self.global_att_embeddings.weight.unsqueeze(1).repeat(1, x_encoded.shape[1], 1)

        if src_mask is not None:
            assert self.global_att_embeddings is None or isinstance(src_mask, tuple)

        if src_mask is None:
            if self.global_att_embeddings is None:
                full_len = len(x_encoded) + len(style_src)
                if self.full_attention:
                    src_mask = bool_mask_to_att_mask(torch.ones((full_len, full_len), dtype=torch.bool)).to(x_encoded.device)
                elif self.efficient_eval_masking:
                    src_mask = single_eval_pos + len(style_src)
                else:
                    src_mask = self.generate_D_q_matrix(full_len, len(x_encoded) - single_eval_pos).to(x_encoded.device)
            else:
                src_mask_args = (self.global_att_embeddings.num_embeddings,
                                 len(x_encoded) + len(style_src),
                                 len(x_encoded) + len(style_src) - single_eval_pos)
                src_mask = (self.generate_global_att_globaltokens_matrix(*src_mask_args).to(x_encoded.device),
                            self.generate_global_att_trainset_matrix(*src_mask_args).to(x_encoded.device),
                            self.generate_global_att_query_matrix(*src_mask_args).to(x_encoded.device))

        train_x = x_encoded[:single_eval_pos]
        if y_encoded is not None:
            train_x = train_x + y_encoded[:single_eval_pos]
        src = torch.cat([global_src, style_src, train_x, x_encoded[single_eval_pos:]], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        if self.pos_encoder is not None:
            src = self.pos_encoder(src)
            
        output = self.transformer_encoder(src, src_mask)
        num_prefix_positions = len(style_src)+(self.global_att_embeddings.num_embeddings if self.global_att_embeddings else 0)
        if self.return_all_outputs:
            out_range_start = num_prefix_positions
        else:
            out_range_start = single_eval_pos + num_prefix_positions

        out_range_end = -len(self.decoder_dict_once_embeddings) if self.decoder_dict_once is not None else None

        output = output[out_range_start:out_range_end]
        
        # Handle unified dict format with eval_infos
        eval_infos = edge_info.get('eval_infos', [])
        if eval_infos:
            # Use structural edge information to build outputs
            return self._build_output_from_eval_infos(output, eval_infos, single_eval_pos, x_src)
        else:
            # No evaluation info available
            return output, None

    def _build_output_from_eval_infos(self, output, eval_infos, single_eval_pos, x_src):
        """
        Build output using structural evaluation information from TSPGraphEncoder.
        This method processes the edge structural information and applies edge prediction MLPs.
        
        Args:
            output: Transformer output [seq_eval_len, batch_size, emsize]
            eval_infos: List of evaluation info dictionaries from TSPGraphEncoder (structural only)
            single_eval_pos: Position where evaluation starts
            x_src: Original input tensor for shape information
            
        Returns:
            edge_predictions: Tensor [seq_eval_len, batch_size, max_edges]
            edge_info_tuple: Tuple containing edge information for compatibility
        """
        seq_eval_len, batch_size, emsize = output.shape
        
        # Group eval_infos by position and batch
        eval_info_dict = {}
        max_edges = 0
        
        for eval_info in eval_infos:
            pos = eval_info['pos']
            batch = eval_info['batch']
            edge_index = eval_info['edge_index']
            
            # Calculate position in output tensor
            if single_eval_pos is not None:
                output_pos = pos - single_eval_pos
            else:
                output_pos = pos
            
            if output_pos >= 0 and output_pos < seq_eval_len:
                if output_pos not in eval_info_dict:
                    eval_info_dict[output_pos] = {}
                eval_info_dict[output_pos][batch] = eval_info
                
                # Track maximum number of edges
                num_edges = edge_index.size(1) if edge_index.size(1) > 0 else 0
                max_edges = max(max_edges, num_edges)
        
        # Create output tensors
        edge_values_padded = torch.zeros(seq_eval_len, batch_size, max_edges, device=output.device)
        edge_index_list = []
        edge_counts = []
        
        # Process each evaluation position and batch
        for pos_idx in range(seq_eval_len):
            for batch_idx in range(batch_size):
                if pos_idx in eval_info_dict and batch_idx in eval_info_dict[pos_idx]:
                    eval_info = eval_info_dict[pos_idx][batch_idx]
                    edge_index = eval_info['edge_index']
                    num_edges = edge_index.size(1)
                    
                    if num_edges > 0:
                        # Apply edge prediction MLP to the transformer output
                        transformer_output = output[pos_idx, batch_idx].unsqueeze(0)  # [1, emsize]
                        
                        # Create edge embeddings by applying edge MLP to transformer output
                        # This simulates the edge embedding processing
                        edge_embs = transformer_output.repeat(num_edges, 1)  # [num_edges, emsize]
                        processed_embs = self.edge_mlp(edge_embs)  # [num_edges, emsize]
                        
                        # Apply edge predictor (similar to the existing logic)
                        transformer_expanded = transformer_output.repeat(num_edges, 1)  # [num_edges, emsize]
                        edge_values = self.edge_hard_mlp(transformer_expanded * processed_embs).squeeze(-1)
                        
                        # Store edge values (padded)
                        edge_values_padded[pos_idx, batch_idx, :num_edges] = edge_values
                        
                        # Store edge index and count information
                        edge_index_list.append(edge_index.t())  # Convert to [num_edges, 2] format
                        edge_counts.append(num_edges)
                    else:
                        # No edges for this instance
                        edge_index_list.append(torch.empty((0, 2), dtype=torch.long, device=output.device))
                        edge_counts.append(0)
                else:
                    # No evaluation info for this position/batch
                    edge_index_list.append(torch.empty((0, 2), dtype=torch.long, device=output.device))
                    edge_counts.append(0)
        
        # Build proper node_offset_map for loss calculation
        # The eval_infos should contain the correct node_offset_map from the encoder
        node_offset_map = {}
        for eval_info in eval_infos:
            pos, batch = eval_info['pos'], eval_info['batch']
            valid_indices = eval_info.get('valid_indices', [])
            node_offset_info = eval_info.get('node_offset_map', {})
            
            # If node_offset_map is available in eval_info, use it directly
            if node_offset_info:
                node_offset_map.update(node_offset_info)
            else:
                # Fallback: create mapping based on edge indices
                # This assumes edge_index contains global node indices
                if valid_indices is not None:
                    edge_index = eval_info.get('edge_index')
                    if edge_index is not None and edge_index.size(1) > 0:
                        # Extract unique global node indices from edges
                        unique_global_nodes = torch.unique(edge_index).cpu().tolist()
                        for i, valid_idx in enumerate(valid_indices):
                            if i < len(unique_global_nodes):
                                node_offset_map[(pos, batch, valid_idx.item())] = unique_global_nodes[i]
        
        ret_info = (edge_index_list, node_offset_map, edge_counts)
        
        return edge_values_padded, ret_info

    @torch.no_grad()
    def init_from_small_model(self, small_model):
        assert isinstance(self.decoder, nn.Linear) and isinstance(self.encoder, (nn.Linear, nn.Sequential)) \
               and isinstance(self.y_encoder, (nn.Linear, nn.Sequential))

        def set_encoder_weights(my_encoder, small_model_encoder):
            my_encoder_linear, small_encoder_linear = (my_encoder, small_model_encoder) \
                if isinstance(my_encoder, nn.Linear) else (my_encoder[-1], small_model_encoder[-1])
            small_in_dim = small_encoder_linear.out_features
            my_encoder_linear.weight.zero_()
            my_encoder_linear.bias.zero_()
            my_encoder_linear.weight[:small_in_dim] = small_encoder_linear.weight
            my_encoder_linear.bias[:small_in_dim] = small_encoder_linear.bias

        set_encoder_weights(self.encoder, small_model.encoder)
        set_encoder_weights(self.y_encoder, small_model.y_encoder)

        small_in_dim = small_model.decoder.in_features

        self.decoder.weight[:, :small_in_dim] = small_model.decoder.weight
        self.decoder.bias = small_model.decoder.bias

        for my_layer, small_layer in zip(self.transformer_encoder.layers, small_model.transformer_encoder.layers):
            small_hid_dim = small_layer.linear1.out_features
            my_in_dim = my_layer.linear1.in_features

            # packed along q,k,v order in first dim
            my_in_proj_w = my_layer.self_attn.in_proj_weight
            small_in_proj_w = small_layer.self_attn.in_proj_weight

            my_in_proj_w.view(3, my_in_dim, my_in_dim)[:, :small_in_dim, :small_in_dim] = small_in_proj_w.view(3,
                                                                                                               small_in_dim,
                                                                                                               small_in_dim)
            my_layer.self_attn.in_proj_bias.view(3, my_in_dim)[:,
            :small_in_dim] = small_layer.self_attn.in_proj_bias.view(3, small_in_dim)

            my_layer.self_attn.out_proj.weight[:small_in_dim, :small_in_dim] = small_layer.self_attn.out_proj.weight
            my_layer.self_attn.out_proj.bias[:small_in_dim] = small_layer.self_attn.out_proj.bias

            my_layer.linear1.weight[:small_hid_dim, :small_in_dim] = small_layer.linear1.weight
            my_layer.linear1.bias[:small_hid_dim] = small_layer.linear1.bias

            my_layer.linear2.weight[:small_in_dim, :small_hid_dim] = small_layer.linear2.weight
            my_layer.linear2.bias[:small_in_dim] = small_layer.linear2.bias

            my_layer.norm1.weight[:small_in_dim] = math.sqrt(small_in_dim / my_in_dim) * small_layer.norm1.weight
            my_layer.norm2.weight[:small_in_dim] = math.sqrt(small_in_dim / my_in_dim) * small_layer.norm2.weight

            my_layer.norm1.bias[:small_in_dim] = small_layer.norm1.bias
            my_layer.norm2.bias[:small_in_dim] = small_layer.norm2.bias

class TransformerEncoderDiffInit(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer_creator: a function generating objects of TransformerEncoderLayer class without args (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer_creator, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
