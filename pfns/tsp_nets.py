import torch
from copy import copy, deepcopy
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, BatchNorm
import torch.nn.functional as F
from torch_geometric.utils import softmax

from torch_scatter import scatter_mean, scatter_add


class SharedBasisFiLMAttentionGNN(nn.Module):
    """
    GNN with shared basis decomposition, FiLM modulation by edge type,
    simple instance offset modulation, and integrated per-layer
    self-attention over instance-center nodes.
    """
    def __init__(self, depth, feats, units, act_fn, agg_fn,
                 num_instances, num_relations, num_bases, num_heads):
        super().__init__()
        self.depth = depth
        self.units = units
        self.act_fn = act_fn
        self.agg_fn = agg_fn
        self.num_bases = num_bases

        # initial projections
        self.v_lin0 = nn.Linear(feats, units)
        self.e_lin0 = nn.Linear(3, units)  # Support up to 3-dimensional edge features

        # shared basis MLPs per layer
        self.basis_mlps = nn.ModuleList([
            nn.ModuleList([nn.Linear(units, units) for _ in range(num_bases)])
            for _ in range(depth)
        ])
        self.coeffs = nn.Parameter(torch.Tensor(depth, num_bases))
        nn.init.xavier_uniform_(self.coeffs)

        # simple instance embedding for offset modulation
        self.inst_emb = nn.Embedding(num_instances, units)
        nn.init.xavier_uniform_(self.inst_emb.weight)

        # embedding and MLP for edge type FiLM
        self.type_emb = nn.Embedding(num_relations, units)
        nn.init.xavier_uniform_(self.type_emb.weight)
        self.type_mlp = nn.Sequential(
            nn.Linear(units, units), act_fn,
            nn.Linear(units, 2*units)
        )

        # per-layer self-attention over instance centers
        self.layer_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=units, num_heads=num_heads)
            for _ in range(depth)
        ])

        # batch norms for node and edge updates
        self.v_bns = nn.ModuleList([BatchNorm(units) for _ in range(depth)])
        self.e_bns = nn.ModuleList([BatchNorm(units) for _ in range(depth)])

    def forward(self, x, edge_index, edge_attr,
                type_index, inst_index, instance_nodes):
        """
        x               : [N, feats]
        edge_index      : [2, E]
        edge_attr       : [E, 1]
        type_index      : [E]  0=graph,1=sol (or more types)
        inst_index      : [E]  instance ID per edge
        instance_nodes  : list of node indices for self-attention
        """
        # initial node & edge embeddings
        x = self.act_fn(self.v_lin0(x))          # h^0
        w = self.act_fn(self.e_lin0(edge_attr))  # w^0

        for l in range(self.depth):
            x0, w0 = x, w

            # shared basis message: [N, units]
            basis_outs = torch.stack([
                self.basis_mlps[l][b](x0) for b in range(self.num_bases)
            ], dim=0)  # [B, N, units]
            shared_msg = torch.einsum('bNv,b->Nv', basis_outs, self.coeffs[l])

            # FiLM by edge type
            type_feat = self.type_emb(type_index)   # [E, units]
            film = self.type_mlp(type_feat)         # [E, 2*units]
            gamma, beta = film.chunk(2, dim=1)      # each [E, units]

            # simple instance offset
            inst_offset = self.inst_emb(inst_index) # [E, units]

            # prepare messages per edge
            src = edge_index[1]
            msg = gamma * shared_msg[src] + beta + inst_offset

            # gated aggregation
            gates = torch.sigmoid(w0)               # [E, units]
            agg = self.agg_fn(gates * msg,
                              edge_index[0],
                              size=x0.size(0))      # [N, units]

            # node & edge updates
            x = x0 + self.act_fn(self.v_bns[l](agg))
            w = w0 + self.act_fn(
                self.e_bns[l](w0 + x[edge_index[0]] + x[edge_index[1]])
            )

            # instance-center self-attention
            H_I = x[instance_nodes]                 # [I, units]
            H_I_seq = H_I.unsqueeze(1)              # [I,1,units]
            attn_out, _ = self.layer_attn[l](H_I_seq, H_I_seq, H_I_seq)
            x = x.clone()
            x[instance_nodes] = attn_out.squeeze(1).to(x.dtype)

        return x, w
    
class InstanceAwareHypergraphGNN(nn.Module):
    """
    Instance-aware Hypergraph GNN with FiLM modulation and instance information exchange.
    
    This model implements the mathematical formulation provided:
    - Graph construction with city nodes V and instance center nodes I
    - Edge features with [distance, is_center] 
    - FiLM modulation based on instance readout
    - Hypergraph GCN for instance information exchange
    - Solution edge prediction
    """
    def __init__(self, depth, feats, units, act_fn, agg_fn, num_instances, num_heads=8, message_agg='mean', prediction_mode='dot_product', use_residual_norm=False, aggregation_mode: str = 'edge_first', share_film_across_layers: bool = True):
        super().__init__()
        self.depth = depth
        self.units = units
        self.act_fn = act_fn
        self.agg_fn = agg_fn
        self.num_instances = num_instances
        self.message_agg = message_agg  # 'mean', 'sum', 'max', etc.
        self.prediction_mode = prediction_mode  # 'dot_product' or 'mlp_concat'
        self.use_residual_norm = use_residual_norm  # Control residual connections and LayerNorm
        self.aggregation_mode = aggregation_mode  # 'edge_first' (default) or 'instance_first'
        self.share_film_across_layers = share_film_across_layers

        self.v_lin0 = nn.Linear(feats, units)  # W_v for nodes
        self.e_lin0 = nn.Linear(2, units)      # W_e for edges [distance, is_center]
        
        self.dist_embedder = nn.Linear(1, units)  # Distance to embedding (same as w_emb)
        self.s_uvi_embedder = nn.Linear(1, units)  # s_uvi (0/1) to embedding (same as w_emb)
        self.ctx_uvi_embedder = nn.Linear(1, units)  # context/target indicator (0=target,1=context)
        self.target_embedder = nn.Linear(1, units)  # target/context embedding (0=context, 1=target)

        self.message_mlps = nn.ModuleList([
            nn.Linear(units, units) for _ in range(depth)
        ])
        
        self.gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(units + units, units // 2),  # [z_i, w_e + s_uvi_emb]
                act_fn,
                nn.Linear(units // 2, units)
            ) for _ in range(depth)
        ])
        # Attention scorer for GAT-style pooling over (edge, instance) pairs
        self.att_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(units + units, units // 2),
                act_fn,
                nn.Linear(units // 2, 1)
            ) for _ in range(depth)
        ])
        # Intra-instance attention scorer (per (inst, dst) group)
        self.intra_att_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(units + units, units // 2),
                act_fn,
                nn.Linear(units // 2, 1)
            ) for _ in range(depth)
        ])
        
        self.shared_film_mlp = nn.Sequential(
            nn.Linear(units, units),  # [z_i + dist_emb + s_uvi_emb]
            act_fn,
            nn.Linear(units, 2 * units)   # output [gamma, beta]
        )

        # Instance-shared FiLM (per instance, shared for all nodes/edges in that instance)
        # Inputs: [z_i, ctx_i]
        film_input_dim = units + 1
        def make_film_block():
            return nn.Sequential(
                nn.Linear(film_input_dim, units),
                act_fn,
                nn.Linear(units, 2 * units)
            )
        if share_film_across_layers:
            self.instance_node_film = make_film_block()
            self.instance_edge_film = make_film_block()
        else:
            self.instance_node_film = nn.ModuleList([make_film_block() for _ in range(depth)])
            self.instance_edge_film = nn.ModuleList([make_film_block() for _ in range(depth)])

        # Edge-conditional FiLM delta (per (edge, instance) pair) to refine instance-shared FiLM
        # Inputs concat: [z_i (units), w_e (units), dist_emb (units), s_uvi_emb (units), ctx_uvi_emb (units)] => 5*units
        def make_edge_cond_film_block():
            return nn.Sequential(
                nn.Linear(5 * units, units),
                act_fn,
                nn.Linear(units, 2 * units)
            )
        self.edge_cond_film_mlps = nn.ModuleList([make_edge_cond_film_block() for _ in range(depth)])
        
        self.node_update_mlps = nn.ModuleList([
            nn.Linear(units, units) for _ in range(depth)
        ])
        
        self.edge_update_mlps = nn.ModuleList([
            nn.Linear(units, units) for _ in range(depth)
        ])
        self.edge_node_mlps = nn.ModuleList([
            nn.Linear(units, units) for _ in range(depth)
        ])
        
        self.instance_self_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=units, num_heads=num_heads, batch_first=True)
            for _ in range(depth)
        ])
        
        if prediction_mode == 'dot_product':
            self.edge_mlp = nn.Sequential(
                nn.Linear(4 * units, units // 2),  # [x_src, x_dst, w_edges, center_features] = 4*units
                act_fn,
                nn.Linear(units // 2, units)
            )
            self.instance_mlp = nn.Sequential(
                nn.Linear(units, units // 2),
                act_fn,
                nn.Linear(units // 2, units)
            )
            
            for layer in self.edge_mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=2.0)  # Larger gain for edge MLP
                    nn.init.zeros_(layer.bias)
            
            for layer in self.instance_mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=2.0)  # Larger gain for instance MLP
                    nn.init.zeros_(layer.bias)

            # Learnable temperature for stabilized dot-product logits
            self.dot_temperature = nn.Parameter(torch.tensor(5.0))
        else:  # mlp_concat
            self.edge_projection = nn.Linear(5 * units, units)  # Project 5*units to units first
            self.output_mlp = nn.Sequential(
                nn.Linear(units, units // 2),  # Reduced size
                act_fn,
                nn.Linear(units // 2, 1)
            )

        # Per-instance EmbNet baseline components (reused when TSP_PER_INSTANCE_EMBNET=1)
        self.per_inst_embnet = EmbNet(depth=self.depth, feats=2, units=self.units, act_fn=self.act_fn, agg_fn=self.agg_fn)
        self.per_inst_head = nn.Sequential(
            nn.Linear(self.units, self.units // 2),
            act_fn,
            nn.Linear(self.units // 2, 1)
            )
        
        if use_residual_norm:
            self.v_lns = nn.ModuleList([nn.LayerNorm(units) for _ in range(depth)])  # LayerNorm for nodes
            self.e_lns = nn.ModuleList([nn.LayerNorm(units) for _ in range(depth)])  # LayerNorm for edges
            self.z_lns = nn.ModuleList([nn.LayerNorm(units) for _ in range(depth)])  # LayerNorm for instances
        else:
            self.v_lns = None
            self.e_lns = None
            self.z_lns = None
        
        self.v_bns = nn.ModuleList([BatchNorm(units) for _ in range(depth)])
        self.e_bns = nn.ModuleList([BatchNorm(units) for _ in range(depth)])
        self.inst_bns = nn.ModuleList([BatchNorm(units) for _ in range(depth)])

    def forward(self, x, edge_index, edge_attr, instance_mapping, edge_to_instances=None, instance_to_edges=None, s_uvi_matrix=None, ctx_uvi_matrix=None, ctx_i_vector=None, single_eval_pos=None):
        """
        Forward pass for one-to-many node/edge-instance mapping.
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        num_instances = s_uvi_matrix.size(1) if s_uvi_matrix is not None else len(instance_mapping)
        units = self.units
        
        # Optional pure per-instance EmbNet prediction path (no internal hypergraph propagation)
        import os
        if os.environ.get('TSP_PER_INSTANCE_EMBNET', '0') == '1' and instance_to_edges is not None:
            # Initialize edge predictions
            edge_predictions = torch.zeros(num_edges, num_instances, device=x.device)
            # For each instance, build subgraph and run EmbNet
            for inst_id in range(num_instances):
                edge_ids = instance_to_edges.get(inst_id, [])
                if not edge_ids:
                    continue
                edge_ids_tensor = torch.tensor(edge_ids, dtype=torch.long, device=x.device)
                # Gather nodes used in these edges
                nodes_used = torch.unique(torch.cat([
                    edge_index[0, edge_ids_tensor], edge_index[1, edge_ids_tensor]
                ], dim=0))
                if nodes_used.numel() < 2:
                    continue
                # Local reindex
                node_id_map = torch.empty(num_nodes, dtype=torch.long, device=x.device).fill_(-1)
                node_id_map[nodes_used] = torch.arange(nodes_used.size(0), device=x.device)
                local_edges = node_id_map[edge_index[:, edge_ids_tensor]]  # [2, E_i]
                # Subgraph node features and edge attrs
                x_sub = x[nodes_used]  # [N_i, 2]
                # Compute distances for subgraph edges
                src_l = local_edges[1]
                dst_l = local_edges[0]
                dist_l = torch.norm(x_sub[src_l] - x_sub[dst_l], dim=1, keepdim=True)
                is_center_l = torch.zeros_like(dist_l)
                pad_zero = torch.zeros_like(dist_l)
                edge_attr_sub = torch.cat([dist_l, is_center_l, pad_zero], dim=1)  # [E_i, 3]
                # Run EmbNet to get edge embeddings
                # Override aggregation to a stable index_add for this subgraph
                def _subgraph_agg(messages, dst_idx, N=x_sub.size(0), U=self.units, dev=x.device):
                    out = torch.zeros(N, U, device=dev)
                    return out.index_add(0, dst_idx, messages)
                self.per_inst_embnet.agg_fn = _subgraph_agg
                w_sub = self.per_inst_embnet(x_sub, local_edges, edge_attr_sub)  # [E_i, units]
                # Predict logits with a trainable head per edge
                logits = self.per_inst_head(w_sub).squeeze(-1)  # [E_i]
                edge_predictions[edge_ids_tensor, inst_id] = logits
            # Return placeholders for embeddings
            x_emb = self.act_fn(self.v_lin0(x))
            w_emb = self.act_fn(self.e_lin0(edge_attr))
            z_tensor_clean = torch.zeros(num_instances, units, device=x.device)
            return x_emb, w_emb, z_tensor_clean, edge_predictions
        
        x_emb = self.act_fn(self.v_lin0(x))  # [N, units]
        w_emb = self.act_fn(self.e_lin0(edge_attr))  # [E, units]
        
        # Build many-to-one mapping from nodes to instances (exclude center nodes)
        instance_tensor = torch.full((num_nodes,), -1, dtype=torch.long, device=x.device)  # kept for compatibility
        inst_ids_list = []
        node_ids_list = []
        for inst_id, node_indices in instance_mapping.items():
            if len(node_indices) > 0:
                # mark for compatibility (optional, not used for z)
                instance_tensor[node_indices] = inst_id
                city_nodes = node_indices[:-1] if len(node_indices) > 1 else node_indices
                for nid in city_nodes:
                    inst_ids_list.append(inst_id)
                    node_ids_list.append(nid)

        if len(node_ids_list) > 0:
            inst_ids_concat = torch.tensor(inst_ids_list, dtype=torch.long, device=x.device)
            node_ids_concat = torch.tensor(node_ids_list, dtype=torch.long, device=x.device)
        else:
            inst_ids_concat = torch.empty(0, dtype=torch.long, device=x.device)
            node_ids_concat = torch.empty(0, dtype=torch.long, device=x.device)
        
        z_tensor = torch.zeros(num_instances, units, device=x.device)
        if node_ids_concat.numel() > 0:
            z_tensor = scatter_mean(
                x_emb[node_ids_concat],
                inst_ids_concat,
                dim=0, 
                dim_size=num_instances
            )
        valid_mask = node_ids_concat.numel() > 0
        
        if instance_mapping:
            max_inst_id = max(instance_mapping.keys()) if instance_mapping else 0
            num_total_instances = max_inst_id + 1
            
            if single_eval_pos is not None:
                seq_len = single_eval_pos + 1  # Minimum seq_len
                batch_size = num_total_instances // seq_len if seq_len > 0 else 1
            else:
                seq_len = int(num_total_instances ** 0.5)
                batch_size = num_total_instances // seq_len if seq_len > 0 else 1
        else:
            seq_len = 1
            batch_size = num_instances
        
        target_labels = torch.zeros(num_instances, device=x.device)
        if single_eval_pos is not None:
            pos_indices = torch.arange(num_instances, device=x.device) % seq_len
            target_labels = (pos_indices > single_eval_pos).float()
        
        target_emb = self.target_embedder(target_labels.unsqueeze(1))  # [num_instances, units]
        z_tensor_clean = z_tensor.clone()  
        enhanced_z_tensor = z_tensor + target_emb  
        
        # Precompute edge-instance pairing tensors once (static across layers)
        if edge_to_instances is not None:
            edge_inst_pairs = []
            for e in range(num_edges):
                inst_list = edge_to_instances.get(e, []) if isinstance(edge_to_instances, dict) else []
                for inst_id in inst_list:
                            edge_inst_pairs.append((e, inst_id))
            if len(edge_inst_pairs) > 0:
                edge_ids_all, inst_ids_all = zip(*edge_inst_pairs)
                edge_ids_all = torch.tensor(edge_ids_all, device=x.device)
                inst_ids_all = torch.tensor(inst_ids_all, device=x.device)
            else:
                edge_ids_all = torch.empty(0, dtype=torch.long, device=x.device)
                inst_ids_all = torch.empty(0, dtype=torch.long, device=x.device)
        else:
            edge_ids_all = torch.empty(0, dtype=torch.long, device=x.device)
            inst_ids_all = torch.empty(0, dtype=torch.long, device=x.device)

        src_all = edge_index[1, edge_ids_all] if edge_ids_all.numel() > 0 else torch.empty(0, dtype=torch.long, device=x.device)
        dst_all = edge_index[0, edge_ids_all] if edge_ids_all.numel() > 0 else torch.empty(0, dtype=torch.long, device=x.device)
        dist_all = edge_attr[edge_ids_all, 0] if edge_ids_all.numel() > 0 else torch.empty(0, device=x.device)
        s_uvi_all = s_uvi_matrix[edge_ids_all, inst_ids_all] if (edge_ids_all.numel() > 0 and s_uvi_matrix is not None) else torch.empty(0, device=x.device)
        ctx_uvi_all = ctx_uvi_matrix[edge_ids_all, inst_ids_all] if (edge_ids_all.numel() > 0 and ctx_uvi_matrix is not None) else torch.empty(0, device=x.device)

        for l in range(self.depth):
            if torch.jit.is_scripting():
                debug_enabled = False
            else:
                import os
                debug_enabled = os.environ.get('TSP_DEBUG_GUARDS', '0') == '1'
                enable_film = os.environ.get('TSP_ENABLE_FILM', '1') == '1'
                step1_use_z = os.environ.get('TSP_STEP1_USE_Z', '1') == '1'

            # Step 1: Intra-instance aggregation (gated mean over incoming edges per instance)
            if self.aggregation_mode == 'instance_first':
                messages_all = torch.zeros(num_edges, units, device=x.device)  # will be produced in Step2

                # Step1a: per-(edge,inst) gated messages (no FiLM, no z_i dependence)
                if edge_ids_all.numel() > 0:
                    msg0 = self.message_mlps[l](x_emb[src_all])
                    dist_emb0 = self.dist_embedder(dist_all.unsqueeze(1))
                    s_emb0 = self.s_uvi_embedder(s_uvi_all.unsqueeze(1)) if s_uvi_all.numel() > 0 else torch.zeros_like(dist_emb0)
                    ctxu_emb0 = self.ctx_uvi_embedder(ctx_uvi_all.unsqueeze(1)) if ctx_uvi_all.numel() > 0 else torch.zeros_like(dist_emb0)
                    suvi_ctx0 = s_emb0 + ctxu_emb0
                    w_e0 = w_emb[edge_ids_all]
                    # Use z_i for gate weighting in Step1 (no FiLM)
                    z_i0 = z_tensor[inst_ids_all] if step1_use_z else torch.zeros_like(z_tensor[inst_ids_all])
                    gate_input0 = torch.cat([z_i0, w_e0 + suvi_ctx0], dim=1)
                    gate0 = 1.0 + torch.tanh(self.gate_mlps[l](gate_input0))
                    mod_msg0 = gate0 * msg0

                    # Step1b: build per-instance node deltas with attention within (inst,dst) group
                    key_node = inst_ids_all * num_nodes + dst_all
                    # group by unique (inst,dst)
                    unique_pair, pair_idx, pair_counts = torch.unique(
                        key_node, return_inverse=True, return_counts=True
                    )
                    # Intra-instance attention logits using the same features as gate_input0
                    att_logits0 = self.intra_att_mlps[l](gate_input0).squeeze(-1)  # [num_pairs]
                    att_alpha0 = softmax(att_logits0, pair_idx, num_nodes=unique_pair.size(0))  # per (inst,dst)
                    weighted_msg0 = att_alpha0.unsqueeze(1) * mod_msg0
                    # aggregate to per (inst,dst)
                    node_delta_pair = scatter_add(weighted_msg0, pair_idx, dim=0, dim_size=unique_pair.size(0))  # [num_unique_pairs, d]

                    # Step1c: per-(edge,inst) edge deltas (before FiLM)
                    edge_delta_ei = mod_msg0  # reuse mod_msg0 as edge-side delta base

                else:
                    node_delta_pair = torch.zeros(0, units, device=x.device)
                    edge_delta_ei = torch.zeros(0, units, device=x.device)

                # Step 2: per-instance shared FiLM over (z_i, ctx_i), then cross-instance aggregation
                if ctx_i_vector is None:
                    ctx_i_vec = torch.zeros(num_instances, device=x.device)
                else:
                    ctx_i_vec = ctx_i_vector
                film_in = torch.cat([z_tensor, ctx_i_vec.unsqueeze(1)], dim=1)
                if enable_film:
                    if self.share_film_across_layers:
                        node_film = self.instance_node_film(film_in)
                        edge_film = self.instance_edge_film(film_in)
                    else:
                        node_film = self.instance_node_film[l](film_in)
                        edge_film = self.instance_edge_film[l](film_in)
                    raw_gamma_n, raw_beta_n = node_film.chunk(2, dim=1)  # [I,d]
                    raw_gamma_e, raw_beta_e = edge_film.chunk(2, dim=1)  # [I,d]
                    # Remove clamping: allow full modulation range
                    gamma_n = 1.0 + raw_gamma_n
                    beta_n = raw_beta_n
                    gamma_e = 1.0 + raw_gamma_e
                    beta_e = raw_beta_e
                else:
                    gamma_n = torch.ones(num_instances, units, device=x.device)
                    beta_n = torch.zeros(num_instances, units, device=x.device)
                    gamma_e = torch.ones(num_instances, units, device=x.device)
                    beta_e = torch.zeros(num_instances, units, device=x.device)

                # Compressed aggregation without allocating I*N buffers:
                if edge_ids_all.numel() > 0:
                    # Apply per-instance node FiLM to each (inst,node) aggregated delta
                    inst_for_pair = unique_pair // num_nodes
                    node_for_pair = unique_pair % num_nodes
                    gn_pair = gamma_n[inst_for_pair]
                    bn_pair = beta_n[inst_for_pair]
                    node_delta_pair_f = gn_pair * node_delta_pair + bn_pair  # [num_unique_pairs, d]
                    # Aggregate across instances to each global node with mean
                    sum_per_node = scatter_add(node_delta_pair_f, node_for_pair, dim=0, dim_size=num_nodes)
                    counts_per_node = scatter_add(
                        torch.ones_like(node_for_pair, dtype=torch.float, device=x.device),
                        node_for_pair, dim=0, dim_size=num_nodes
                    ).unsqueeze(1).clamp_min(1.0)
                    m_node_final = sum_per_node / counts_per_node  # [N,d]
                else:
                    m_node_final = torch.zeros(num_nodes, units, device=x.device)

                # Apply FiLM to edge deltas per (edge,inst), then mean across instances per edge
                if edge_ids_all.numel() > 0:
                    if enable_film:
                        # Edge-conditional FiLM delta
                        edge_cond_in = torch.cat([z_tensor[inst_ids_all], w_e0, dist_emb0, s_emb0, ctxu_emb0], dim=1)
                        raw_dgamma_dbeta = self.edge_cond_film_mlps[l](edge_cond_in)
                        raw_dgamma, raw_dbeta = raw_dgamma_dbeta.chunk(2, dim=1)
                        # Remove clamping: use raw deltas
                        dgamma = raw_dgamma
                        dbeta = raw_dbeta
                        gei_base = gamma_e[inst_ids_all]
                        bei_base = beta_e[inst_ids_all]
                        gei = gei_base + dgamma
                        bei = bei_base + dbeta
                    else:
                        gei = gamma_e[inst_ids_all]
                        bei = beta_e[inst_ids_all]
                    edge_delta_ei_f = gei * edge_delta_ei + bei  # [num_pairs,d]
                    messages_all = torch.zeros(num_edges, units, device=x.device)
                    messages_all = messages_all.index_add(0, edge_ids_all, edge_delta_ei_f)
                    counts_e = torch.zeros(num_edges, device=x.device)
                    counts_e = counts_e.index_add(0, edge_ids_all, torch.ones_like(edge_ids_all, dtype=torch.float))
                    messages_all = messages_all / torch.clamp_min(counts_e, 1.0).unsqueeze(1)
                else:
                    messages_all = torch.zeros(num_edges, units, device=x.device)

                # Use node aggregate directly as agg for node update
                agg = m_node_final
            else:
                # Original 'edge_first' path (current behavior with GAT-style pooling)
                messages_all = torch.zeros(num_edges, units, device=x.device)
                if edge_ids_all.numel() > 0:
                    msg = self.message_mlps[l](x_emb[src_all])
                    w_e = w_emb[edge_ids_all]
                    z_i = enhanced_z_tensor[inst_ids_all]
                    s_uvi = s_uvi_all
                    ctx_uvi = ctx_uvi_all
                    dist = dist_all
                    dist_emb = self.dist_embedder(dist.unsqueeze(1))
                    s_uvi_emb = self.s_uvi_embedder(s_uvi.unsqueeze(1)) if s_uvi.numel() > 0 else torch.zeros_like(dist_emb)
                    ctx_uvi_emb = self.ctx_uvi_embedder(ctx_uvi.unsqueeze(1)) if ctx_uvi.numel() > 0 else torch.zeros_like(dist_emb)
                    suvi_ctx_emb = s_uvi_emb + ctx_uvi_emb
                    gate_input = torch.cat([z_i, w_e + suvi_ctx_emb], dim=1)
                    gate = 1.0 + torch.tanh(self.gate_mlps[l](gate_input))
                    film_input = z_i + dist_emb + suvi_ctx_emb
                    film_out = self.shared_film_mlp(film_input)
                    gamma, beta = film_out.chunk(2, dim=1)
                    mod_msg = gate * (gamma * msg + beta)
                    att_logits = self.att_mlps[l](gate_input).squeeze(-1)
                    att_alpha = softmax(att_logits, edge_ids_all, num_nodes=num_edges)
                    weighted_msg = att_alpha.unsqueeze(1) * mod_msg
                    messages_all = messages_all + scatter_add(weighted_msg, edge_ids_all, dim=0, dim_size=num_edges)
                if debug_enabled:
                    assert messages_all.size(0) == num_edges, "messages_all first dim mismatch"
                    assert not torch.isnan(messages_all).any(), "messages_all has NaN"
                    assert not torch.isinf(messages_all).any(), "messages_all has Inf"
                    torch.cuda.synchronize(messages_all.device) if messages_all.is_cuda else None
            if self.aggregation_mode != 'instance_first':
                agg = torch.zeros(num_nodes, units, device=x.device)
                dst = edge_index[0]
                agg = agg.index_add(0, dst, messages_all)
            
            if self.use_residual_norm:
                node_residual = x_emb
                node_update = self.node_update_mlps[l](x_emb) + agg
                node_update = node_residual + node_update  # Residual connection first
                x_emb = self.act_fn(self.v_lns[l](node_update))  # Then LayerNorm and activation
            else:
                x_emb = self.act_fn(self.v_bns[l](self.node_update_mlps[l](x_emb) + agg))
            
            src = edge_index[1]
            dst = edge_index[0]
            w1 = self.edge_update_mlps[l](w_emb)
            edge_node_contrib = self.edge_node_mlps[l](x_emb[src]) + self.edge_node_mlps[l](x_emb[dst])
            # Incorporate cross-instance edge delta aggregate when in instance_first mode
            if self.aggregation_mode == 'instance_first':
                edge_update_input = w1 + edge_node_contrib + messages_all
            else:
                edge_update_input = w1 + edge_node_contrib
            
            if self.use_residual_norm:
                edge_residual = w_emb
                edge_update = edge_update_input
                edge_update = edge_residual + edge_update  # Residual connection first
                w_emb = self.act_fn(self.e_lns[l](edge_update))  # Then LayerNorm and activation
            else:
                w_emb = self.act_fn(self.e_bns[l](edge_update_input))
            
            if node_ids_concat.numel() > 0:
                # Pre-attention instance readout
                z_pre = scatter_mean(
                    x_emb[node_ids_concat], 
                    inst_ids_concat, 
                    dim=0, 
                    dim_size=num_instances
                )
                # Masked center self-attention within each batch
                instance_center_indices = []
                for inst_id in range(num_instances):
                    if inst_id in instance_mapping and len(instance_mapping[inst_id]) > 0:
                        instance_center_indices.append(instance_mapping[inst_id][-1])
                if instance_center_indices:
                    instance_center_indices = torch.tensor(instance_center_indices, device=x.device)
                    instance_center_nodes = x_emb[instance_center_indices]  # [num_instances, units]
                    if num_instances > 1:
                        H = instance_center_nodes.unsqueeze(0)  # [1, I, d]
                        batch_ids = torch.arange(num_instances, device=x.device) // seq_len  # [I]
                        same_batch = batch_ids.unsqueeze(0) == batch_ids.unsqueeze(1)  # [I,I]
                        attn_mask = ~same_batch  # True means mask
                        attn_out, _ = self.instance_self_attention[l](H, H, H, attn_mask=attn_mask)
                        instance_center_nodes = attn_out.squeeze(0)
                    x_emb = x_emb.clone()
                    x_emb[instance_center_indices] = instance_center_nodes
                    # Post-attention instance readout
                    z_post = scatter_mean(
                        x_emb[node_ids_concat], 
                        inst_ids_concat, 
                        dim=0, 
                        dim_size=num_instances
                    )
                else:
                    z_post = z_pre
                # Apply residual-norm pipeline and update tensors for next layer (z_post does not include target_emb)
                if self.use_residual_norm:
                    z_tensor_clean = self.z_lns[l](z_post)
                else:
                    z_tensor_clean = z_post
                enhanced_z_tensor = z_tensor_clean + target_emb 
                z_tensor = z_tensor_clean
        
        edge_predictions = torch.zeros(num_edges, num_instances, device=x.device)
        
        if instance_to_edges is not None:
            inst_edge_pairs = []
            for inst_id in range(num_instances):
                edge_ids = instance_to_edges.get(inst_id, [])
                for edge_id in edge_ids:
                    inst_edge_pairs.append((inst_id, edge_id))
            
            if inst_edge_pairs:
                inst_ids, edge_ids = zip(*inst_edge_pairs)
                inst_ids = torch.tensor(inst_ids, device=x.device)
                edge_ids = torch.tensor(edge_ids, device=x.device)
                
                # Align prediction direction with message passing: src = edge_index[1], dst = edge_index[0]
                src = edge_index[1, edge_ids]
                dst = edge_index[0, edge_ids]
                w_edges = w_emb[edge_ids]
                x_src = x_emb[src]
                x_dst = x_emb[dst]
                z_i = enhanced_z_tensor[inst_ids]  
                center_features = torch.zeros_like(x_src)
                
                if self.prediction_mode == 'dot_product':
                    # For dot_product, use z without target_emb to avoid bias leakage
                    z_i = z_tensor_clean[inst_ids]
                    edge_features = torch.cat([x_src, x_dst, w_edges, center_features], dim=1)  # [num_pairs, 4*units]
                    edge_emb = self.edge_mlp(edge_features)  # [num_pairs, units]
                    instance_emb = self.instance_mlp(z_i)  # [num_pairs, units]
                    # Normalize and scale by learnable temperature for stable logits
                    edge_emb = F.normalize(edge_emb, dim=1)
                    instance_emb = F.normalize(instance_emb, dim=1)
                    temperature = F.softplus(self.dot_temperature)
                    batch_predictions = temperature * torch.sum(edge_emb * instance_emb, dim=1)  # [num_pairs]
                else:  # mlp_concat
                    pred_features_raw = torch.cat([x_src, x_dst, w_edges, z_i, center_features], dim=1)
                    pred_features_projected = self.edge_projection(pred_features_raw)
                    # Return logits (no sigmoid).
                    batch_predictions = self.output_mlp(pred_features_projected).squeeze(-1)
                
                edge_predictions[edge_ids, inst_ids] = batch_predictions
        
        return x_emb, w_emb, z_tensor_clean, edge_predictions
    
class MultiRelBasisHeteroGNN(nn.Module):
    """
    Multi-relation GNN with basis decomposition and integrated instance-center self-attention per layer.
    """
    def __init__(self, depth, feats, units, act_fn, agg_fn,
                 num_relations, num_bases, num_heads, instance_node_indices):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = act_fn
        self.agg_fn = agg_fn
        self.num_rel = num_relations
        self.num_bases = num_bases
        self.instance_nodes = instance_node_indices

        # Input projections
        self.v_lin0 = nn.Linear(feats, units)
        self.e_lin0 = nn.Linear(3, units)  # Support up to 3-dimensional edge features

        # Basis MLPs per layer
        self.basis_mlps = nn.ModuleList([
            nn.ModuleList([nn.Linear(units, units) for _ in range(num_bases)])
            for _ in range(depth)
        ])
        # Relation-to-basis coefficients
        self.coeffs = nn.Parameter(torch.Tensor(depth, num_relations, num_bases))
        self.v_bns = nn.ModuleList([BatchNorm(units) for _ in range(depth)])
        self.e_bns = nn.ModuleList([BatchNorm(units) for _ in range(depth)])

        # Self-attention for instance centers integrated per layer
        self.layer_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=units, num_heads=num_heads)
            for _ in range(depth)
        ])

        nn.init.xavier_uniform_(self.coeffs)

    def forward(self, x, edge_index, edge_attr, relation_index):
        # Initial embeddings
        x = self.act_fn(self.v_lin0(x))           # h^{(0)}
        w = self.act_fn(self.e_lin0(edge_attr))   # w^{(0)}

        # Layer-wise message passing + instance self-attention
        for l in range(self.depth):
            x0, w0 = x, w
            # Basis outputs: [B, N, units]
            basis_outs = torch.stack([
                self.basis_mlps[l][b](x0) for b in range(self.num_bases)
            ], dim=0)
            coeff_l = self.coeffs[l]               # [R, B]
            coeff_e = coeff_l[relation_index]      # [E, B]
            # Messages per edge: [E, units]
            m_feats = torch.einsum('rb, bNv -> Nv', coeff_e, basis_outs[:, edge_index[1]])
            gates = torch.sigmoid(w0)
            agg = self.agg_fn(gates.unsqueeze(-1) * m_feats, edge_index[0])

            # Node & edge updates
            x = x0 + self.act_fn(self.v_bns[l](agg))
            w = w0 + self.act_fn(self.e_bns[l](w0 + x[edge_index[0]] + x[edge_index[1]]))

            # Integrated instance-center self-attention
            H_I = x[self.instance_nodes]            # [I, units]
            # Prepare for multihead: seq_len, batch, embed
            H_I_seq = H_I.unsqueeze(1)              # [I, 1, units]
            attn = self.layer_attn[l]
            attn_out, _ = attn(H_I_seq, H_I_seq, H_I_seq)  # [I, 1, units]
            x = x.clone()
            x[self.instance_nodes] = attn_out.squeeze(1)

        return x, w
    
class MultiRelBasisEmbNet(nn.Module):
    """
    Multi-relation Embedding Network with basis decomposition.
    Supports R relations with B basis MLPs per layer.
    """
    def __init__(self, depth, feats, units, act_fn, agg_fn, num_relations, num_bases):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = act_fn
        self.agg_fn = agg_fn
        self.num_rel = num_relations
        self.num_bases = num_bases

        # Node input projection
        self.v_lin0 = nn.Linear(feats, units)
        # Edge input projection
        self.e_lin0 = nn.Linear(3, units)  # Support up to 3-dimensional edge features

        # Basis MLPs per layer
        # shape: [depth][num_bases]
        self.basis_mlps = nn.ModuleList([
            nn.ModuleList([nn.Linear(units, units) for _ in range(num_bases)])
            for _ in range(depth)
        ])
        # Relation-to-basis coefficients per layer
        # shape: [depth, num_relations, num_bases]
        self.coeffs = nn.Parameter(
            torch.Tensor(depth, num_relations, num_bases)
        )
        # BatchNorms
        self.v_bns = nn.ModuleList([BatchNorm(units) for _ in range(depth)])
        self.e_bns = nn.ModuleList([BatchNorm(units) for _ in range(depth)])

        # Initialize coefficients
        nn.init.xavier_uniform_(self.coeffs)

    def forward(self, x, edge_index, edge_attr, relation_index):
        # x: [N, feats]
        # edge_attr: [E, 1]
        # relation_index: [E] in 0..num_rel-1
        x = self.act_fn(self.v_lin0(x))
        w = self.act_fn(self.e_lin0(edge_attr))

        for l in range(self.depth):
            x0 = x
            w0 = w
            # Compute relation-specific message transforms
            # Precompute basis outputs
            # basis_outs: [num_bases, E, units]
            basis_outs = []
            for b in range(self.num_bases):
                basis_outs.append(self.basis_mlps[l][b](x0))
            basis_outs = torch.stack(basis_outs, dim=0)
            # coeffs: [num_rel, num_bases]
            coeff_l = self.coeffs[l]  # [R, B]
            # Select coeffs for each edge: [E, B]
            coeff_e = coeff_l[relation_index]  # gather per-edge
            # Compute MLP_m^{(l,r)} h_v transforms: weighted sum of bases
            # m_feats: [E, units]
            m_feats = torch.einsum('rb, bEv -> Ev', coeff_e, basis_outs)

            # Gate using edge features
            gates = torch.sigmoid(w0)
            # Message passing aggregate
            agg = self.agg_fn(gates.unsqueeze(-1) * m_feats[edge_index[1]], edge_index[0])

            # Node update
            x = x0 + self.act_fn(self.v_bns[l](agg))
            # Edge update: optional self-interaction + node contributions
            w = w0 + self.act_fn(
                self.e_bns[l](
                    w0 + x[edge_index[0]] + x[edge_index[1]]
                )
            )
        return w
    
# GNN for edge embeddings
class EmbNet(nn.Module):
    @classmethod
    def make(cls, args):
        return cls(args.emb_depth, 2, args.net_units, args.net_act_fn, args.emb_agg_fn).to(args.device)
    def __init__(self, depth, feats, units, act_fn, agg_fn):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = act_fn
        self.agg_fn = agg_fn
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(3, self.units)  # Support up to 3-dimensional edge features
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
    def reset_parameters(self):
        raise NotImplementedError
    def forward(self, x, edge_index, edge_attr):
        x = x
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return w

# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device
    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad = False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = act_fn
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
    def reset_parameters(self):
        for layer in self.lins:
            layer.reset_parameters()
    @staticmethod
    def is_trainable(par):
        return par.requires_grad
    def trainables(self):
        for par in self.parameters():
            if self.is_trainable(par):
                yield par
    def named_trainables(self):
        for name, par in self.named_parameters():
            if self.is_trainable(par):
                yield name, par
    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
        return x

class Net(nn.Module):
    def __init__(self, args, use_multi_rel_emb_net=False, use_shared_basis_film=False, use_instance_hypergraph=False, num_relations=2, num_bases=4, num_instances=20, num_heads=8, prediction_mode='dot_product', use_residual_norm=False):
        super().__init__()
        self.use_shared_basis_film = use_shared_basis_film
        self.use_instance_hypergraph = use_instance_hypergraph
        
        if use_instance_hypergraph:
            import os
            agg_mode = os.environ.get('TSP_AGGREGATION_MODE', 'edge_first')
            share_film = os.environ.get('TSP_SHARE_FILM', '1') == '1'
            self.emb_net = InstanceAwareHypergraphGNN(
                depth=args.emb_depth,
                feats=2,  # 2D coordinates
                units=args.net_units,
                act_fn=args.net_act_fn,
                agg_fn=args.emb_agg_fn,
                num_instances=num_instances,
                num_heads=num_heads,
                prediction_mode=prediction_mode,
                use_residual_norm=use_residual_norm,
                aggregation_mode=agg_mode,
                share_film_across_layers=share_film
            )
        elif use_shared_basis_film:
            self.emb_net = SharedBasisFiLMAttentionGNN(
                depth=args.emb_depth,
                feats=2,  # 2D coordinates
                units=args.net_units,
                act_fn=args.net_act_fn,
                agg_fn=args.emb_agg_fn,
                num_instances=num_instances,
                num_relations=num_relations,  # Now supports 3 types: 0=graph, 1=tour, 2=instance-center
                num_bases=num_bases,
                num_heads=num_heads
            )
        elif use_multi_rel_emb_net:
            self.emb_net = MultiRelBasisEmbNet(
                depth=args.emb_depth,
                feats=2,  # 2D coordinates
                units=args.net_units,
                act_fn=args.net_act_fn,
                agg_fn=args.emb_agg_fn,
                num_relations=num_relations,
                num_bases=num_bases
            )
        else:
            self.emb_net = EmbNet.make(args)
    
    def forward(self, x, edge_index, edge_attr, batch, position=None, emb_net=None, gat_pooling=None, **kwargs):
        return self.infer(
            x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch,
            position=position,
            emb_net=self.emb_net if emb_net is None else emb_net,
            gat_pooling=gat_pooling,
            use_shared_basis_film=self.use_shared_basis_film,
            **kwargs
        )
    
    @staticmethod
    def infer(x, edge_index, edge_attr, batch, emb_net, position=None, gat_pooling=None, use_shared_basis_film=False, use_instance_hypergraph=False, **kwargs):
        # Handle edge attribute padding/processing based on model type
        if use_instance_hypergraph:
            # InstanceAwareHypergraphGNN expects 2D edge features [distance, is_center]
            if edge_attr.size(1) == 1:
                # Add is_center feature (assume all are graph edges = 0)
                is_center = torch.zeros(edge_attr.size(0), 1, device=edge_attr.device, dtype=edge_attr.dtype)
                edge_attr = torch.cat([edge_attr, is_center], dim=1)
            elif edge_attr.size(1) > 2:
                # Truncate to first 2 dimensions
                edge_attr = edge_attr[:, :2]
            
            # Get additional parameters for InstanceAwareHypergraphGNN
            instance_mapping = kwargs.get('instance_mapping')
            instance_edges = kwargs.get('instance_edges')
            
            if instance_mapping is None:
                raise ValueError("InstanceAwareHypergraphGNN requires instance_mapping")
            
            # Use InstanceAwareHypergraphGNN with integrated prediction
            node_emb, edge_emb, z_instances, predictions = emb_net(x, edge_index, edge_attr, **kwargs)
            
            # For InstanceAwareHypergraph mode, we return node embeddings and predictions
            return node_emb, edge_emb, z_instances, predictions
        else:
            # Ensure edge_attr has the correct shape for other models (pad to 3 dimensions if needed)
            if edge_attr.size(1) == 1:
                # Pad 1D edge features to 3D with zeros
                padding = torch.zeros(edge_attr.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
                edge_attr = torch.cat([edge_attr, padding], dim=1)
            elif edge_attr.size(1) == 2:
                # Pad 2D edge features to 3D with zeros
                padding = torch.zeros(edge_attr.size(0), 1, device=edge_attr.device, dtype=edge_attr.dtype)
                edge_attr = torch.cat([edge_attr, padding], dim=1)
            # If already 3D or higher, use as is (will be truncated to first 3 dims by linear layer)
            
        if use_shared_basis_film:
            # For SharedBasisFilmAttentionGNN, we need additional parameters
            type_index = kwargs.get('type_index')
            inst_index = kwargs.get('inst_index') 
            instance_nodes = kwargs.get('instance_nodes')
            
            if type_index is None or inst_index is None or instance_nodes is None:
                raise ValueError("SharedBasisFilmAttentionGNN requires type_index, inst_index, and instance_nodes")
            
            # Use SharedBasisFilmAttentionGNN
            node_emb, edge_emb = emb_net(x, edge_index, edge_attr, type_index, inst_index, instance_nodes)
            
            # For SharedBasisFilm mode, we return node embeddings directly without pooling
            # The graph_emb is not used in this mode as we do direct edge prediction
            return node_emb, edge_emb
        else:
            # Standard EmbNet or MultiRelBasisEmbNet processing
            if hasattr(emb_net, 'forward') and len(emb_net.forward.__code__.co_varnames) > 4:
                # MultiRelBasisEmbNet case - needs relation_index
                relation_index = kwargs.get('relation_index')
                if relation_index is not None:
                    edge_emb = emb_net(x, edge_index, edge_attr, relation_index)
                else:
                    edge_emb = emb_net(x, edge_index, edge_attr)
            else:
                # Standard EmbNet case
                edge_emb = emb_net(x, edge_index, edge_attr)
            
        edge_batch = batch[edge_index[0]]
        edge_position = position[edge_index[0]]
        
        unique_positions = torch.unique(position).sort()[0]
        unique_batches = torch.unique(batch).sort()[0]
        
        seq_len = len(unique_positions)
        batch_size = len(unique_batches)
        hidden_size = edge_emb.size(1)
        graph_emb = torch.zeros(seq_len, batch_size, hidden_size, device=edge_emb.device)
        
        for pos_idx, pos in enumerate(unique_positions):
            for batch_idx, b in enumerate(unique_batches):
                mask = (edge_position == pos) & (edge_batch == b)
                if mask.sum() > 0:
                    batch_edge_emb = edge_emb[mask]  # [num_edges_in_batch, hidden_size]
                    batch_indices = torch.zeros(mask.sum(), dtype=torch.long, device=edge_emb.device)
                    
                    if gat_pooling is not None:
                        pooled = gat_pooling(batch_edge_emb, batch_indices)  # [1, hidden_size]
                        graph_emb[pos_idx, batch_idx] = pooled[0]
                    else:
                        # Fallback to mean pooling
                        graph_emb[pos_idx, batch_idx] = batch_edge_emb.mean(dim=0)
            
        return graph_emb, edge_emb
