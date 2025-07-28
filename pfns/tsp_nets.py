import torch
from copy import copy, deepcopy
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, BatchNorm
import torch.nn.functional as F

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
    def __init__(self, depth, feats, units, act_fn, agg_fn, num_instances, num_heads=8, message_agg='mean', prediction_mode='dot_product', use_residual_norm=False):
        super().__init__()
        self.depth = depth
        self.units = units
        self.act_fn = act_fn
        self.agg_fn = agg_fn
        self.num_instances = num_instances
        self.message_agg = message_agg  # 'mean', 'sum', 'max', etc.
        self.prediction_mode = prediction_mode  # 'dot_product' or 'mlp_concat'
        self.use_residual_norm = use_residual_norm  # Control residual connections and LayerNorm

        self.v_lin0 = nn.Linear(feats, units)  # W_v for nodes
        self.e_lin0 = nn.Linear(2, units)      # W_e for edges [distance, is_center]
        
        self.dist_embedder = nn.Linear(1, units)  # Distance to embedding (same as w_emb)
        self.s_uvi_embedder = nn.Linear(1, units)  # s_uvi (0/1) to embedding (same as w_emb)
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
        
        self.shared_film_mlp = nn.Sequential(
            nn.Linear(units, units),  # [z_i + dist_emb + s_uvi_emb]
            act_fn,
            nn.Linear(units, 2 * units)   # output [gamma, beta]
        )
        
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
        else:  # mlp_concat
            self.edge_projection = nn.Linear(5 * units, units)  # Project 5*units to units first
            self.output_mlp = nn.Sequential(
                nn.Linear(units, units // 2),  # Reduced size
                act_fn,
                nn.Linear(units // 2, 1)
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

    def forward(self, x, edge_index, edge_attr, instance_mapping, edge_to_instances=None, instance_to_edges=None, s_uvi_matrix=None, single_eval_pos=None):
        """
        Forward pass for one-to-many node/edge-instance mapping.
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        num_instances = s_uvi_matrix.size(1) if s_uvi_matrix is not None else len(instance_mapping)
        units = self.units
        
        x_emb = self.act_fn(self.v_lin0(x))  # [N, units]
        w_emb = self.act_fn(self.e_lin0(edge_attr))  # [E, units]
        
        instance_tensor = torch.full((num_nodes,), -1, dtype=torch.long, device=x.device)
        for inst_id, node_indices in instance_mapping.items():
            if len(node_indices) > 0:
                instance_tensor[node_indices] = inst_id
        
        z_tensor = torch.zeros(num_instances, units, device=x.device)
        valid_mask = instance_tensor >= 0
        if valid_mask.any():
            z_tensor = scatter_mean(
                x_emb[valid_mask], 
                instance_tensor[valid_mask], 
                dim=0, 
                dim_size=num_instances
            )
        
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
        
        for l in range(self.depth):
            messages_all = torch.zeros(num_edges, units, device=x.device)
            counts = torch.zeros(num_edges, device=x.device)
            
            if edge_to_instances is not None:
                edge_inst_pairs = []
                for e in range(num_edges):
                    if e in edge_to_instances:
                        for inst_id in edge_to_instances[e]:
                            edge_inst_pairs.append((e, inst_id))
                
                if edge_inst_pairs:
                    edge_ids, inst_ids = zip(*edge_inst_pairs)
                    edge_ids = torch.tensor(edge_ids, device=x.device)
                    inst_ids = torch.tensor(inst_ids, device=x.device)
                    
                    src = edge_index[1, edge_ids]
                    dst = edge_index[0, edge_ids]
                    
                    msg = self.message_mlps[l](x_emb[src])
                    w_e = w_emb[edge_ids]
                    z_i = enhanced_z_tensor[inst_ids]  # Use enhanced instance embeddings
                    
                    s_uvi = s_uvi_matrix[edge_ids, inst_ids] if s_uvi_matrix is not None else torch.zeros(len(edge_ids), device=x.device)
                    dist = edge_attr[edge_ids, 0]
                    
                    dist_emb = self.dist_embedder(dist.unsqueeze(1))  # [num_pairs, units]
                    s_uvi_emb = self.s_uvi_embedder(s_uvi.unsqueeze(1))  # [num_pairs, units]
                    
                    gate_input = torch.cat([z_i, w_e + s_uvi_emb], dim=1)
                    gate = torch.sigmoid(self.gate_mlps[l](gate_input))
                    
                    film_input = z_i + dist_emb + s_uvi_emb
                    film_out = self.shared_film_mlp(film_input)
                    gamma, beta = film_out.chunk(2, dim=1)
                    
                    mod_msg = gate * (gamma * msg + beta)
                    
                    messages_all = messages_all + scatter_add(mod_msg, edge_ids, dim=0, dim_size=num_edges)
                    counts = counts + scatter_add(torch.ones_like(edge_ids, dtype=torch.float), edge_ids, dim=0, dim_size=num_edges)
                    
                    mask = counts > 0
                    messages_all = torch.where(mask.unsqueeze(1), messages_all / counts.unsqueeze(1), messages_all)
            
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
            
            if self.use_residual_norm:
                edge_residual = w_emb
                edge_update = w1 + edge_node_contrib
                edge_update = edge_residual + edge_update  # Residual connection first
                w_emb = self.act_fn(self.e_lns[l](edge_update))  # Then LayerNorm and activation
            else:
                w_emb = self.act_fn(self.e_bns[l](w1 + edge_node_contrib))
            
            if valid_mask.any():
                z_tensor = scatter_mean(
                    x_emb[valid_mask], 
                    instance_tensor[valid_mask], 
                    dim=0, 
                    dim_size=num_instances
                )
                
                instance_center_indices = []
                for inst_id in range(num_instances):
                    if inst_id in instance_mapping and len(instance_mapping[inst_id]) > 0:
                        instance_center_indices.append(instance_mapping[inst_id][-1])
                
                if instance_center_indices:
                    instance_center_indices = torch.tensor(instance_center_indices, device=x.device)
                    instance_center_nodes = x_emb[instance_center_indices]  # [num_instances, units]
                    
                    if num_instances > 1:
                        updated_center_nodes = instance_center_nodes.clone()
                        
                        for batch_idx in range(batch_size):
                            batch_start = batch_idx * seq_len
                            batch_end = min((batch_idx + 1) * seq_len, num_instances)
                            
                            if batch_start < batch_end and batch_end - batch_start > 1:
                                batch_center_nodes = instance_center_nodes[batch_start:batch_end]  # [batch_seq_len, units]
                                
                                batch_center_nodes = batch_center_nodes.unsqueeze(1)  # [batch_seq_len, 1, units]
                                attn_out, _ = self.instance_self_attention[l](batch_center_nodes, batch_center_nodes, batch_center_nodes)
                                updated_center_nodes[batch_start:batch_end] = attn_out.squeeze(1)  # [batch_seq_len, units]
                        
                        instance_center_nodes = updated_center_nodes
                    
                    x_emb = x_emb.clone()
                    x_emb[instance_center_indices] = instance_center_nodes
                    
                    z_tensor_new = scatter_mean(
                        x_emb[valid_mask], 
                        instance_tensor[valid_mask], 
                        dim=0, 
                        dim_size=num_instances
                    )
                    if self.use_residual_norm:
                        z_tensor_new = z_tensor_clean + z_tensor_new  
                        z_tensor_clean = self.z_lns[l](z_tensor_new)  
                        enhanced_z_tensor = z_tensor_clean + target_emb 
                    else:
                        z_tensor_clean = z_tensor_new
                        enhanced_z_tensor = z_tensor_clean + target_emb 
        
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
                
                src = edge_index[0, edge_ids]
                dst = edge_index[1, edge_ids]
                w_edges = w_emb[edge_ids]
                x_src = x_emb[src]
                x_dst = x_emb[dst]
                z_i = enhanced_z_tensor[inst_ids]  
                center_features = torch.zeros_like(x_src)
                
                if self.prediction_mode == 'dot_product':
                    edge_features = torch.cat([x_src, x_dst, w_edges, center_features], dim=1)  # [num_pairs, 4*units]
                    edge_emb = self.edge_mlp(edge_features)  # [num_pairs, units]
                    instance_emb = self.instance_mlp(z_i)  # [num_pairs, units]
                    batch_predictions = torch.sigmoid(torch.sum(edge_emb * instance_emb, dim=1))  # [num_pairs]
                else:  # mlp_concat
                    pred_features_raw = torch.cat([x_src, x_dst, w_edges, z_i, center_features], dim=1)
                    pred_features_projected = self.edge_projection(pred_features_raw)
                    batch_predictions = torch.sigmoid(self.output_mlp(pred_features_projected)).squeeze(-1)
                
                edge_predictions[edge_ids, inst_ids] = batch_predictions
        
        return x_emb, w_emb, z_tensor, edge_predictions
    
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
            self.emb_net = InstanceAwareHypergraphGNN(
                depth=args.emb_depth,
                feats=2,  # 2D coordinates
                units=args.net_units,
                act_fn=args.net_act_fn,
                agg_fn=args.emb_agg_fn,
                num_instances=num_instances,
                num_heads=num_heads,
                prediction_mode=prediction_mode,
                use_residual_norm=use_residual_norm
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
