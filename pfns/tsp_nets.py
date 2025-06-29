import torch
from copy import copy, deepcopy
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, BatchNorm
import torch.nn.functional as F


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
        self.e_lin0 = nn.Linear(1, units)

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
        self.e_lin0 = nn.Linear(1, units)

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
        self.e_lin0 = nn.Linear(1, units)

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
        self.e_lin0 = nn.Linear(1, self.units)
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
    def __init__(self, args, use_multi_rel_emb_net=False, use_shared_basis_film=False, num_relations=2, num_bases=4, num_instances=20, num_heads=8):
        super().__init__()
        self.use_shared_basis_film = use_shared_basis_film
        
        if use_shared_basis_film:
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
    def infer(x, edge_index, edge_attr, batch, emb_net, position=None, gat_pooling=None, use_shared_basis_film=False, **kwargs):
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
