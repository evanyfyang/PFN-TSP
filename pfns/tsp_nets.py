import torch
from copy import copy, deepcopy
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import torch.nn.functional as F

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
    def __init__(self, args):
        super().__init__()
        self.emb_net = EmbNet.make(args)
    
    def forward(self, x, edge_index, edge_attr, batch, position=None, emb_net=None, gat_pooling=None):
        return self.infer(
            x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch,
            position=position,
            emb_net=self.emb_net if emb_net is None else emb_net,
            gat_pooling=gat_pooling
        )
    
    @staticmethod
    def infer(x, edge_index, edge_attr, batch, emb_net, position=None, gat_pooling=None):
        # Generate edge embeddings
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
