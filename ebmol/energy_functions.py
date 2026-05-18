import copy
import torch
from torch import nn

from ebmol.layers.egnn_new import EGNN 

class Swish(nn.Module):
    
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class EGNN_EnergyModel(nn.Module):
    def __init__(self, config, device='cpu'):
        super(EGNN_EnergyModel, self).__init__()
        self.n_feat = config.n_feat
        self.hidden_nf = config.hidden_nf
        self.out_node_nf = config.out_node_nf
        self.in_edge_nf = config.get('in_edge_nf', 1)
        self.n_layers = config.n_layers
        self.fully_connected = config.get('fully_connected', True)
        self.agg_method = config.get('agg_method', 'mean')
        self.agg_method_energy = config.get('agg_method_energy', 'mean')
        self.tanh = config.get('tanh', False)
        self.attention = config.get('attention', False)
        self.old_egnn = config.get('old_egnn', False)
        self.spectral_norm = config.get('spectral_norm', False)
        self.feat_scale_cat = config.get('feat_scale_cat', 1.0)
        self.inv_sublayers = config.get('inv_sublayers', 1)
        self.normalization_factor = config.get("normalization_factor", 1.0)
        self.config = config
        self.composition = False
        self.composites = []
        self.composite_weights = []
        self.verbose = True
        self.device = device

        assert self.fully_connected, 'Only fully connected edges are supported'
        
        if config.energy_based:
            self.out_node_nf = 1  # output log probability per graph
        else:
            self.out_node_nf = self.n_feat  # output features per node
        
        if hasattr(config, 'energy_feats') and config.energy_feats == 'h_out':
            self.skip_last_coord_update = True  # use h_out and x features for energy
        else:
            self.skip_last_coord_update = False  # use only h_out features for energy
            
        #define network

        self.egnn = EGNN(
                in_node_nf=self.n_feat, 
                in_edge_nf=1,
                out_node_nf=self.out_node_nf,
                hidden_nf=self.hidden_nf, 
                device=self.device, 
                act_fn=torch.nn.SiLU(),
                n_layers=self.n_layers, attention=self.attention, tanh=self.tanh, norm_constant=1,
                inv_sublayers=self.inv_sublayers, sin_embedding=False,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.agg_method,
                spectral_norm=self.spectral_norm,
                output_embedding=config.output_embedding,
                output_bias=config.output_bias,
                output_activation=config.output_activation,
                skip_last_coord_update=self.skip_last_coord_update,  # skip last pos layer to get h_out
                )
        
        if hasattr(config, 'use_ema') and config.use_ema:
            self.init_ema_model(config.ema_decay)
        else:
            self.ema_model = None

    def init_ema_model(self, decay):
        self.ema_model = copy.deepcopy(self.egnn)

        self.ema_decay = decay
        self.ema_model.to(self.device)
        # ema model doesnt require gradients
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def forward(self, data, return_node_energy=False, return_xpos=False, guidance_weights=None):
        
        if not hasattr(data, 'num_graphs'):
            num_graphs = 1
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        else:
            num_graphs = data.num_graphs

        cat_in = data.x 

        h_out, x = self.egnn(cat_in, data.pos, data.edge_index)

        # if not energy based, return features and positions directly
        if not self.config.energy_based: 
            if return_xpos:
                return h_out, x
            else:
                return torch.tensor(0.0, device=data.x.device)    #return zero for logp to keep interface consistent
        
        #calculate energy from features
        if self.config.energy_feats == 'h_out_and_x':
            # compute logp with h and x features
            out_feats = torch.mean(torch.concat([h_out, x], dim=-1), dim=-1, keepdim=True)
            h_out_batch = torch.scatter_reduce(torch.zeros(num_graphs, device=data.x.device), 0, data.batch, out_feats.squeeze(), reduce=self.agg_method_energy, include_self=False)  # dims [batch_size, out_node_nf=1]
        elif self.config.energy_feats == 'h_out':
            # compute logp only with h features
            h_out_batch = torch.scatter_reduce(torch.zeros(num_graphs, device=data.x.device), 0, data.batch, h_out.squeeze(), reduce=self.agg_method_energy, include_self=False)  # dims [batch_size, out_node_nf=1]
        else:
            raise ValueError(f"Unknown energy_feats: {self.config.energy_feats}. Choose from 'h_out_and_x' or 'h_out'.")
        
        energy = h_out_batch.view(-1,1)
        
        if self.composition and len(self.composites) > 0:
            composite_energy = 0
            if guidance_weights is None:
                guidance_weights = torch.ones(1).to(data.x.device)
            for comp, weight in zip(self.composites, self.composite_weights):
                composite_energy += weight * comp(data) * guidance_weights
            energy = energy + composite_energy.view(-1,1)

        if return_node_energy:
            return energy.detach(), h_out, None
     
        return energy

    def get_edges_batch(self, n_nodes, batch_size):
        edges = self.get_edges(n_nodes)
        edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
        edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
        if batch_size == 1:
            return edges, edge_attr
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges, edge_attr

    def get_edges(self, n_nodes):
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)

        edges = [rows, cols]
        return edges
    

    def get_grad(self, data, gradient_mask=None, guidance_weights=None):

        #just return network output if not energy based
        if not self.config.energy_based:
            with torch.no_grad():
                h_pred, x_pred = self.forward(data, return_xpos=True)
            v_pos =  -(x_pred - data.pos)
            v_h = -(h_pred - data.x)
            return (v_pos, v_h), torch.tensor(0.0, device=data.x.device)      #return zero logp to keep interface consistent
        
        if data.pos.grad is not None:
            data.pos.grad.zero_()
        
        if data.x.grad is not None:
            data.x.grad.zero_()

        data.pos.requires_grad_()
        data.x.requires_grad_()

        inputs = [data.pos, data.x]
        logp = self(data, guidance_weights=guidance_weights)

        num_graphs = data.num_graphs
        grad_outputs = torch.ones(num_graphs, device=logp.device, dtype=logp.dtype)
        grads_all = torch.autograd.grad(outputs=logp.squeeze(-1), 
                                        inputs=inputs, 
                                        grad_outputs=grad_outputs)

        #detach logp 
        logp = logp.detach()

        data.pos.requires_grad_(False)
        data.x.requires_grad_(False)

        if gradient_mask is not None:
            grads_all[0][gradient_mask] = torch.zeros_like(grads_all[0][gradient_mask])
            grads_all[1][gradient_mask] = torch.zeros_like(grads_all[1][gradient_mask]) 

        return grads_all, logp

    def update_ema(self, decay=None):
        """
        Exponential Moving Average update.
        """
        if decay is None:
            decay = self.ema_decay
        source_dict = self.egnn.state_dict()
        target_dict = self.ema_model.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay + source_dict[key].data * (1 - decay)
            )