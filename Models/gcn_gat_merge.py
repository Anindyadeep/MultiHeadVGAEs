import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch_geometric.nn as gnn 
import torch_geometric.nn.models as M 

class GCNMerge(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, num_heads):
        super(GCNMerge, self).__init__()
        self.gcn = gnn.GCNConv(input_feat_dim, hidden_dim1)
        self.gat = gnn.GATConv(input_feat_dim, hidden_dim1, heads = num_heads, concat=True)

        mul = (hidden_dim1 + (num_heads * hidden_dim1)) // hidden_dim1

        self.mu_gcn = gnn.GCNConv(mul * hidden_dim1, hidden_dim2)
        self.logvar_gat = gnn.GATConv(mul * hidden_dim1, hidden_dim2, heads=num_heads, concat=False)

        self.decoder = M.InnerProductDecoder()

    
    def encode(self, x, edge_index):
        hidden_gcn = self.gcn(x, edge_index)
        hidden_gat = self.gat(x, edge_index)
        hidden_cat = torch.cat((hidden_gcn, hidden_gat), dim=1)

        mu = self.mu_gcn(hidden_cat, edge_index)
        logvar = self.logvar_gat(hidden_cat, edge_index)
        return mu, logvar
    
    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
    
    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparametrize(mu, logvar)
        return self.decoder.forward_all(z, sigmoid=True), mu, logvar