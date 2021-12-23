import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch_geometric.nn as gnn 
import torch_geometric.nn.models as M 

class GCNMerge(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2):
        super(GCNMerge, self).__init__()
        self.gcn1 = gnn.GCNConv(input_feat_dim, hidden_dim1)
        self.gcn2 = gnn.GCNConv(input_feat_dim, hidden_dim1)
        mul = (hidden_dim1 + hidden_dim1) // hidden_dim2
        self.mu_gcn = gnn.GCNConv(mul * hidden_dim2, hidden_dim2)
        self.logvar_gcn = gnn.GCNConv(mul * hidden_dim2, hidden_dim2)

        self.decoder = M.InnerProductDecoder()

    
    def encode(self, x, edge_index):
        hidden1 = self.gcn1(x, edge_index)
        hidden2 = self.gcn2(x, edge_index)
        hidden_cat = torch.cat((hidden1, hidden2), dim=1)
        mu = self.mu_gcn(hidden_cat, edge_index)
        logvar = self.logvar_gcn(hidden_cat, edge_index)
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