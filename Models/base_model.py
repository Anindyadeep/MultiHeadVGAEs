import torch 
import torch.nn as nn 
import torch_geometric.nn as gnn 
import torch.nn.functional as F 
import torch_geometric.nn.models as M


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.gcn_shared = gnn.GCNConv(in_channels, hidden_channels)
        self.gcn_mu = gnn.GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = gnn.GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar

class BaseVGAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(BaseVGAE, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = M.InnerProductDecoder()
    
    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encoder(x, adj)
        z = self.reparametrize(mu, logvar)
        return self.decoder.forward_all(z, sigmoid=True), mu, logvar
        