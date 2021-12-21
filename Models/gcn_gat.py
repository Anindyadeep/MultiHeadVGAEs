import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch_geometric.nn as gnn 
import torch_geometric.nn.models as M 

class GCNGATVGAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, num_heads = 3):
        super(GCNGATVGAE, self).__init__()
        self.gcn = gnn.GCNConv(input_feat_dim, hidden_dim1)
        self.gcn_mu = gnn.GCNConv(hidden_dim1, hidden_dim2)
        self.gcn_logvar = gnn.GCNConv(hidden_dim1, hidden_dim2)

        self.gat = gnn.GATConv(input_feat_dim, hidden_dim1, heads=num_heads, concat = True)
        self.gat_mu = gnn.GATConv(num_heads * hidden_dim1, hidden_dim2, heads=num_heads, concat = False)
        self.gat_logvar = gnn.GATConv(num_heads * hidden_dim1, hidden_dim2, heads=num_heads, concat = False)

        self.decoder = M.InnerProductDecoder()
        print(f"Using {num_heads} heads")

    
    def encode(self, x, edge_index):
        h_gcn = self.gcn(x, edge_index)
        h_gcn = F.relu(h_gcn)

        h_gat = self.gat(x, edge_index)
        h_gat = F.relu(h_gat)

        mu_gcn, logvar_gcn = self.gcn_mu(h_gcn, edge_index), self.gcn_logvar(h_gcn, edge_index)
        mu_gat, logvar_gat = self.gat_mu(h_gat, edge_index), self.gat_logvar(h_gat, edge_index)
        return [mu_gcn, logvar_gcn, mu_gat, logvar_gat]
    
    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
    
    def forward(self, x, edge_index):
        mu_gcn, logvar_gcn, mu_gat, logvar_gat = self.encode(x, edge_index)
        z_gcn = self.reparametrize(mu_gcn, logvar_gcn)
        z_gat = self.reparametrize(mu_gat, logvar_gat)
        z_max = torch.max(z_gcn, z_gat)
        return self.decoder.forward_all(z_max, sigmoid=True), torch.max(mu_gcn, mu_gat), torch.max(logvar_gcn, logvar_gat)