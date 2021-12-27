import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch_geometric.nn as gnn 
import torch_geometric.nn.models as M 

import itertools

class MultiHeadGCNGATMergeVGAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, arch_list, num_heads):
        super(MultiHeadGCNGATMergeVGAE, self).__init__()
        self.gcn_encoders = {}
        self.gat_encoders = {}

        gcn_gat = [list(i) for j, i in itertools.groupby(sorted(arch_list))]

        gcns = gcn_gat[1]
        gats = gcn_gat[0]
        gcn_len = len(gcns)
        gat_len = len(gats)

        mul = ((gcn_len * hidden_dim1) + (gat_len * (num_heads * hidden_dim1))) // hidden_dim1

        if gcn_len != 0:
            for i in range(gcn_len):
                self.gcn_encoders[f'gcn_{i}'] = gnn.GCNConv(input_feat_dim, hidden_dim1)
            self.gcn_encoders = nn.ModuleDict(self.gcn_encoders)
                
        
        if gat_len != 0:
            for i in range(gat_len):
                self.gat_encoders[f'gat_{i}'] = gnn.GATConv(input_feat_dim, hidden_dim1, heads = num_heads, concat = True)
            self.gat_encoders = nn.ModuleDict(self.gat_encoders)
    
        self.mu_gcn = gnn.GCNConv(mul * hidden_dim1, hidden_dim2)
        self.logvar_gat = gnn.GATConv(mul * hidden_dim1, hidden_dim2, heads = num_heads, concat = False)
        self.decoder = M.InnerProductDecoder()
        

    def encode(self, x, edge_index):
        hidden_list = []

        for gcn in self.gcn_encoders:
            gcn_conv = self.gcn_encoders[gcn]
            hidden_list.append(
                gcn_conv(x, edge_index)
            )
        
        for gat in self.gat_encoders:
            gat_conv = self.gat_encoders[gat]
            hidden_list.append(
                gat_conv(x, edge_index)
            )
        
        h_concat = hidden_list[0]
        for h in hidden_list[1:]:
            h_concat = torch.cat((h_concat, h), dim=1)

        mu = self.mu_gcn(h_concat, edge_index)
        logvar = self.logvar_gat(h_concat, edge_index)
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


if __name__ == '__main__':
    arch_list = ['gcn', 'gat', 'gcn', 'gat']
    model = MultiHeadGCNGATMergeVGAE(
        input_feat_dim = 1433,
        hidden_dim1 = 32,
        hidden_dim2 = 16,
        arch_list = arch_list,
        num_heads = 3
    )

    print(model)