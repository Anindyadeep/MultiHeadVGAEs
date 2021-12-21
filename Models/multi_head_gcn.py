import torch 
import torch.nn as nn 
import torch_geometric.nn as gnn 
import torch.nn.functional as F 
import torch_geometric.nn.models as M



class HiddenConv(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2):
        super(HiddenConv, self).__init__()
        # self.bn = nn.BatchNorm1d(input_feat_dim)
        self.gc_conv = gnn.GCNConv(input_feat_dim, hidden_dim1)
        self.mu_conv = gnn.GCNConv(hidden_dim1, hidden_dim2)
        self.logvar_conv = gnn.GCNConv(hidden_dim1, hidden_dim2)

    def forward(self, x, adj):
        #x = self.bn(x)
        hidden = self.gc_conv(x, adj)
        hidden = F.relu(hidden)
        return self.mu_conv(hidden, adj), self.logvar_conv(hidden, adj)



class MultiHeadGcnGVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, num_heads=3):
        super(MultiHeadGcnGVAE, self).__init__()
        self.input_feat_dim = input_feat_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_heads = num_heads
        self.hidden_convs = []

        for i in range(self.num_heads):
            self.hidden_convs.append(
                HiddenConv(
                    self.input_feat_dim, 
                    self.hidden_dim1, 
                    self.hidden_dim2))
        self.model_list = nn.ModuleList(self.hidden_convs)

        self.decoder = M.InnerProductDecoder()
        
    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, x, adj):
        mu_list = [] 
        logvar_list = []
        for i in range(self.num_heads):
            hidden_conv = self.hidden_convs[i]
            mu, logvar = hidden_conv(x, adj)
            mu_list.append(mu)
            logvar_list.append(logvar)

        z_list = []
        for mu,logvar in zip(mu_list, logvar_list):
            z_list.append(self.reparametrize(mu, logvar))

        
        mu_max = mu_list[0]
        logvar_max = logvar_list[0]
        z_max = z_list[0]

        for i  in range(1, self.num_heads):
            mu_max = torch.max(mu_max, mu_list[i])
            logvar_max = torch.max(logvar_max, logvar_list[i])
            z_max = torch.max(z_max, z_list[i])

        return self.decoder.forward_all(z_max, sigmoid=True), mu_max, logvar_max