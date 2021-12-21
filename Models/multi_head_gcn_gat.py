import torch 
import torch.nn as nn 
import torch_geometric.nn as gnn 
import torch.nn.functional as F 
import torch_geometric.nn.models as M


class HiddenGCNEncoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2):
        super(HiddenGCNEncoder, self).__init__()
        self.gcn = gnn.GCNConv(input_feat_dim, hidden_dim1)
        self.gcn_mu = gnn.GCNConv(hidden_dim1, hidden_dim2)
        self.gcn_logvar = gnn.GCNConv(hidden_dim1, hidden_dim2)

    def forward(self, x, adj):
        hidden_gcn = self.gcn(x, adj)
        hidden_gcn = F.relu(hidden_gcn)
        return self.gcn_mu(hidden_gcn, adj), self.gcn_logvar(hidden_gcn, adj)


class HiddenGATEncoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, num_heads = 3):
        super(HiddenGATEncoder, self).__init__()
        self.gat = gnn.GATConv(input_feat_dim, hidden_dim1, heads=num_heads, concat = True) 
        self.gat_mu = gnn.GATConv(num_heads * hidden_dim1, hidden_dim2, heads=num_heads, concat = False)
        self.gat_logvar = gnn.GATConv(num_heads * hidden_dim1, hidden_dim2, heads=num_heads, concat = False)

    def forward(self, x, adj):
        hidden_gat = self.gat(x, adj)
        hidden_gat = F.relu(hidden_gat)
        return self.gat_mu(hidden_gat, adj), self.gat_logvar(hidden_gat, adj)


class MultiHeadGCNGATVGAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, arch_list, num_heads_gat = 3):
        super(MultiHeadGCNGATVGAE, self).__init__()
        """
        parameters:
        ----------
        input_feat_dim : number of the features of the input
        hidden_dim1    : hidden dim of the encoder
        hidden_dim2    : hidden_dim for mu and logvar
        arch_list      : list of the architectures that will be involved in this VGAE for e.g. ['gcn', 'gat', 'gcn', 'gat']
        num_heads_gat  : number of the heads of the GAT layer
        """
        self.input_feat_dim = input_feat_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_heads_gat = num_heads_gat
        self.arch_list = arch_list 
        self.hidden_encoders = []

        for arch in self.arch_list:
            if arch == 'gcn':
                self.hidden_encoders.append(
                    HiddenGCNEncoder(
                        self.input_feat_dim,
                        self.hidden_dim1,
                        self.hidden_dim2))
            
            else:
                self.hidden_encoders.append(
                    HiddenGATEncoder(
                        self.input_feat_dim,
                        self.hidden_dim1,
                        self.hidden_dim2))
        self.model_list = nn.ModuleList(self.hidden_encoders)
        self.decocder = M.InnerProductDecoder()

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
    
    def forward(self, x, adj):
        mu_gat_list, logvar_gat_list = [], []
        mu_gcn_list, logvar_gcn_list = [], []

        for i, arch in enumerate(self.arch_list):
            if arch == "gcn":
                hidden_conv = self.hidden_encoders[i]
                mu_gcn, logvar_gcn = hidden_conv(x, adj)
                mu_gcn_list.append(mu_gcn)
                logvar_gcn_list.append(logvar_gcn)
            
            else:
                hidden_gat = self.hidden_encoders[i]
                mu_gat, logvar_gat = hidden_gat(x, adj)
                mu_gat_list.append(mu_gat)
                logvar_gat_list.append(logvar_gat)
        
        z_gcn_list, z_gat_list = [], []
        for gcn_mu, gcn_logvar in zip(mu_gcn_list, logvar_gcn_list):
            z_gcn_list.append(self.reparametrize(gcn_mu, gcn_logvar))
        
        for gat_mu, gat_logvar in zip(mu_gat_list, logvar_gat_list):
            z_gat_list.append(self.reparametrize(gat_mu, gat_logvar))
        
        gcn_mu_max, gcn_logvar_max = mu_gcn_list[0], logvar_gcn_list[0]
        gat_mu_max, gat_logvar_max = mu_gat_list[0], logvar_gat_list[0]
        z_gcn_max, z_gat_max = z_gcn_list[0], z_gat_list[0]

        for i in range(1, len(mu_gcn_list)):
            gcn_mu_max = torch.max(gcn_mu_max, mu_gcn_list[i])
            gcn_logvar_max = torch.max(gcn_logvar_max, logvar_gcn_list[i])
            z_gcn_max = torch.max(z_gcn_max, z_gcn_list[i])
        
        for i in range(1, len(mu_gat_list)):
            gat_mu_max = torch.max(gat_mu_max, mu_gat_list[i])
            gat_logvar_max = torch.max(gat_logvar_max, logvar_gat_list[i])
            z_gat_max = torch.max(z_gat_max, z_gat_list[i])
        
        mu_max = torch.max(gcn_mu_max, gat_mu_max)
        logvar_max = torch.max(gcn_logvar_max, gat_logvar_max)
        z_max = torch.max(z_gcn_max, z_gat_max)
        return self.decocder.forward_all(z_max, sigmoid=True), mu_max, logvar_max