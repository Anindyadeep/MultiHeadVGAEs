import sys
import time 
import warnings
from pathlib import Path

import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 

import torch_geometric.nn as gnn

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(str(BASE_DIR) + "/")
warnings.filterwarnings("ignore")

from utils import metrics, input_data
from Models import base_model, gcn_gcn_merge, multi_head_gcn, gcn_gat, multi_head_gcn_gat, gcn_gat_merge
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SampleConv(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2):
        super(SampleConv, self).__init__()
        """
        This SampleConv is just a Fake Encoder in order to make compatible with the PyG encoder, but we are 
        not using any sort of features of this encoder.
        """
        self.gc_conv = gnn.GCNConv(input_feat_dim, hidden_dim1)
        self.mu_conv = gnn.GCNConv(hidden_dim1, hidden_dim2)
        self.logvar_conv = gnn.GCNConv(hidden_dim1, hidden_dim2)

    def forward(self, x, adj):
        hidden = self.gc_conv(x, adj)
        hidden = F.relu(hidden)
        return self.mu_conv(hidden, adj), self.logvar_conv(hidden, adj)


class TorchTrain(object):
    def __init__(self, parameters):
        self.data, self.all_edge_index = input_data.Dataset(root = "auto", 
                                                name = parameters['dataset'], 
                                                transform=None).get_train_test_splted_graph_data(valid = 0.05, 
                                                                                           test = 0.1, 
                                                                                           return_all_edge=True)
        self.hidden_dim1 = parameters['hidden_dim1']
        self.hidden_dim2 = parameters['hidden_dim2']
        self.num_heads = parameters['num_heads']
        self.lr = parameters['lr']
        self.epoch = parameters['epochs']
        model_name = parameters['model']
        optimizer_name = parameters['optimizer']
        heads_list = parameters['heads'].split('-')

        self.data = self.data.to(device)
        self.models = {
            "base_model" : base_model.BaseVGAE(self.data.x.shape[1], self.hidden_dim1, self.hidden_dim2).to(device),
            "multi_head_gcn" : multi_head_gcn.MultiHeadGcnGVAE(self.data.x.shape[1], self.hidden_dim1, self.hidden_dim2, self.num_heads).to(device),
            "gcn_gat" : gcn_gat.GCNGATVGAE(self.data.x.shape[1], self.hidden_dim1, self.hidden_dim2, self.num_heads).to(device),
            "gcn_gat_cat" : gcn_gat.GCNGATVGAE(self.data.x.shape[1], self.hidden_dim1, self.hidden_dim2, self.num_heads).to(device),
            "multi_head_gcn_gat" : multi_head_gcn_gat.MultiHeadGCNGATVGAE(self.data.x.shape[1], self.hidden_dim1, self.hidden_dim2, arch_list=heads_list, num_heads_gat=self.num_heads).to(device),
            "gcn_gcn_merge" : gcn_gcn_merge.GCNMerge(self.data.x.shape[1], self.hidden_dim1, self.hidden_dim2).to(device),

            # best model
            "gcn_gat_merge" : gcn_gat_merge.GCNMerge(self.data.x.shape[1], self.hidden_dim1, self.hidden_dim2, self.num_heads).to(device)
            
            # multi-head gcn-gat merge is only model which is left to make.

        }

        self.model = self.models[model_name]
        # printing the model
        print(self.model)
        self.model_encoder = SampleConv(self.data.x.shape[1], self.hidden_dim1, self.hidden_dim2)

        self.optimizers = {
            'adam'  : optim.Adam(self.model.parameters(), lr = self.lr),
            'adamw' : optim.AdamW(self.model.parameters(), lr = self.lr),
            'rms'   : optim.RMSprop(self.model.parameters(), lr = self.lr),
            'sgd'   : optim.SGD(self.model.parameters(), lr = self.lr)
        }
        
        self.optimizer =  self.optimizers[optimizer_name]
        self.criterion = metrics.VGAEMetrics(encoder = self.model_encoder)

    def train(self):
        time_total_start = time.time()
        for epoch in range(self.epoch):
            t = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            adj_recon, mu, logvar = self.model(self.data.x, self.all_edge_index)
            loss = self.criterion.loss_fn(mu=mu,
                                    logvar=logvar,
                                    pos_edge_index=self.data.train_pos_edge_index,
                                    all_edge_index=self.all_edge_index)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                self.model.eval()
                roc_auc, ap = self.criterion.single_test(mu = mu,
                                                    logvar=logvar,
                                                    test_pos_edge_index=self.data.test_pos_edge_index,
                                                    test_neg_edge_index=self.data.test_neg_edge_index)
                print(f"TESTING | AT epoch {'00' + str(epoch + 1) if epoch + 1 < 10 else ('0' +  str(epoch + 1) if epoch + 1 < 100 else str(epoch + 1))}, loss: {loss.item():.4f}, ROC_AUC: {roc_auc:.4f}, AP: {ap:.4f}, Time: {(time.time() - t):.4f}")
        
        print(f"The total time taken: {(time.time() - time_total_start):.4f}")
        return self.model