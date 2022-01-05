import os 
import sys
from pathlib import Path

import torch 
import torch.nn as nn 
import torch.optim as optim  

from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn.models import VGAE, InnerProductDecoder
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops

import warnings 

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
warnings.filterwarnings("ignore")

EPS = 1e-15
MAX_LOGSTD = 10

class VGAEMetrics(nn.Module):
    def __init__(self):
        """
        The encoder we will be using here has no link with the actual Encoder block, as fake encoder blocks, just works fine.
        We have to do so, because, since we are using the self.test() from VGAE, so super() requires encoder and decoder. 
        """
        #super(VGAEMetrics, self).__init__(encoder=encoder, decoder = InnerProductDecoder())
        super(VGAEMetrics, self).__init__()
        self.decoder = InnerProductDecoder()
    
    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def kl_loss(self, mu=None, logstd=None):
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
    
    def loss_fn(self, mu, logvar, pos_edge_index, all_edge_index):
        z = self.reparametrize(mu, logvar)
        pos_loss =  -torch.log(self.decoder(z, pos_edge_index, sigmoid=True)).mean()

        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)
        neg_edge_index = negative_sampling(all_edge_index_tmp, z.size(0), pos_edge_index.size(1))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

        kl_loss = 1 / z.size(0) * self.kl_loss(mu, logvar)
        return pos_loss + neg_loss + kl_loss
    
    def test(self, z, pos_edge_index, neg_edge_index):
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

    def single_test(self, mu, logvar, test_pos_edge_index, test_neg_edge_index):
        with torch.no_grad():
            z = self.reparametrize(mu, logvar)
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score
