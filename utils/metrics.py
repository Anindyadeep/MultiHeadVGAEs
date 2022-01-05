import os 
import sys
from pathlib import Path

import torch 
import torch.nn as nn 
import torch.optim as optim  

from torch_geometric.nn.models import VGAE, InnerProductDecoder
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops

import warnings 

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
warnings.filterwarnings("ignore")

class VGAEMetrics(VGAE):
    def __init__(self, encoder):
        """
        The encoder we will be using here has no link with the actual Encoder block, as fake encoder blocks, just works fine.
        We have to do so, because, since we are using the self.test() from VGAE, so super() requires encoder and decoder. 
        """
        super(VGAEMetrics, self).__init__(encoder=encoder, decoder = InnerProductDecoder())
    
    def loss_fn(self, mu, logvar, pos_edge_index, all_edge_index):
        z = self.reparametrize(mu, logvar)
        pos_loss =  -torch.log(self.decoder(z, pos_edge_index, sigmoid=True)).mean()

        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)
        neg_edge_index = negative_sampling(all_edge_index_tmp, z.size(0), pos_edge_index.size(1))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

        kl_loss = 1 / z.size(0) * self.kl_loss(mu, logvar)
        return pos_loss + neg_loss + kl_loss
    
    def single_test(self, mu, logvar, test_pos_edge_index, test_neg_edge_index):
        with torch.no_grad():
            z = self.reparametrize(mu, logvar)
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score
