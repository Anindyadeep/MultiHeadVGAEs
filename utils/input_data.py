import os 
import sys 
import warnings
from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import NormalizeFeatures


BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
warnings.filterwarnings("ignore")


class Dataset(object):
    def __init__(self, root, name, transform=None):
        self.root = root 
        self.name = name 
        self.transform = transform if transform is not None else NormalizeFeatures

    def get_dataset(self):
        if self.root == "auto":
            path_to_make = os.path.join(BASE_DIR, "Data")
            if not os.path.isdir(path_to_make):
                os.mkdir(path_to_make)
            dataset = Planetoid(root = path_to_make, name = self.name, transform=self.transform())
            return dataset 
        
        dataset =  Planetoid(root = self.root, name = self.name, transform=self.transform())
        return dataset
    
    def get_graph_data(self):
        self.dataset = self.get_dataset()
        return self.dataset[0]
    
    def get_train_test_splted_graph_data(self, valid, test, return_all_edge = False):
        graph_data = self.get_graph_data()
        all_edge_index = graph_data.edge_index
        data = train_test_split_edges(graph_data, val_ratio=valid, test_ratio=test)
        return (data, all_edge_index) if return_all_edge else data 

