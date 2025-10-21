import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader, Batch
import pandas as pd
import os

# create dataset class 
class GraphImageDataset(Dataset):
    def __init__(self, root, sub_list, dataset='CBT'):
        self.root = root
        self.sub = sub_list
        self.path = []
        self.dataset=dataset
        if dataset=='CBT':
            for sub in sub_list:
                self.path.append(root+sub+'_0_0')
        elif dataset=='ABIDE':
            for sub in sub_list:
                self.path.append(root+sub)
        self.path=np.array(self.path)
        print('dataset length: ', len(self.path))
        
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        # base data instance
        data = torch.load(self.path[idx])
        if self.dataset=='CBT':
            data.img=torch.tensor(data.img)[None,:,:,:]
        return data

def load_fc_data(root, sub_list):
    path=[]
    for sub in sub_list:
        path.append(root+sub)
    x_arr = []
    y_arr = []
    for filename in path:
        data = torch.load(filename)
        x_arr.append(data['x'])
        y_arr.append(data['y'])
    return x_arr, y_arr

# create dataset class 
class FC_Dataset(Dataset):
    def __init__(self, root, sub_list):
        x_arr, y_arr = load_fc_data(root, sub_list)
        self.x = x_arr
        self.y = y_arr
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y 
