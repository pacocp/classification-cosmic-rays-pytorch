import torch
from torch.utils import data


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, inputs, labels):
        'Initialization'
        self.labels = labels
        self.inputs = inputs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)
        
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        x = torch.from_numpy(self.inputs[index]).float()
        y = torch.from_numpy(self.labels[index]).float()   
        return x, y
