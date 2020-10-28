import os
import config
import numpy as np
import pickle
from torch.utils.data import Dataset
from torch import from_numpy

class CycleGanDataset(Dataset):
    def __init__(self):
        self.CT0 = os.listdir(config.preprocessed_data_paths['CT0'])
        self.CT3 = os.listdir(config.preprocessed_data_paths['CT3'])


    def __getitem__(self, index):
        CT0_file = open(self.CT0[index % len(self.CT0)], 'rb')
        pickle.load(CT0_file)


    def __len__(self):
        return max(self.CT0, self.CT3)