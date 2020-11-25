import os
import config
import pickle
from src.transformations import Normalize, ToTensor
from torch.utils.data import Dataset


class CycleGanDataset(Dataset):
    def __init__(self):
        self.paths_A = os.listdir(config.training_data["A"])
        self.paths_B = os.listdir(config.training_data["B"])
        self.normalize = Normalize()
        self.to_tensor = ToTensor()

    def __getitem__(self, index):
        file_handler_CT0 = open(os.path.join(config.training_data["A"], self.paths_A[index % len(self.paths_A)]), "rb")
        file_handler_CT3 = open(os.path.join(config.training_data["B"], self.paths_B[index % len(self.paths_B)]), "rb")
        image_CT0 = pickle.load(file_handler_CT0)["data"]
        image_CT3 = pickle.load(file_handler_CT3)["data"]
        image_CT0, image_CT3 = self.normalize(image_CT0), self.normalize(image_CT3)
        image_CT0, image_CT3 = self.to_tensor(image_CT0), self.to_tensor(image_CT3)
        file_handler_CT0.close()
        file_handler_CT3.close()
        return image_CT0, image_CT3

    def __len__(self):
        return max(len(self.paths_A), len(self.paths_B))
