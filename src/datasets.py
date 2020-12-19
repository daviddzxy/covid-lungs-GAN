import os
import config
import pickle
from torch.utils.data import Dataset


class CycleGanDataset(Dataset):
    def __init__(self, transforms=None):
        self.paths_A = os.listdir(config.training_data["A"])
        self.paths_B = os.listdir(config.training_data["B"])
        self.transforms = transforms

    def __getitem__(self, index):
        file_handler_A = open(os.path.join(config.training_data["A"], self.paths_A[index % len(self.paths_A)]), "rb")
        file_handler_B = open(os.path.join(config.training_data["B"], self.paths_B[index % len(self.paths_B)]), "rb")
        image_A = pickle.load(file_handler_A)["data"]
        image_B = pickle.load(file_handler_B)["data"]
        if self.transforms:
            image_A, image_B = self.transforms(image_A), self.transforms(image_B)

        file_handler_A.close()
        file_handler_B.close()
        return image_A, image_B

    def __len__(self):
        return max(len(self.paths_A), len(self.paths_B))
