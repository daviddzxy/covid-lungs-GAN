import os
import config
import pickle
from torch.utils.data import Dataset
from transformations import Normalize, ToTensor
from torchvision import transforms


class CycleGanDataset(Dataset):
    def __init__(self, _transforms=None):
        self.paths_A = os.listdir(config.cyclegan_data_train["A"])
        self.paths_B = os.listdir(config.cyclegan_data_train["B"])
        with open(config.dataset_metadata, "rb") as handle:
            self._dataset_metadata = pickle.load(handle)

        self.transforms = []
        if _transforms:
            self.transforms = _transforms
        self.transforms.extend([
            Normalize(self._dataset_metadata["min"], self._dataset_metadata["max"]),
            ToTensor()
        ])

        self.transforms = transforms.Compose(self.transforms)

    def __getitem__(self, index):
        file_handler_A = open(os.path.join(config.cyclegan_data_train["A"], self.paths_A[index % len(self.paths_A)]), "rb")
        file_handler_B = open(os.path.join(config.cyclegan_data_train["B"], self.paths_B[index % len(self.paths_B)]), "rb")
        image_A = pickle.load(file_handler_A)["data"]
        image_B = pickle.load(file_handler_B)["data"]
        if self.transforms:
            image_A, image_B = self.transforms(image_A), self.transforms(image_B)

        file_handler_A.close()
        file_handler_B.close()
        return image_A, image_B

    def __len__(self):
        return max(len(self.paths_A), len(self.paths_B))
