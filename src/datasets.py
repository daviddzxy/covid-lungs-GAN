import os
import config
import pickle
from torch.utils.data import Dataset
from transformations import Normalize, ToTensor
from torchvision import transforms


class CycleGanDataset(Dataset):
    def __init__(self, images_A, images_B, metadata, _transforms=None):
        self.dir_A = images_A
        self.dir_B = images_B
        self.files_A = os.listdir(images_A)
        self.files_B = os.listdir(images_B,)
        with open(metadata, "rb") as handle:
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
        file_handler_A = open(os.path.join(self.dir_A, self.files_A[index % len(self.files_A)]), "rb")
        file_handler_B = open(os.path.join(self.dir_B, self.files_B[index % len(self.files_B)]), "rb")
        image_A = pickle.load(file_handler_A)["data"]
        image_B = pickle.load(file_handler_B)["data"]
        if self.transforms:
            image_A, image_B = self.transforms(image_A), self.transforms(image_B)

        file_handler_A.close()
        file_handler_B.close()
        return image_A, image_B

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
