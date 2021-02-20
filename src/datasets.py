import os
import pickle
from torch.utils.data import Dataset
from transformations import ToTensor


class CycleGanDataset(Dataset):
    def __init__(self, images_A, images_B, mask=None, rotation=None, crop=None, normalize=None):
        self.dir_A = images_A
        self.dir_B = images_B
        self.files_A = os.listdir(images_A)
        self.files_B = os.listdir(images_B, )
        self.mask = mask
        self.rotate = rotation
        self.crop = crop
        self.normalize = normalize
        self.to_tensor = ToTensor()

    def __getitem__(self, index):
        file_handler_A = open(os.path.join(self.dir_A, self.files_A[index % len(self.files_A)]), "rb")
        file_handler_B = open(os.path.join(self.dir_B, self.files_B[index % len(self.files_B)]), "rb")
        A = pickle.load(file_handler_A)
        B = pickle.load(file_handler_B)
        image_A, image_B = A["data"], B["data"]
        mask_A, mask_B = A["mask"], B["mask"]

        if self.normalize:
            image_A, image_B = self.normalize(image_A), self.normalize(image_B)

        if self.mask:
            image_A, image_B = self.mask(image_A, mask_A), self.mask(image_B, mask_B)

        if self.rotate:
            image_A, image_B = self.rotate(image_A), self.rotate(image_B)

        if self.crop:
            image_A, image_B = self.crop(image_A), self.crop(image_B)

        image_A, image_B = self.to_tensor(image_A), self.to_tensor(image_B)

        file_handler_A.close()
        file_handler_B.close()
        return image_A, image_B

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
