import os
import pickle
import random
from torch.utils.data import Dataset
from transformations import ToTensor


class CoivdLungHealthyLungDataset(Dataset):
    def __init__(self, images_A, images_B, mask=None, rotation=None, crop=None, normalize=None):
        self.dir_A = images_A
        self.dir_B = images_B
        self.files_A = os.listdir(images_A)
        self.files_B = os.listdir(images_B)
        self.mask = mask
        self.rotate = rotation
        self.crop = crop
        self.normalize = normalize
        self.to_tensor = ToTensor()

    def __getitem__(self, index):
        file_handler_A = open(os.path.join(self.dir_A, self.files_A[index % len(self.files_A)]), "rb")
        file_handler_B = open(os.path.join(self.dir_B, self.files_B[index % len(self.files_B)]), "rb")
        file_A = pickle.load(file_handler_A)
        file_B = pickle.load(file_handler_B)
        image_A, image_B = file_A["data"], file_B["data"]
        mask_A, mask_B = file_A["mask"], file_B["mask"]

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


class CoivdLungMaskLungDataset(Dataset):
    def __init__(self, images, mask_covid, mask_lungs=None, max_rotation=None, rotation=None, crop=None,
                 normalize=None):
        self.dir = images
        self.files = os.listdir(images)
        self.mask_covid = mask_covid
        self.mask_lungs = mask_lungs
        self.max_rotation = max_rotation
        self.rotate = rotation
        self.crop = crop
        self.normalize = normalize
        self.to_tensor = ToTensor()

    def __getitem__(self, index):
        file_handler = open(os.path.join(self.dir, self.files[index]), "rb")
        file = pickle.load(file_handler)
        image, image_mask = file["data"], file["mask"]
        if self.normalize:
            image = self.normalize(image)

        if self.mask_lungs:
            image = self.mask_lungs(image, image_mask)

        masked_image = self.mask_covid(image, image_mask)

        if self.rotate:
            rand_angle = random.randint(-self.max_rotation, self.max_rotation)
            image, masked_image = self.rotate(image, rand_angle), self.rotate(masked_image, rand_angle)

        if self.crop:
            image, masked_image = self.crop(image), self.crop(masked_image)

        image, masked_image = self.to_tensor(image), self.to_tensor(masked_image)

        return image, masked_image

    def __len__(self):
        return len(self.files)

