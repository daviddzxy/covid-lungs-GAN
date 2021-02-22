import numpy as np
import random
from torch import from_numpy
from scipy import ndimage


class SelectCovidSlices:
    """
    Selects slices with tissue damaged by covid.
    """

    def __init__(self, covid_tissue_value, min_covid_pixels):
        """
        :param covid_tissue_value: Integer value of masked area marking damage caused by covid.
        :param min_covid_pixels: Sum of pixels that has to be present in the image in order to
         return the scan slice in result list.
        """
        self.covid_tissue_value = covid_tissue_value
        self.min_covid_pixels = min_covid_pixels

    def __call__(self, mask, volume):
        new_mask = []
        new_volume = []
        for i in range(mask.shape[2]):
            if np.any(np.isin(mask[:, :, i], self.covid_tissue_value)) and \
               np.sum(mask[:, :, i] == self.covid_tissue_value) > self.min_covid_pixels:
                new_mask.append(mask[:, :, i])
                new_volume.append(volume[:, :, i])

        return new_mask, new_volume


class Crop:
    """
    Crop image.
    """
    def __init__(self, dimensions):
        """
        :param dimensions: Output shape(H_out, W_out) of cropped image.
        """
        assert len(dimensions) == 2
        self.dim_x = dimensions[0] // 2
        self.dim_y = dimensions[1] // 2

    def __call__(self, image):
        """
        :param image: Image of shape (H_in, W_in) to be cropped.
        return: Cropped image of shape (H_out, W_out).
        """
        y_mid = image.shape[0] // 2
        x_mid = image.shape[1] // 2
        return image[y_mid - self.dim_y:y_mid + self.dim_y, x_mid - self.dim_x:x_mid + self.dim_x]


class RandomRotation:
    """
    Randomly rotates image.
    """
    def __init__(self, max_rotation):
        """
        :param max_rotation: Max angle of random rotation in degrees.
        """
        self.max_rotation = max_rotation

    def __call__(self, image):
        """
        :param image: Image with shape (H_in, W_in).
        :return: Returns rotated image with shape (W_in, W_in).
        """
        rand_angle = random.randint(-self.max_rotation, self.max_rotation)
        return ndimage.rotate(image, angle=rand_angle, reshape=False)


class Rotation:
    """
    Rotates image.
    """
    def __call__(self, image, angle):
        return ndimage.rotate(image, angle=angle, reshape=False)


class PadVolume:
    """
    Pads volume with zeros.
    """
    def __init__(self, dimensions):
        """
        :param dimensions: Output shape(H_out, W_out, D_out) of padded volume.
        """
        assert len(dimensions) == 3
        self.dimensions = dimensions

    def __call__(self, volume):
        """
        :param volume: Volume of shape (H_in, W_in, D_in) to be padded.
        :return: Returns padded volume with zeros with shape H_out, W_out, D_out).
        """
        pad1, pad2, pad3 = self.dimensions[0] - volume.shape[0], \
                           self.dimensions[1] - volume.shape[1], \
                           self.dimensions[2] - volume.shape[2],

        pad1, pad2, pad3 = int(pad1 / 2), int(pad2 / 2), int(pad3 / 2)
        odd1, odd2, odd3 = 0, 0, 0

        odd1 = 1 if volume.shape[0] % 2 else odd1
        odd2 = 1 if volume.shape[1] % 2 else odd2
        odd3 = 1 if volume.shape[2] % 2 else odd3

        padded = np.pad(volume, ((pad1, pad1 + odd1), (pad2, pad2 + odd2), (pad3, pad3 + odd3)), mode='constant')
        return padded


class GetMiddleSlices:
    """
    Get n middle slices from volume.
    """
    def __init__(self, n):
        """
        :param n: Number of slices to be selected.
        """
        self.n = n

    def __call__(self, volume):
        """
        :param volume: Volume of shape (H_in, W_in, D_in).
        :return: Returns list of slices of shape (H_in, W_in).
        """
        assert self.n <= volume.shape[2]
        middle_slice_idx = volume.shape[2] // 2
        bottom_slice_idx = middle_slice_idx - (self.n // 2)
        slices = [None] * self.n
        for i, slice_idx in enumerate(range(bottom_slice_idx, bottom_slice_idx + self.n)):
            slices[i] = volume[:, :, slice_idx]

        assert len(slices) == self.n
        return slices


class RemoveEmptySlices:
    """
    Remove empty padding slices.
    Empty slice means that columns and rows of the slice contain zero elements only.
    """
    def __call__(self, volume):
        """
        :param volume: Volume of input shape (H_in, W_in, D_in).
        :return: Returns volume of shape (H_in, W_in, D_out).
        """
        return volume[:, :, ~(volume == 0).all(axis=(0, 1))]


class ResampleVolume:
    """
    Resample volume to different voxel_spacing.
    Voxel spacing (1, 1, 1) means that one pixel represents volume of 1x1x1mm in real world.
    """
    def __init__(self, new_spacing):
        """
        :param new_spacing: Tuple of integers of new voxel_spacing.
        """
        self.new_spacing = new_spacing

    def __call__(self, volume, current_spacing):
        """
        :param volume: Volume to be resampled of shape (H_in, W_in, D_in).
        :param current_spacing: Current voxel spacing values. Current spacing has to be a tuple of length equal to
        length of self.new_spacing.
        :return: Returns volume of shape (H_out, W_out, D_out)
                 and tuple with corresponding resize factors.
        """
        resize_factor = current_spacing / self.new_spacing
        new_shape = np.round(volume.shape * resize_factor)
        real_resize_factor = new_shape / volume.shape
        resampled_volume = ndimage.interpolation.zoom(volume, real_resize_factor, mode='nearest')
        return resampled_volume, real_resize_factor


class ApplyMask:
    """
    Applies mask to volume.
    """
    def __init__(self, value_to_mask):
        self.value_to_mask = value_to_mask

    def __call__(self, volume, mask):
        """
        :param volume: Volume of size (H_in, W_in, D_in) to be masked.
        :param mask: Mask of shape (H_in, W_in, D_in), which contains nonzero elements,
                     which will be used to keep relevant infromation in volume.
        :return: Returns masked volume.
        """
        return np.where(mask == self.value_to_mask, mask, volume)


class ToTensor:
    """
    Transforms numpy array to tensor.
    """
    def __call__(self, ndarray):
        """
        :param ndarray: If ndarray is of shape (H_in, W_in), then ndarray is transformed into (C_out=1, H_out, W_out).
                        If ndarray is of shape (H_in, W_in, D_in), then ndarray is transformed
                        into (C_out=1, D_out, H_out, W_out).
        :return: Returns corresponding tensor, which is compatible with conv2d or conv3d torch modules.
        """
        if len(ndarray.shape) == 2:
            image = ndarray[np.newaxis, :, :]
            return from_numpy(image)

        if len(ndarray.shape) == 3:
            volume = ndarray.transpose(2, 0, 1)
            volume = volume[np.newaxis, :, :, :]
            return from_numpy(volume)


class Normalize:
    """
    Normalize ndarray between values -1, 1.
    """
    def __init__(self, dataset_min, dataset_max):
        self.min = dataset_min
        self.max = dataset_max

    def __call__(self, ndarray):
        return 2 * ((ndarray - self.min) / (self.max - self.min)) - 1
