import numpy as np
import scipy
from torch import from_numpy


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
        :param volume: Volume with shape (H_in, W_in, D_in) to be padded.
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
        :return: List of slcies of shape (H_out, W_out)
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
        :return: Returns only nonzero slices.
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
        :param volume: Volume to be resampled of size (H_in, W_in, D_in).
        :param current_spacing: Current voxel spacing values. Current spacing has to be a tuple of length equal to
        length of self.new_spacing.
        :return: Returns volume with new size of (H_out, W_out, D_out)
                 and tuple with corresponding resize factors.
        """
        resize_factor = current_spacing / self.new_spacing
        new_shape = np.round(volume.shape * resize_factor)
        real_resize_factor = new_shape / volume.shape
        resampled_volume = scipy.ndimage.interpolation.zoom(volume, real_resize_factor, mode='nearest')
        return resampled_volume, real_resize_factor


class ApplyMask:
    """
    Applies mask to volume.
    """
    def __call__(self, volume, mask):
        """
        :param volume: Volume of size (H_in, W_in, D_in) to be masked.
        :param mask: Mask of size (H_in, W_in, D_in), which contains nonzero elements,
                     which will be used to keep relevant infromation in volume.
        :return: Returns masked volume.
        """
        return np.where(mask == 0, mask, volume)


class ToTensor:
    """
    Transforms numpy array to tensor.
    """
    def __call__(self, ndarray):
        """
        :param ndarray: If ndarray is of size (H_in, W_in), then ndarray is transformed into (C_out=1, H_out, W_out).
                        If ndarray is of size (H_in, W_in, D_in), then ndarray is transformed
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
    def __call__(self, ndarray):
        return (ndarray - np.mean(ndarray)) / (np.max(ndarray) - np.min(ndarray))
