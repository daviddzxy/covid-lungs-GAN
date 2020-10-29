import numpy as np
import scipy
from torch import from_numpy


class PadVolume:
    """
    Pads volume with zeros.
    """

    def __init__(self, dimensions):
        """
        :param dimensions: Output dimensions(H_out, W_out, D_out) of padded volume.
        """
        assert len(dimensions.shape) == 3
        self.dimensions = dimensions

    def __call__(self, volume):
        """
        :param volume: Volume with size (H_in, W_in, D_in) to be padded.
        :return: Returns padded volume with zeros with size H_out, W_out, D_out).
        """
        pad1, pad2, pad3 = self.dimensions[0] - volume.shape[0], \
                           self.dimensions[1] - volume.shape[1], \
                           self.dimensions[2] - volume.shape[2],

        pad1, pad2, pad3 = int(pad1 / 2), int(pad2 / 2), int(pad3 / 2)
        odd1, odd2, odd3 = 0, 0, 0

        if pad1 // 2:
            odd1 = 1
        if pad2 // 2:
            odd2 = 1
        if pad3 // 2:
            odd3 = 1

        return np.pad(volume, ((pad1, pad1 + odd1), (pad2, pad2 + odd2), (pad3, pad3 + odd3)), mode='constant')


class GetMiddleLungSlice:
    """
    Returns middle slice of segmented lung volume.
    """

    def __call__(self, volume):
        """
        :param volume: Volume of input size (H_in, W_in, D_in).
        :return: Returns middle slice(H_out, W_out) of segmented lung volume of size (H_in, W_in, D_in).
        """
        # Remove all elements from rows and columns that contain zeroes only.
        lung_slices = volume[:, :, ~(volume == 0).all(axis=(0, 1))]
        return lung_slices[:, :, lung_slices.shape[2] // 2]


class ResampleVolume:
    """
    Resample volume to different voxel_spacing.
    Voxel spacing (1, 1, 1) means that one pixel represents volume of 1x1x1mm in real world.
    """
    def __init__(self, new_spacing):
        """
        :param new_spacing: new_spacing has to be a tuple of length 3.
        """
        self.new_spacing = new_spacing

    def __call__(self, volume, current_spacing):
        """
        :param volume: Volume to be resampled of size (H_in, W_in, D_in).
        :param current_spacing: Current voxel spacing values. current spacing has to be a tuple of length 3.
        :return: Returns volume with new size of (H_out, W_out, D_out)
                 and tuple of length 3 with corresponding resize factors. 
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


class Scale:
    """
    Scale nd array between values [out_min, out_max].
    """
    def __init__(self, out_min, out_max):
        self.out_min = out_min
        self.out_max = out_max

    def __call__(self, ndarray):
        return (self.out_max - self.out_min) * ((ndarray - np.min(ndarray)) / (np.max(ndarray))) + self.out_min