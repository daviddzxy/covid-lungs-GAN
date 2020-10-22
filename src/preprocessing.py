import nibabel as nib
import SimpleITK as sitk
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
from lungmask import mask


def resample(image, pix_dims, new_spacing=(1, 1, 8)):
    resize_factor = pix_dims / new_spacing
    new_shape = np.round(image.shape * resize_factor)
    real_resize_factor = new_shape / image.shape
    resampled_image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return resampled_image


def preprocess():
    for key, path in config.data_paths.items():
        for f in os.listdir(path):
            sitk_img = sitk.ReadImage(os.path.join(path, f))
            nib_img = nib.load(os.path.join(path, f))  # mask function from lungmask module works with sitk objects
            # only, might implement adapter for this object so the image does not have to be loaded two times

            segmented_vol = mask.apply(sitk_img, batch_size=1)
            resampled_vol = resample(nib_img.get_fdata().transpose(1, 0, 2), nib_img.header['pixdim'][1:4])
            resampled_segmented_vol = resample(segmented_vol.transpose(1, 2, 0), nib_img.header['pixdim'][1:4])
            resampled_masked_vol = np.where(resampled_segmented_vol == 0, resampled_segmented_vol, resampled_vol)

if __name__ == '__main__':
    os.sys.path.append('../')
    import config
    preprocess()

