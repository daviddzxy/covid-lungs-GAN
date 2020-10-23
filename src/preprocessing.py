import logging
import nibabel as nib
import SimpleITK as sitk
import os
import numpy as np
import scipy
from lungmask import mask


def resample(image, pix_dims, new_spacing=(1, 1, 8)):
    resize_factor = pix_dims / new_spacing
    new_shape = np.round(image.shape * resize_factor)
    real_resize_factor = new_shape / image.shape
    resampled_image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return resampled_image


def _preprocess():
    for key, path in config.data_paths.items():
        for f in os.listdir(path):
            logging.info("Preprocessing of file {} started".format(f))
            sitk_img = sitk.ReadImage(os.path.join(path, f))
            nib_img = nib.load(os.path.join(path, f))  # mask function from lungmask module works with sitk objects
            # only, might implement adapter for this object so the image does not have to be loaded two times

            segmented_vol = mask.apply(sitk_img, batch_size=1)
            resampled_vol = resample(nib_img.get_fdata().transpose(1, 0, 2), nib_img.header['pixdim'][1:4])
            resampled_segmented_vol = resample(segmented_vol.transpose(1, 2, 0), nib_img.header['pixdim'][1:4])
            resampled_masked_vol = np.where(resampled_segmented_vol == 0, resampled_segmented_vol, resampled_vol)
            np.save(os.path.join(config.preprocessed_data_paths[key], f.split(".")[0] + ".npy"), resampled_masked_vol)
            logging.info("Preprocessing of file {}  finished".format(f))


if __name__ == '__main__':
    import config
    os.sys.path.append(config.project_root)
    logging.basicConfig(filename=os.path.join(config.log_dir, "preprocessing_logs.log"),
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S',
                        filemode='w',
                        level=logging.INFO,
                        force=True)
    _preprocess()


