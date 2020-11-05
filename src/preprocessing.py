import logging
import nibabel as nib
import SimpleITK as sitk
import os
import pickle
import config
from lungmask import mask
from transformations import ResampleVolume, ApplyMask
import matplotlib.pyplot as plt


if __name__ == '__main__':
    os.sys.path.append(config.project_root)
    logging.basicConfig(filename=os.path.join(config.log_dir, 'preprocessing_logs.log'),
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S',
                        filemode='w',
                        level=logging.INFO,
                        force=True)

    apply_mask = ApplyMask()
    resample = ResampleVolume(new_spacing=(1, 1, 8))
    for key, path in config.data_paths.items():
        for f in os.listdir(path):
            logging.info('Preprocessing of file {} started'.format(f))
            sitk_img = sitk.ReadImage(os.path.join(path, f))
            # mask function from lungmask module works with sitk objects only,
            # I might implement adapter for this object so the image does not have to be loaded two times
            nib_img = nib.load(os.path.join(path, f))
            img = nib_img.get_fdata().transpose(1, 0, 2)
            img_s = mask.apply(sitk_img, batch_size=1, noHU=False).transpose(1, 2, 0)
            img = apply_mask(img, img_s)
            img, resize_factor = resample(img, nib_img.header['pixdim'][1:4])
            logging.info('Volume shape before resampling: {}, volume shape after resampling {}'.format(
                nib_img.get_fdata().shape, img.shape)
            )
            with open(os.path.join(config.preprocessed_data_paths[key], f.split('.')[0] + '.pkl'), 'wb') as handle:
                pickle.dump({'data': img, 'resize_factor': resize_factor}, handle)
            logging.info('Preprocessing of file {} finished'.format(f))

