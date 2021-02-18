import logging
import config
import os
import re
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import pickle
from lungmask import mask as segmentation_mask

from transformations import ResampleVolume, PadVolume, SelectCovidSlices

os.sys.path.append(config.project_root)
logging.basicConfig(filename=os.path.join(config.preprocessing_logs, "preprocessing_logs.log"),
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S",
                    filemode="w",
                    level=logging.INFO,
                    force=True)

resample = ResampleVolume(new_spacing=(1, 1, 8))
pad_volume = PadVolume(config.padding_shape)
select_slices = SelectCovidSlices(config.covid_tissue_value, config.min_covid_pixels)

for mask_file in os.listdir(config.masks):
    try:
        mask_id = re.search(r"(.+)\_(.+)\_.+", mask_file).group(2)
        # masks are available only for images from category CT1
        study_path = os.path.join(os.path.join(config.data["CT1"], "study_{}.nii.gz".format(mask_id)))
        mask_nib = nib.load(os.path.join(config.masks, mask_file))
        scan_nib = nib.load(study_path)
        scan_sitk = sitk.ReadImage(study_path)
        mask_nib_data = mask_nib.get_fdata().transpose(1, 0, 2)
        scan_nib_data = scan_nib.get_fdata().transpose(1, 0, 2)
        lung_mask = segmentation_mask.apply(scan_sitk, batch_size=1, noHU=False).transpose(1, 2, 0)
        combined_masks = np.where(mask_nib_data == 1, config.covid_tissue_value, lung_mask)
        scan_nib_data, resize_factor = resample(scan_nib_data, scan_nib.header["pixdim"][1:4])
        combined_masks, _ = resample(combined_masks, scan_nib.header["pixdim"][1:4])
        scan_nib_data = pad_volume(scan_nib_data)
        combined_masks = pad_volume(combined_masks)
        combined_masks, scan_nib_data = select_slices(combined_masks, scan_nib_data)
        for i, (mask, scan) in enumerate(zip(combined_masks, scan_nib_data)):
            with open(
                    os.path.join(config.preprocessed_data_cgan, "image" + mask_id + "slice " + str(i) + ".pkl"), "wb"
            ) as handle:
                pickle.dump({"data": scan, "mask": mask, "resize_factor": resize_factor}, handle)
        logging.info("Preprocessing of file {} finished".format(mask_id))
    except Exception as err:
        logging.error("Error occured while preprocessing file {}, error: {}".format(mask_id, err))





