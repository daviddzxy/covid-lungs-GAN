import logging
import nibabel as nib
import SimpleITK as sitk
import os
import pickle
import config
from lungmask import mask
from transformations import ResampleVolume, ApplyMask, PadVolume, GetMiddleSlices, RemoveEmptySlices

os.sys.path.append(config.project_root)
logging.basicConfig(filename=os.path.join(config.preprocessing_logs, "preprocessing_logs.log"),
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S",
                    filemode="w",
                    level=logging.INFO,
                    force=True)

apply_mask = ApplyMask()
pad_volume = PadVolume(config.padding_shape)
get_middle_lung_slice = GetMiddleSlices(n=5)
resample = ResampleVolume(new_spacing=(1, 1, 8))
remove_empty_slices = RemoveEmptySlices()

for key, path in config.data.items():
    for f in os.listdir(path):
        try:
            sitk_img = sitk.ReadImage(os.path.join(path, f))
            nib_img = nib.load(os.path.join(path, f))
            img = nib_img.get_fdata().transpose(1, 0, 2)
            img_s = mask.apply(sitk_img, batch_size=1, noHU=False).transpose(1, 2, 0)
            img, resize_factor = resample(img, nib_img.header["pixdim"][1:4])
            img_s, _ = resample(img_s, nib_img.header["pixdim"][1:4])
            img, img_s = pad_volume(img), pad_volume(img_s)
            img, img_s = remove_empty_slices(img), remove_empty_slices(img_s)
            slices, slices_s = get_middle_lung_slice(img), get_middle_lung_slice(img_s)
            for i, (_slice, _slice_s) in enumerate(zip(slices, slices_s)):
                with open(
                        os.path.join(config.cyclegan_data[key], f.split(".")[0] + "slice " + str(i) + ".pkl"), "wb"
                ) as handle:
                    pickle.dump({"data": _slice, "mask": _slice_s, "resize_factor": resize_factor}, handle)
            logging.info("Preprocessing of file {} finished".format(f))
        except Exception as err:
            logging.error("Error occured while preprocessing file {}, error: {}".format(f, err))
