import glob
import os

import SimpleITK as sitk
import numpy as np
from medpy.metric.binary import assd


# compute ASSD
def compute_mean_assd(result_path, reference_path):
    results = glob.glob(os.path.join(result_path, '*.nii.gz'))
    assds = []
    for each_result in results:
        reference = os.path.join(reference_path, os.path.basename(each_result))

        result_im = sitk.ReadImage(each_result)
        reference_im = sitk.ReadImage(reference)
        voxel_spacing = reference_im.GetSapcing()

        result_arr = sitk.GetArrayFromImage(result_im)
        reference_arr = sitk.GetArrayFromImage(result_im)
        assd_value = assd(result_arr, reference_arr, voxel_spacing)
        assds.append(assd_value)

    return np.mean(assds)
