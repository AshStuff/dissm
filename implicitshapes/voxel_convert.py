import glob
import json
import math
import os
from collections import defaultdict
from multiprocessing import Pool

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import trimesh
from tqdm import tqdm

from dataset import CustomRandomAffine
from itk_utils import get_affine_from_itk, \
    origin_adjust_from_pad, \
    unpad


def resample_image(itk_image, out_spacing, mask=False):
    """
    conveinence function to resample an sitk image
    """
    input_spacing = itk_image.GetSpacing()
    input_size = itk_image.GetSize()

    out_size = [int(np.round(i_size * (i_spacing / o_spacing))) for i_size, o_spacing, i_spacing in
                zip(input_size, out_spacing, input_spacing)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    if mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:

        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def scale_and_clip(sdf_im, scale_factor):
    np_im = sitk.GetArrayFromImage(sdf_im)
    np_im /= scale_factor
    np_im[np_im > 1] = 1
    np_im[np_im < -1] = -1
    new_im = sitk.GetImageFromArray(np_im)
    new_im.SetSpacing(sdf_im.GetSpacing())
    new_im.SetDirection(sdf_im.GetDirection())
    new_im.SetOrigin(sdf_im.GetOrigin())
    return new_im


def create_resample_sdfs(in_folder, out_folder, ct_folder, anchor_mesh):
    """
    This function will convert all masks in in_folder to SDFs, then normalize
    them based on anchor mesh, and then resample them based on the CTs found in 
    ct_folder (these CTs should be the padded and resampled CTs used as input to the encoder)
    anchor_mesh is typically the first mesh in Simplfiy
    """

    # grab all nii.gz files in in_folder
    filelist = glob.glob(os.path.join(in_folder, '*.nii.gz'))
    os.makedirs(out_folder, exist_ok=True)

    # get the size of the anchor mesh (in world coordinates)
    mesh = trimesh.load(anchor_mesh)
    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    norm_fact = np.max(distances)

    # to speed things up we will use multiprocessing
    pool = Pool(8)
    pbar = tqdm(total=len(filelist))

    def pbar_update(*args):
        pbar.update()

    for cur_file in filelist:
        # import pdb;pdb.set_trace()
        basename = os.path.basename(cur_file).replace('_Larynx', '')
        ct_file = os.path.join(ct_folder, basename)
        out_file = os.path.join(out_folder, basename)
        # create_resample_sdfs_func(cur_file, ct_file, out_file, norm_fact)
        # execute the sdf computation and resampling for this mask
        pool.apply_async(create_resample_sdfs_func, args=(cur_file, ct_file, out_file, norm_fact), callback=pbar_update)
        # create_resample_sdfs_func(cur_file, ct_file, out_file, norm_fact)
    pool.close()
    pool.join()
    pool.close()


def create_resample_sdfs_func(in_file, ct_file, out_file, norm_factor):
    """
    Function to compute SDF and resample for individual mask
    """
    # read the mask and compute an SDF, note we use image spacing here for the SDF values
    im_sitk = sitk.ReadImage(in_file)
    distance_map = sitk.SignedMaurerDistanceMap(im_sitk, squaredDistance=False, useImageSpacing=True)
    # normalize the SDF values based on the size of the anchor mesh
    distance_map = scale_and_clip(distance_map, norm_factor)

    # now resample the SDF to the same coordinates as the corresponding CT (padded and resampled)
    im_ct = sitk.ReadImage(ct_file)

    resample_im = sitk.Resample(distance_map, im_ct, sitk.Transform(), sitk.sitkLinear, 1)

    sitk.WriteImage(resample_im, out_file)


def resample_cts(in_folder, out_folder, out_spacing, mask=False):
    """
    resamples all CTs in a folder to the same resolution
    """
    filelist = glob.glob(os.path.join(in_folder, '*.nii.gz'))
    os.makedirs(out_folder, exist_ok=True)
    max_x = 0
    max_y = 0
    max_z = 0
    spacing_dict = defaultdict(list)
    for cur_file in tqdm(filelist):
        im_sitk = sitk.ReadImage(cur_file)
        input_spacing = im_sitk.GetSpacing()
        resample_im = resample_image(im_sitk, out_spacing, mask)

        max_x = max(max_x, resample_im.GetSize()[0])
        max_y = max(max_y, resample_im.GetSize()[1])
        max_z = max(max_z, resample_im.GetSize()[2])
        basename = os.path.basename(cur_file)
        spacing_dict[basename.replace('.nii.gz', '')] = input_spacing
        out_file = os.path.join(out_folder, basename)
        sitk.WriteImage(resample_im, out_file)
        print(max_x, max_y, max_z)
    json_obj = json.dumps(spacing_dict, indent=4)
    with open(os.path.join(out_folder, 'spacing.json'), 'w') as f:
        f.write(json_obj)


def resample_cts_v1(in_folder, out_folder, resample_dict_path, mask=False):
    filelist = glob.glob(os.path.join(in_folder, '*.nii.gz'))
    os.makedirs(out_folder, exist_ok=True)

    with open(resample_dict_path) as f:
        resample_dict = json.load(f)

    for cur_file in tqdm(filelist):
        basename = os.path.basename(cur_file)
        im_sitk = sitk.ReadImage(cur_file)
        required_spacing = resample_dict[basename.replace('.nii.gz', '')]
        resample_im = resample_image(im_sitk, required_spacing, mask)
        out_file = os.path.join(out_folder, basename)
        sitk.WriteImage(resample_im, out_file)


def get_bbox_from_mask(mask, outside_value=0, margin=(30, 30, 5)):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx - margin[0], maxzidx + margin[0]], [minxidx - margin[1], maxxidx + margin[1]],
            [minyidx - margin[2], maxyidx + margin[2]]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def convert_sdf_to_mask(sdf):
    mask = np.zeros_like(sdf)
    mask[sdf <= 0] = 1
    mask[sdf > 0] = 0
    return mask


def get_union(mask_bbox, sdf_bbox):
    bbox = []
    for m_bbox, s_bbox in zip(mask_bbox, sdf_bbox):
        bbox.append([min(m_bbox[0], s_bbox[0]), max(m_bbox[1], s_bbox[1])])
    return bbox


def get_max_bbox(sdfs_folder, masks_folder, out_folder):
    filelist = glob.glob(os.path.join(sdfs_folder, '*.nii.gz'))

    max_x = 0
    max_y = 0
    max_z = 0

    bbox_dict = defaultdict(list)
    for cur_file in filelist:
        print('reading {}'.format(cur_file))
        basename = os.path.basename(cur_file)
        mask_file = os.path.join(masks_folder, basename)

        sdf = sitk.ReadImage(cur_file)
        org_mask = sitk.ReadImage(mask_file)
        sdf_np = sitk.GetArrayFromImage(sdf)
        org_mask_np = sitk.GetArrayFromImage(org_mask)
        sdf_mask = convert_sdf_to_mask(sdf_np)
        org_mask_np = convert_sdf_to_mask(org_mask_np)
        sdf_bbox = get_bbox_from_mask(sdf_mask, margin=(0, 0, 0))
        mask_bbox = get_bbox_from_mask(org_mask_np, margin=(0, 0, 0))

        union_box = get_union(mask_bbox, sdf_bbox)

        bbox_dict[basename.replace('.nii.gz', '')] = union_box
        size = (union_box[0][1] - union_box[0][0],
                union_box[1][1] - union_box[1][0],
                union_box[2][1] - union_box[2][0])
        max_x = max(max_x, size[0])
        max_y = max(max_y, size[1])
        max_z = max(max_z, size[2])
        print(max_x, max_y, max_z)

    json_obj = json.dumps(bbox_dict, indent=4)
    with open(os.path.join(out_folder, 'bbox.json'), 'w') as f:
        f.write(json_obj)


def get_max_bbox1(sdfs_folder, masks_folder, out_folder):
    filelist = glob.glob(os.path.join(sdfs_folder, '*.nii.gz'))

    max_x = 0
    max_y = 0
    max_z = 0

    bbox_dict = defaultdict(list)
    for cur_file in filelist:
        print('reading {}'.format(cur_file))
        basename = os.path.basename(cur_file)
        mask_file = os.path.join(masks_folder, basename)

        sdf = sitk.ReadImage(cur_file)
        org_mask = sitk.ReadImage(mask_file)
        sdf_np = sitk.GetArrayFromImage(sdf)
        org_mask_np = sitk.GetArrayFromImage(org_mask)
        sdf_mask = convert_sdf_to_mask(sdf_np)
        org_mask_np = convert_sdf_to_mask(org_mask_np)
        sdf_bbox = get_bbox_from_mask(sdf_mask, margin=(0, 0, 0))
        mask_bbox = get_bbox_from_mask(org_mask_np, margin=(0, 0, 0))
        size = (mask_bbox[0][1] - mask_bbox[0][0],
                mask_bbox[1][1] - mask_bbox[1][0],
                mask_bbox[2][1] - mask_bbox[2][0])
        print(size)
        # union_box = get_union(mask_bbox, sdf_bbox)
        #
        # bbox_dict[basename.replace('.nii.gz', '')] = union_box
        # size = (union_box[0][1] - union_box[0][0],
        #         union_box[1][1] - union_box[1][0],
        #         union_box[2][1] - union_box[2][0])
        max_x = max(max_x, size[0])
        max_y = max(max_y, size[1])
        max_z = max(max_z, size[2])
        print(max_x, max_y, max_z)

    json_obj = json.dumps(bbox_dict, indent=4)
    with open(os.path.join(out_folder, 'bbox.json'), 'w') as f:
        f.write(json_obj)


def pad_bbox(bbox_dict_path, out_folder, pad_size):
    with open(bbox_dict_path) as f:
        bbox_dict = json.load(f)

    res_bbox_dict = defaultdict()

    for key in bbox_dict.keys():
        value = bbox_dict[key]
        res_bbox_dict[key] = defaultdict(list)
        res_bbox_dict[key]['bbox'] = value
        pad_amount = (value[0][1] - value[0][0],
                      value[1][1] - value[1][0],
                      value[2][1] - value[2][0])

        pad_amount = (pad_size[0] - pad_amount[0],
                      pad_size[1] - pad_amount[1],
                      pad_size[2] - pad_amount[2])

        pad_amount = [(math.floor(cur_pad / 2), math.ceil(cur_pad / 2)) for cur_pad in pad_amount]

        pad_value = [[value[0][0] - pad_amount[0][0], value[0][1] + pad_amount[0][1]],
                     [value[1][0] - pad_amount[1][0], value[1][1] + pad_amount[1][1]],
                     [value[2][0] - pad_amount[2][0], value[2][1] + pad_amount[2][1]]]
        if pad_value[0][0] < 0:
            import pdb;
            pdb.set_trace()

            pad_value[0][1] += (-1 * pad_value[0][0])
            pad_value[0][0] = 0

        if pad_value[1][0] < 0:
            import pdb;
            pdb.set_trace()

            pad_value[1][1] += (-1 * pad_value[1][0])
            pad_value[1][0] = 0

        if pad_value[2][0] < 0:
            import pdb;
            pdb.set_trace()

            pad_value[2][1] += (-1 * pad_value[2][0])
            pad_value[2][0] = 0
        res_bbox_dict[key]['bbox_pad'] = pad_value

    json_obj = json.dumps(res_bbox_dict)
    with open(os.path.join(out_folder, 'bbox_with_pad.json'), 'w') as f:
        f.write(json_obj)


def crop_sdfs_masks(in_folder, sdfs_folder, bbox_dict_path, out_folder):
    with open(bbox_dict_path) as f:
        bbox_dict = json.load(f)

    sdfs_files = glob.glob(os.path.join(sdfs_folder, '*.nii.gz'))
    os.makedirs(os.path.join(out_folder, 'volume_crop'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'sdf_crop'), exist_ok=True)
    for sdf_file in tqdm(sdfs_files):
        print(sdf_file)
        basename = os.path.basename(sdf_file)
        sdf = sitk.ReadImage(sdf_file)
        im = os.path.join(in_folder, basename)
        im = sitk.ReadImage(im)
        im_np = sitk.GetArrayFromImage(im)
        sdf_np = sitk.GetArrayFromImage(sdf)
        bbox_pad = bbox_dict[basename.replace('.nii.gz', '')]['bbox_pad']

        crop_im = crop_to_bbox(im_np, bbox_pad)
        crop_sdf = crop_to_bbox(sdf_np, bbox_pad)
        print(bbox_pad)
        crop_im = sitk.GetImageFromArray(crop_im)
        crop_sdf = sitk.GetImageFromArray(crop_sdf)

        sitk.WriteImage(crop_im, os.path.join(out_folder, 'volume_crop', basename))
        sitk.WriteImage(crop_sdf, os.path.join(out_folder, 'sdf_crop', basename))


def pad_cts(in_folder, out_folder, out_size=[256, 256, 162], constant_values=-1024):
    """
    Pads all CTs to be the same size (this size should be greater or equal to max CT size in the folder)
    """
    out_size = np.asarray(out_size)
    filelist = glob.glob(os.path.join(in_folder, '*.nii.gz'))
    os.makedirs(out_folder, exist_ok=True)
    # sitk puts z first, so we reverse dimensiosn
    out_size = out_size[::-1]
    pad_dict = defaultdict(list)
    for cur_file in tqdm(filelist):
        im_itk = sitk.ReadImage(cur_file)
        im_np = sitk.GetArrayFromImage(im_itk)
        # get the amount we need to pad for this image
        pad_amount = out_size - im_np.shape
        pad_amount = [(math.floor(cur_pad / 2), math.ceil(cur_pad / 2)) for cur_pad in pad_amount]
        pad_dict[os.path.basename(cur_file).replace('.nii.gz', '')] = pad_amount
        # pad it using numpy
        im_np = np.pad(im_np, pad_amount, mode='constant', constant_values=constant_values)
        new_im = sitk.GetImageFromArray(im_np)
        new_im.SetSpacing(im_itk.GetSpacing())
        new_im.SetDirection(im_itk.GetDirection())
        # now get the pad amount on 'left' side of the image
        pad_adjust = [cur_pad[0] for cur_pad in pad_amount]
        pad_adjust = pad_adjust[::-1]
        # get teh affine matrix from the sitk image
        affine = get_affine_from_itk(im_itk)
        # adjust the origin based on the new pad amount
        origin_adjust = origin_adjust_from_pad(affine, pad_adjust)
        old_origin = list(im_itk.GetOrigin())
        old_origin -= origin_adjust
        new_im.SetOrigin(old_origin)
        # save the file
        basename = os.path.basename(cur_file)
        out_file = os.path.join(out_folder, basename)
        sitk.WriteImage(new_im, out_file)
    json_obj = json.dumps(pad_dict, indent=4)
    with open(os.path.join(out_folder, 'pad_dict.json'), 'w') as f:
        f.write(json_obj)


def unpad_cts(in_folder, out_folder, pad_dict_path):
    """
    remove pad from the padded cts
    """
    filelist = glob.glob(os.path.join(in_folder, '*.nii.gz'))
    os.makedirs(out_folder, exist_ok=True)
    with open(pad_dict_path) as f:
        pad_dict = json.load(f)
    for cur_file in tqdm(filelist):
        basename = os.path.basename(cur_file).replace('.nii.gz', '')
        im_itk = sitk.ReadImage(cur_file)
        im_np = sitk.GetArrayFromImage(im_itk)
        pad_amount = pad_dict[basename]

        new_im = unpad(im_np, pad_amount)
        new_im = sitk.GetImageFromArray(new_im)
        new_im.SetSpacing(im_itk.GetSpacing())
        new_im.SetDirection(im_itk.GetDirection())

        pad_adjust = [cur_pad[0] for cur_pad in pad_amount]
        pad_adjust = pad_adjust[::-1]

        affine = get_affine_from_itk(im_itk)
        origin_adjust = origin_adjust_from_pad(affine, pad_adjust)
        old_origin = list(im_itk.GetOrigin())
        old_origin += origin_adjust
        new_im.SetOrigin(old_origin)

        basename = os.path.basename(cur_file)
        out_file = os.path.join(out_folder, basename)
        sitk.WriteImage(new_im, out_file)


def create_sdf_voxel_samples(in_folder, out_folder):
    filelist = glob.glob(os.path.join(in_folder, '*.nii.gz'))
    os.makedirs(out_folder, exist_ok=True)
    pool = Pool(12)
    # pool = Pool(1)
    pbar = tqdm(total=len(filelist))

    def pbar_update(*args):
        pbar.update()

    # ff = []
    # for f in filelist:
    # basename = os.path.basename(f)
    # if 'RTCT_patient_126' in basename:
    #     ff.append(f)

    for cur_file in filelist:
        basename = os.path.basename(cur_file)
        basename = basename[:-6]
        basename = basename + 'npz'
        out_path = os.path.join(out_folder, basename)
        pool.apply_async(create_sdf_voxel_samples_func, args=(cur_file, out_path), callback=pbar_update)
        # create_sdf_voxel_samples_func(cur_file, out_path)
    pool.close()
    pool.join()
    # pool.close()


def create_sdf_voxel_samples_func(in_file, out_file):
    """
    given an SDF volume, this function will sample sdf values, biased toward the boundary.
    It will save them in an npz file compatible with the SDF shape decoder (4 numbers per sample,
    with the first number being the sdf value and the remaining 3 being the coorindate)
    NOTE: this fucntion assumes sdf volumes are normalized, i.e., created using create_resample_sdfs
    """

    # load the SDF volume
    ni_im = nib.load(in_file)
    sdf = ni_im.get_fdata()
    # find positive points close to the boundary
    pos_points = np.where((sdf >= 0) & (sdf <= .1))
    # unlike meshgrid, np.where does not require swapping the x and y

    # sample positive points and sdf values
    pos_points = np.stack((pos_points[0], pos_points[1], pos_points[2]))
    pos_points = pos_points.transpose()
    pos_sdf = sdf[(sdf >= 0) & (sdf <= .1)]

    # do the same with negative values
    neg_points = np.where((sdf < 0) & (sdf >= -.1))

    neg_points = np.stack((neg_points[0], neg_points[1], neg_points[2]))
    neg_points = neg_points.transpose()
    neg_sdf = sdf[(sdf < 0) & (sdf >= -.1)]
    neg_sdf = neg_sdf.astype(np.float32)
    pos_sdf = pos_sdf.astype(np.float32)
    neg_points = neg_points.astype(np.float32)
    pos_points = pos_points.astype(np.float32)
    # now also sample some positive and negative points uniformly through the volume
    # similar to DeepSDF, we only sample a fraction of these points compared to the boundary points
    uniform_len_pos = int(0.1 * len(pos_sdf))
    uniform_len_neg = int(0.1 * len(neg_sdf))

    pos_uniform = np.where((sdf >= 0))
    pos_uniform_sdf = sdf[(sdf >= 0)]

    pos_uniform = np.stack((pos_uniform[0], pos_uniform[1], pos_uniform[2], pos_uniform_sdf))
    pos_uniform = pos_uniform.transpose()
    np.random.shuffle(pos_uniform)
    pos_uniform = pos_uniform[:uniform_len_pos, :]
    pos_points = np.concatenate((pos_points, pos_uniform[:, :3]))
    pos_sdf = np.concatenate((pos_sdf, pos_uniform[:, 3]))

    neg_uniform = np.where((sdf < 0))
    neg_uniform_sdf = sdf[(sdf < 0)]

    neg_uniform = np.stack((neg_uniform[0], neg_uniform[1], neg_uniform[2], neg_uniform_sdf))
    neg_uniform = neg_uniform.transpose()
    np.random.shuffle(neg_uniform)
    neg_uniform = neg_uniform[:uniform_len_neg, :]
    neg_points = np.concatenate((neg_points, neg_uniform[:, :3]))
    neg_sdf = np.concatenate((neg_sdf, neg_uniform[:, 3]))
    if neg_sdf.shape[0] == 0:
        print(in_file)
    # save the pos and neg points into an npz
    np.savez(out_file, pos_points=pos_points, neg_points=neg_points, pos_sdf=pos_sdf,
             neg_sdf=neg_sdf)


def clean_path_end(path):
    while path[-1] == '/':
        path = path[:-1]

    return path


def create_sample_jsonv2(ct_folder, sdf_folder, embed_json, json_out_path, gt_scale_json_path=None):
    """
    Creates a training json from a set of SDF npz files and corresponding resampled and padded
    CTs. It will also add the GT translations, scales, and rotations calculated using CPD
    Assumes ct_folder and sdf_folder describe two subfolders in the same folder
    """

    with open(embed_json, 'r') as f:
        embed_json = json.load(f)

    if gt_scale_json_path:
        # load the gt json
        with open(gt_scale_json_path, 'r') as f:
            gt_scale_json = json.load(f)

        gt_scale_json_dict = {cur_dict['im']: cur_dict for cur_dict in gt_scale_json}
    else:
        gt_scale_json_dict = None



    json_list = []

    for embed_dict in embed_json:

        basename = embed_dict['path'][:-8]
        ct_subfolder = os.path.basename(ct_folder)
        sdf_subfolder = os.path.basename(sdf_folder)

        ct_path = os.path.join(ct_subfolder, basename + '.nii.gz')

        sdf_path = os.path.join(sdf_subfolder, basename + '.npz')
        cur_dict = {
            'path': sdf_path,
            'im': ct_path
        }

        if gt_scale_json_dict: 
            gt_dict = gt_scale_json_dict[basename + '.nii.gz']

            for k, v in gt_dict.items():
                if k != 'im':
                    cur_dict[k] = v
        json_list.append(cur_dict)

    with open(json_out_path, 'w') as f:
        json.dump(json_list, f)

    # pos_points = points[sdf >= 0]
    # neg_points = points[sdf < 0]
    # pos_sdf = sdf[sdf >= 0]
    # neg_sdf = sdf[sdf < 0]
    # np.savez(out_path, pos_points=pos_points, neg_points=neg_points, pos_sdf=pos_sdf,
    # neg_sdf=neg_sdf)


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def crop_data_based_on_trans(in_folder, out_folder, trans=[255, 217, 275], pad=[60, 60, 60]):
    filelist = glob.glob(os.path.join(in_folder, '*.nii.gz'))

    crop_bbox = [[trans[0] - pad[0], trans[0] + pad[0]],
                 [trans[1] - pad[1], trans[1] + pad[1]],
                 [trans[2] - pad[2], trans[2] + pad[2]]]

    for cur_file in tqdm(filelist, total=len(filelist)):
        basename = os.path.basename(cur_file)
        im_itk = nib.load(cur_file)
        im_np = im_itk.get_fdata()
        crop_im = crop_to_bbox(im_np, crop_bbox)
        mask_new = im_np.copy()
        # mask_new[crop_bbox[0][0]:crop_bbox[0][1],
        # crop_bbox[1][0]:crop_bbox[1][1],
        # crop_bbox[2][0]:crop_bbox[2][1]] = 0
        # if np.sum(mask_new) > 0:
        #     import pdb;
        #     pdb.set_trace()
        #
        crop_im = nib.Nifti1Image(crop_im, np.eye(4))
        nib.save(crop_im, os.path.join(out_folder, basename))
        # crop_itk = sitk.GetImageFromArray(crop_im)
        # sitk.WriteImage(crop_itk, os.path.join(out_folder, basename))


def test_augmentation(sdf_im, sdf_ni):
    from monai.transforms import LoadNiftid
    data_dict = {'im': sdf_im}
    loader = LoadNiftid(keys=('im'))
    data_dict = loader(data_dict)
    rand_affine = CustomRandomAffine(
        keys=('im'),
        mode=("bilinear"),
        prob=1.0,
        translate_range=(20, 20, 20),
        rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
        scale_range=(0.1, 0.1, 0.1),
        padding_mode="border"
    )

    sdf_points = np.load(sdf_ni)
    test_point = sdf_points['pos_points'][-1]
    orig_im = data_dict['im']

    orig_im = np.expand_dims(orig_im, 0)
    data_dict['im'] = orig_im
    # ni_im = nib.Nifti1Image(orig_im, affine=np.eye(4))
    # nib.save(ni_im, '/home/adam/Downloads/orig_im.nii.gz')

    print(test_point)
    test_point_copy = np.copy(test_point).astype(np.int)
    test_point -= (np.asarray(orig_im.shape[1:]) - 1) / 2
    # temp = test_point[0]

    # test_point[0] = test_point[1]
    # test_point[1] = temp
    test_point = np.concatenate((test_point, [1]))

    test_sdf = sdf_points['pos_sdf'][-1]
    new_data_dict = rand_affine(data_dict)
    new_sdf = new_data_dict['im']
    new_sdf = new_sdf[0, :]
    # ni_im = nib.Nifti1Image(new_sdf.cpu().detach().numpy(), affine=np.eye(4))
    # nib.save(ni_im, '/home/adam/Downloads/new_im.nii.gz')
    print("%f" % new_sdf[test_point_copy[0], test_point_copy[1], test_point_copy[2]])

    affine_matrix = new_data_dict['affine_matrix']

    # new_point = affine_matrix @ test_point 

    new_point = np.linalg.solve(affine_matrix, test_point)
    new_point = new_point[:3]
    # temp = new_point[0]
    # new_point[0] = new_point[1]
    # new_point[1] = temp

    test_point = test_point[:3].astype(np.float32)
    test_point += (np.asarray(orig_im.shape[1:]) - 1) / 2
    new_point += (np.asarray(orig_im.shape[1:]) - 1) / 2
    print(new_point)
    new_point = np.round(new_point).astype(np.int)
    print("%f" % new_sdf[new_point[0], new_point[1], new_point[2]])
    import ipdb;
    ipdb.set_trace()
