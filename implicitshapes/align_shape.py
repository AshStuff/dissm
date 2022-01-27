import glob
import json
import os

import SimpleITK as sitk
import numpy as np
import tqdm
import trimesh
from mesh_utils import scale_mesh, rigid_align_meshes

from itk_utils import get_affine_from_itk


def align_translate_scale(anchor_mesh, moving_mesh):
    a_mesh = trimesh.load(anchor_mesh)
    m_mesh = trimesh.load(moving_mesh)

    vertices = a_mesh.vertices - a_mesh.bounding_box.centroid
    a_scale = np.max(np.abs(vertices), axis=0)
    # a_scale = np.max(np.linalg.norm(vertices, axis=1))

    vertices = m_mesh.vertices - m_mesh.bounding_box.centroid
    m_scale = np.max(np.abs(vertices), axis=0)
    # m_scale = np.max(np.linalg.norm(vertices, axis=1))

    m_scale = a_scale / m_scale

    rel_scale = calculate_relative_scale(a_mesh, m_mesh)
    vertices[:, 0] *= rel_scale[0]
    vertices[:, 1] *= rel_scale[1]
    vertices[:, 2] *= rel_scale[2]

    vertices += a_mesh.bounding_box.centroid

    new_mesh = trimesh.Trimesh(vertices, m_mesh.faces)
    # new_mesh.export(save_loc)
    return new_mesh


def calculate_relative_scale(a_mesh, m_mesh):
    vertices = a_mesh.vertices - a_mesh.bounding_box.centroid
    a_scale = np.max(np.abs(vertices), axis=0)
    # a_scale = np.max(np.linalg.norm(vertices, axis=1))

    vertices = m_mesh.vertices - m_mesh.bounding_box.centroid
    m_scale = np.max(np.abs(vertices), axis=0)
    # m_scale = np.max(np.linalg.norm(vertices, axis=1))

    rel_scale = a_scale / m_scale

    return rel_scale


def project_to_pixels(mesh, affine):
    vertices = mesh.vertices
    vertices = np.transpose(vertices)

    ones = np.ones((1, vertices.shape[1]))
    vertices = np.concatenate([vertices, ones])
    vertices = np.linalg.solve(affine, vertices)
    vertices = vertices[:3, :]
    vertices = np.transpose(vertices)

    return trimesh.Trimesh(vertices, mesh.faces)


def create_scale_translation_json(in_folder, im_folder, mean_mesh_file, scale_factor):
    """
    Computes the scale and translation needed to align a mean mesh to each of the images
    """

    filelist = glob.glob(os.path.join(in_folder, "*.obj"))
    filelist.sort()
    json_list = []
    for cur_file in tqdm.tqdm(filelist):
        basename = os.path.basename(cur_file)
        im_basename = basename[:-4] + '.nii.gz'
        im_path = os.path.join(im_folder, im_basename)
        # get the transformation parameters
        _, trans_dict = align_meshes_pixel_space(cur_file, mean_mesh_file, im_path, scale_factor)

        cur_dict = {
            'im': im_basename,
            't': trans_dict['t'],
            'R': trans_dict['R'],
            's': trans_dict['s']
        }
        json_list.append(cur_dict)

    with open(os.path.join(im_folder, 't_scale_list.json'), 'w') as f:
        json.dump(json_list, f)


def move_to_pixel_space(anchor_path, anchor_image):
    anchor_mesh = trimesh.load(anchor_path)

    im_sitk = sitk.ReadImage(anchor_image)

    spacing = im_sitk.GetSpacing()
    direction = im_sitk.GetDirection()
    origin = im_sitk.GetOrigin()
    affine = np.eye(4)
    affine[0, :3] = np.asarray(direction[:3]) * spacing[0]
    affine[1, :3] = np.asarray(direction[3:6]) * spacing[1]
    affine[2, :3] = np.asarray(direction[6:9]) * spacing[2]
    affine[0, 3] = origin[0]
    affine[1, 3] = origin[1]
    affine[2, 3] = origin[2]

    a_mesh = project_to_pixels(anchor_mesh, affine)

    return a_mesh


def align_meshes_pixel_space(anchor_path, moving_path, ref_im_path, scale_factor):

    # get reference image
    ref_im = sitk.ReadImage(ref_im_path)
    # load anchor mesh in world coordinates
    a_mesh = trimesh.load(anchor_path)
    # load moving mesh
    m_mesh = trimesh.load(moving_path)
    affine = get_affine_from_itk(ref_im)

    # we need to scale the mean mesh to project it to pixel coordinates since it's in canonical [-1, 1] coordinates
    # it will be set to be located in the center of the image in pixel coordinates
    im_size = ref_im.GetSize()
    ref_spacing = ref_im.GetSpacing()

    Apx2sdf = np.eye(4)
    Apx2sdf[0, 3] = -(im_size[0] - 1)/2
    Apx2sdf[1, 3] = -(im_size[1] - 1)/2
    Apx2sdf[2, 3] = -(im_size[2] - 1)/2

    Apx2sdf[0, :] *= -ref_spacing[0] / scale_factor
    Apx2sdf[1, :] *= -ref_spacing[1] / scale_factor
    Apx2sdf[2, :] *= ref_spacing[2] / scale_factor


    # we project both to pixel space 
    a_mesh = project_to_pixels(a_mesh, affine)
    m_mesh = project_to_pixels(m_mesh, Apx2sdf)

    # because cpd does scale, roation, then translation, the t value will not match the pose estimation value
    a_center = a_mesh.bounding_box.centroid
    t_override = [cur_a - (cur_size - 1)/2 for cur_a, cur_size in zip(a_center, im_size)]

    # now conduct cpd alignment
    new_mesh, trans_dict = rigid_align_meshes(a_mesh, m_mesh)
    trans_dict['t'] = t_override
    return new_mesh, trans_dict


def calculate_mean_values(json_file):
    with open(json_file, 'r') as f:
        json_list = json.load(f)

    sum_t = np.zeros(3)
    sum_s = 0
    for cur_dict in json_list:
        cur_t = cur_dict['t']
        cur_s = cur_dict['s']

        sum_t += np.asarray(cur_t)
        sum_s += cur_s

    print(sum_t / len(json_list))
    print(sum_s / len(json_list))
