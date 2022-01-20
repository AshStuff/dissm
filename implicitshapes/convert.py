import mcubes
import math
from tqdm import tqdm
from multiprocessing import Pool
import trimesh
from mesh_to_sdf import sample_sdf_near_surface, mesh_to_voxels, get_surface_point_cloud, mesh_to_sdf
import shutil
import json
import os
import glob
import nibabel as ni
from skimage.transform import resize
from skimage.measure import marching_cubes_lewiner
import numpy as np
import SimpleITK as sitk
from copy import deepcopy
import pymesh
from itk_utils import get_affine_from_itk, origin_adjust_from_pad
from mesh_utils import scale_mesh, rigid_align_meshes, get_scale





def convert_binary2mesh(path, out_path):
    """
    Converts binary mask to mesh using mcubes library
    """
    im_sitk = sitk.ReadImage(path)

    origin = im_sitk.GetOrigin()
    # pull affine matrix from sitk image
    affine = get_affine_from_itk(im_sitk)
    im = sitk.GetArrayFromImage(im_sitk)
    # pad image to make sure the mask is never touching the boundary
    pad = 5
    im = np.pad(im, pad)
    origin_adjust = origin_adjust_from_pad(affine, [pad, pad, pad])
    origin = list(origin) - origin_adjust

    affine[:3,3] = origin

    # smooth binary mask using simple Gaussian smoothin
    smoothed = mcubes.smooth(im, method='gaussian')
    # perform marching cubes
    vertices, triangles, _, _ = marching_cubes_lewiner(smoothed, 0, allow_degenerate=False)

    # because sitk images have z direction first, we must swap x and z coordinates
    vertices = vertices[:, ::-1]


    # convert pixel coordinates to world coordinates using the image's affine
    vertices = np.transpose(vertices)
    
    ones = np.ones((1, vertices.shape[1]))
    vertices = np.concatenate([vertices, ones])
    vertices = np.matmul(affine, vertices)
    vertices = vertices[:3,:]
    vertices = np.transpose(vertices)

    # save the resulting set of vertices and triangles
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.export(out_path)


def remove_extension(filename):
    while '.' in filename:
        filename = os.path.splitext(filename)[0]

    return filename

def convert_all_masks(in_folder, out_folder):
    """
    Used to convert all masks in a folder to a set of meshes
    """

    query_path = os.path.join(in_folder, '*.nii.gz')
    paths = glob.glob(query_path)

    # find all .nii.gz files and convert them to mesh .obj files
    for cur_path in paths:
        print(cur_path)
        cur_filename = os.path.basename(cur_path)
        cur_filename = remove_extension(cur_filename)
    
        cur_filename = cur_filename + '.obj'
        out_file = os.path.join(out_folder, cur_filename)
        convert_binary2mesh(cur_path, out_file)

def simplify_all_meshes(in_folder, out_folder, bin_path, factor):
    """
    calls the Fast-Quadric-Mesh Simplication routine to reduce the complexity of all meshes in a folder
    """

    query_path = os.path.join(in_folder, '*.obj')
    paths = glob.glob(query_path)

    os.makedirs(out_folder, exist_ok=True)

    for cur_path in paths:
        print(cur_path)
        
        cur_filename = os.path.basename(cur_path)
        out_file = os.path.join(out_folder, cur_filename)
        os.system(bin_path + ' ' + cur_path + ' ' + out_file + ' ' + str(factor))


def register_meshes(in_folder, out_folder):
    """
    Rigidly registers all meshes to an anchor mesh using coherent point drift. Best performed
    on very simplified meshes, otherwise it will take a long time
    """

    # find all mesh files
    query_path = os.path.join(in_folder, '*.obj')
    paths = glob.glob(query_path)
    paths.sort()
    os.makedirs(out_folder, exist_ok=True)
    # make the first mesh the anchor mesh, and simply copy it to out_folder
    anchor_mesh_path = paths[0]
    anchor_name = os.path.basename(anchor_mesh_path)
    shutil.copyfile(anchor_mesh_path, os.path.join(out_folder, anchor_name))
    print('Anchor', anchor_mesh_path)
    anchor_mesh = trimesh.load(anchor_mesh_path)

    # for every other mesh
    for cur_path in paths[1:]:
        print(cur_path)
        cur_filename = os.path.basename(cur_path) 
        out_file = os.path.join(out_folder, cur_filename)
        moving_mesh = trimesh.load(cur_path)
        # perform rigid registration using CPD based on the two sets of vertices
        new_mesh, cur_dict = rigid_align_meshes(anchor_mesh, moving_mesh)
        cur_dict['anchor_name'] = anchor_name
        new_mesh.export(out_file)
        # save the sacle, translation, and rotation and anchor mesh name to a json file
        with open(out_file + '.json', 'w') as f:
            json.dump(cur_dict, f)


def align_large_meshes(in_folder, aligned_mesh_folder, out_folder):
    """
    If an alignment was performed using a simplified set of meshes, from "register_meshes",
    then this function can apply the registration transform to a more complex version of teh same
    mesh
    """


    # get all meshes in in_folder and get the first mesh as anchor
    query_path = os.path.join(in_folder, '*.obj')
    paths = glob.glob(query_path)
    paths.sort()
    os.makedirs(out_folder, exist_ok=True)
    anchor_mesh_path = paths[0]
    anchor_name = os.path.basename(anchor_mesh_path)
    shutil.copyfile(anchor_mesh_path, os.path.join(out_folder, anchor_name))
    print('Anchor', anchor_mesh_path)

    for cur_path in paths[1:]:
        print(cur_path)
        cur_filename = os.path.basename(cur_path) 
        out_file = os.path.join(out_folder, cur_filename)
        # load up the transform parameters from the json file
        json_file = os.path.join(aligned_mesh_folder, cur_filename + '.json')
        with open(json_file, 'r') as f:
            params = json.load(f)
        scale = params['s']
        R = np.array(params['R'])
        t = np.array(params['t'])
        # apply the transform to the vertices
        moving_mesh = trimesh.load(cur_path)
        moving_vertices = moving_mesh.vertices
        moving_vertices = scale * np.dot(moving_vertices, R) + t
        # save the aligned mesh
        new_mesh = trimesh.Trimesh(moving_vertices, moving_mesh.faces)
        new_mesh.export(out_file)


# best configs: number_of_points=1000000, uniform_proportion=.2, jitter=.1

def sample_sdf(in_folder, out_folder, number_of_points=500000, uniform_proportion=0.5, jitter=0.0025):
    """
    Given a set of meshes, this function will sample a set of points from the sdf, biased heavily toward
    the boundary. Each file will be saved as a npz file
    """

    # default is 500000
    query_path = os.path.join(in_folder, '*.obj')
    paths = glob.glob(query_path)
    paths.sort()
    os.makedirs(out_folder, exist_ok=True)
    # run the sampling asynchronously in parallel
    pool = Pool(12)


    # get a rough scale for use later in estimation
    first_path = paths[0]
    first_mesh = trimesh.load(first_path)
    rough_scale = get_scale(first_mesh)
    print('Rough scale is ', rough_scale) 
    txt_path = os.path.join(out_folder, 'scale.txt')

    with open(txt_path, 'w') as f:
        f.write(str(rough_scale))



    for cur_path in paths:
        
        cur_filename = os.path.basename(cur_path) 
        out_file = os.path.join(out_folder, cur_filename + '.npz')
        pool.apply_async(sample_one_sdf_file, args=(cur_path, out_file, number_of_points, uniform_proportion, jitter))

    pool.close()
    pool.join()


def sample_one_sdf_file(cur_path, out_path, number_of_points=500000, uniform_proportion=0.5, jitter=0.0025):
    """
    A synchronous function call for sdf sampling
    """

    print(cur_path)
    mesh = trimesh.load(cur_path)
    points, sdf = modified_sample_sdf_near_surface(mesh, number_of_points=number_of_points, uniform_proportion=uniform_proportion, surface_point_method='sample', jitter=jitter)

    # get the positive and negative points, useful for trainign the DeepSDF model
    pos_points = points[sdf >= 0]
    neg_points = points[sdf < 0]
    pos_sdf = sdf[sdf >= 0]
    neg_sdf = sdf[sdf < 0]
    np.savez(out_path, pos_points=pos_points, neg_points=neg_points, pos_sdf=pos_sdf,
            neg_sdf=neg_sdf)

def voxelize_one_mesh_file(cur_path, out_path):

    mesh = trimesh.load(cur_path)
    voxels = mesh_to_voxels(mesh, 200, pad=True)

    np_im = ni.Nifti1Image(voxels, np.eye(4))
    ni.save(np_im, out_path)


def modified_sample_sdf_near_surface(mesh, number_of_points = 500000, surface_point_method='scan', sign_method='normal', scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, min_size=0, uniform_proportion=.5, jitter=0.0025):
    """
    Modification of the mesh_to_sdf library, to allow for more configurable sampling, e.g., more toward the uniform samplinga and a higher
    jitter
    """
    mesh = scale_mesh(mesh)
    
    if surface_point_method == 'sample' and sign_method == 'depth':
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = 'normal'

    surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method, 1, scan_count, scan_resolution, sample_point_count, calculate_normals=sign_method=='normal')

    return surface_point_cloud.sample_sdf_near_surface(number_of_points, surface_point_method=='scan', sign_method, normal_sample_count, min_size, uniform_proportion=uniform_proportion, jitter=jitter)



def render_sampled_pts(mesh_file):
    """
    Can be used to render the sampled points to make sure they make sense
    """
    import pyrender
    mesh = trimesh.load(mesh_file)


    points, sdf = modified_sample_sdf_near_surface(mesh, surface_point_method='sample', number_of_points=100000)

    colors = np.zeros(points.shape)
    colors[sdf < 0, 2] = 1
    colors[sdf > 0, 0] = 1
    cloud = pyrender.Mesh.from_points(points, colors=colors)
    scene = pyrender.Scene()
    scene.add(cloud)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

def create_sample_json(in_folder):
    """
    Simple function to create a json file of the npz files, for later use in training DeepSDF model
    """

    query_path = os.path.join(in_folder, '*.npz')
    paths = glob.glob(query_path)
    paths.sort()

    json_list = []
    for cur_path in paths:
        print(cur_path)
        cur_filename = os.path.basename(cur_path)
        json_list.append({'path': cur_filename})

    save_file = os.path.join(in_folder, 'json_list.json')
    with open(save_file, 'w') as f:
        json.dump(json_list, f)


def sample_sdf_at_point(points, mesh_file):
    mesh = trimesh.load(mesh_file)
    mesh = scale_mesh(mesh)
    points = np.asarray(points)
    if points.ndim == 1:
        points = np.expand_dims(points, 0)
    
    sdf = mesh_to_sdf(mesh, points, surface_point_method='sample')
    return sdf



