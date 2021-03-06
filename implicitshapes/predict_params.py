import json
import os
import tempfile

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch
from infer_shape import infer_mean_aligned_sdf
from networks.encoder import WrapperResidualEncoder

from dataset import Affine_w_Mtx
from itk_utils import create_sitk_im


def predict_params(model_path, volume_file):
    model = WrapperResidualEncoder(num_outputs=6, in_channels=1, f_maps=16)
    loaded = torch.load(model_path)

    model_weights = loaded['component.model']

    model_dict = {k[15:]: v for k, v in model_weights.items() if 'encoder' in k}

    model.load_state_dict(model_dict)
    model = model.cuda()
    model.eval()

    im = nib.load(volume_file)

    im_np = im.get_fdata()
    im_np = np.clip(im_np, -160, 240)

    im_np = np.expand_dims(im_np, 0)
    im_np = np.expand_dims(im_np, 0)
    im_np = torch.from_numpy(im_np).float().cuda()
    data_dict = {'im': im_np}
    with torch.no_grad():
        data_dict = model(data_dict)
    return data_dict


def predict_sdf(model_encoder_path, model_decoder_path, decoder_config, volume_file, save_path, do_scale=True,
                sdf_gt_json_file=None):
    data_dict = predict_params(model_encoder_path, volume_file)

    trans = data_dict['theta'][0, :3].cpu().numpy()
    print(trans)

    if sdf_gt_json_file:
        with open(sdf_gt_json_file, 'r') as f:
            json_list = json.load(f)
        volume_basename = os.path.basename(volume_file)
        for cur_dict in json_list:
            cur_file = os.path.basename(cur_dict['im'])
            if cur_file == volume_basename:
                break
        init_trans = cur_dict['t']
    else:
        init_trans = [155, 140, 80]

    if do_scale:
        trans += init_trans
        scale = data_dict['theta'][0, 3:6].cpu().numpy() + 1


    else:
        trans += init_trans
        # default scale
        scale = np.ones(3)

    print(scale)

    scale[0] *= -1
    scale[1] *= -1
    scale[2] *= 5 / 2
    scale = scale / 60

    infer_mean_aligned_sdf(model_decoder_path, decoder_config, save_path, volume_file, trans, scale)


def calculate_error(model_decoder_path, decoder_config, sdf_gt_file, trans, scale):
    fd, path = tempfile.mkstemp(suffix='.nii.gz')
    try:
        infer_mean_aligned_sdf(model_decoder_path, decoder_config, path, sdf_gt_file, trans, scale, batch_size=2000000)
        gt_sdf = sitk.ReadImage(sdf_gt_file)
        gt_np = sitk.GetArrayFromImage(gt_sdf)
        infer_sdf = sitk.ReadImage(path)
        infer_np = sitk.GetArrayFromImage(infer_sdf)
        infer_np[infer_np < 0] = 0
        infer_np[infer_np > 0] = 1

        gt_np[gt_np < 0] = 0
        gt_np[gt_np > 0] = 1
        diff = gt_np - infer_np
        diff = np.abs(diff)

        diff_im = create_sitk_im(diff, gt_sdf)
        sitk.WriteImage(diff_im, '/home/adam/Downloads/diff_im.nii.gz')

        diff = np.mean(np.abs(diff))
        print(np.sum(diff))
    finally:
        os.remove(path)


def load_model(model_path):

    loaded = torch.load(model_path)

    model_weights = loaded['state_dict']

    model_dict = {k[15:]: v for k, v in model_weights.items() if 'encoder' in k}
    num_outputs = model_dict['_model_classifier.weight'].shape[0]

    model = WrapperResidualEncoder(num_outputs=num_outputs, in_channels=2, f_maps=32)
    model.eval()


    model.load_state_dict(model_dict)
    model = model.cuda()
    model.eval()
    return model

def predict_params_channel(model, volume_file, mean_sdf):

    im = nib.load(volume_file)

    im_np = im.get_fdata()
    im_np = np.clip(im_np, -160, 240)
    im_np -= 40
    im_np /= 200

    im_np = np.stack((im_np, mean_sdf))

    im_np = np.expand_dims(im_np, 0)
    im_np = torch.from_numpy(im_np).float().cuda()
    data_dict = {'im': im_np}
    with torch.no_grad():
        data_dict = model(data_dict)
    return data_dict


def predict_sdf_steps(model_encoder_path, volume_file, mean_sdf_file,
                      init_affine=None, steps=5, do_scale=True):
    # gt_trans = np.asarray([141.2, 133.0, 81.02])

    # gt_trans = gt_trans - Asdf2px[:3,3]
    # init_affine = np.eye(4)
    # init_affine[:3,3] = gt_trans

    model = load_model(model_encoder_path)
    if init_affine is None:
        init_affine = np.eye(4)

    mean_sdf = nib.load(mean_sdf_file)
    mean_sdf = mean_sdf.get_fdata()

    # mean_sdf = np.zeros_like(mean_sdf)
    affine_mtx = init_affine
    for cur_step in range(steps):
        im_affine_mtx = np.linalg.inv(affine_mtx)
        affine_trans = Affine_w_Mtx(affine_mtx=torch.tensor(im_affine_mtx), padding_mode='border')
        new_mean_np = affine_trans(np.expand_dims(mean_sdf, 0))
        new_mean_np = new_mean_np[0, :]

        data_dict = predict_params_channel(model, volume_file, new_mean_np)
        theta = data_dict['theta'][0, :].detach().cpu().numpy()
        cur_refine_aff = np.eye(4)

        if do_scale:
            cur_refine_aff[0, :] *= (theta[3] + 1)
            cur_refine_aff[1, :] *= (theta[4] + 1)
            cur_refine_aff[2, :] *= (theta[5] + 1)
        cur_refine_aff[0, 3] = -theta[0]
        cur_refine_aff[1, 3] = -theta[1]
        cur_refine_aff[2, 3] = -theta[2]
        # we describe an affine goign from canonical to image space
        affine_mtx = np.linalg.solve(cur_refine_aff, affine_mtx)
        print('cur_refine_aff', np.linalg.inv(cur_refine_aff))
        print('cur_aff', affine_mtx)


    # get transform to center coordinates
    centering = -(np.asarray(mean_sdf.shape) - 1) / 2

    # create affine matrix to center pixel coordiantes, b/c world coordiante transform works on uncentered pixel coordinates
    center_affine = np.eye(4)
    center_affine[:3, 3] = centering

    volume = nib.load(volume_file)
    im_coord_aff = volume.affine
    # make the wolrd coordinate transform operate on centered pixels to be compatible with how affines are used in the encoder
    im_coord_aff = im_coord_aff @ np.linalg.inv(center_affine)
   
    # create a final affine matrix from the pose encoder in world coordinates
    world_affine_matrix = im_coord_aff @ affine_mtx

    print('Final Affine', affine_mtx)
    print('Final world affine', world_affine_matrix)
    return affine_mtx, world_affine_matrix

def create_post_training_affines(model_encoder_path, im_root, json_list_path, mean_sdf_file, output_json_path, steps=10, do_scale=False, do_rotate=False, do_pca=False):


    if not do_scale and not do_rotate and not do_pca:
        key = 'post_translate'
        world_key = 'post_translate_world'

    output_json_list = []
    with open(json_list_path, 'r') as f:
        json_list = json.load(f)

    for cur_dict in json_list:
        im_path = os.path.join(im_root, cur_dict['im'])
        final_aff, final_aff_world = predict_sdf_steps(model_encoder_path, im_path, mean_sdf_file, init_affine=None, steps=steps, do_scale=do_scale)
        cur_dict[key] = final_aff.tolist()
        cur_dict[world_key] = final_aff_world.tolist()

        output_json_list.append(cur_dict)

    with open(output_json_path, 'w') as f:
        json.dump(output_json_list, f)

def render_sdf(model_encoder_path_trans, model_decoder_path, decoder_config, volume_file, mean_sdf_file, save_path, scale,
        init_affine=None, steps=5):

    ref_im = sitk.ReadImage(volume_file)
    sdf_size = ref_im.GetSize()
    ref_spacing = ref_im.GetSpacing()

    trans_affine_mtx, _ = predict_sdf_steps(model_encoder_path_trans, volume_file,
                                         mean_sdf_file, init_affine, steps, do_scale=False)
    infer_affine_mtx = trans_affine_mtx

    # if model_encoder_path_scale:
        # scale_affine_mtx = predict_sdf_steps(model_encoder_path_scale, model_decoder_path, decoder_config, volume_file,
                                             # mean_sdf_file, trans_affine_mtx, steps=1, do_scale=True, Apx2sdf=Apx2sdf)
        # infer_affine_mtx = scale_affine_mtx

    infer_mean_aligned_sdf(model_decoder_path, decoder_config, save_path, volume_file, scale, infer_affine_mtx )

def create_trans_json(source_json, model_encoder_path, model_decoder_path, decoder_config, volume_root, mean_sdf_file,
                      save_path, init_affine=None, steps=5, Apx2sdf=None):
    if Apx2sdf is None:
        Apx2sdf = np.eye(4)
        Apx2sdf[0, 3] = -127.5
        Apx2sdf[1, 3] = -127.5
        Apx2sdf[2, 3] = -80.5
        Apx2sdf[0, :] *= -1 / 63.5
        Apx2sdf[1, :] *= -1 / 63.5
        Apx2sdf[2, :] *= 1 / 63.5 * (5 / 2)
    Asdf2px = np.linalg.inv(Apx2sdf)

    with open(source_json, 'r') as f:
        json_list = json.load(f)

    for cur_dict in json_list:
        im_file = os.path.join(volume_root, cur_dict['im'])
        trans_affine_mtx = predict_sdf_steps(model_encoder_path, model_decoder_path, decoder_config, im_file,
                                             mean_sdf_file, init_affine, steps, do_scale=False, Apx2sdf=Apx2sdf)
        predict_trans = Asdf2px[:3, 3] + trans_affine_mtx[:3, 3]
        cur_dict['predict_trans'] = predict_trans.tolist()

    with open(save_path, 'w') as f:
        json.dump(json_list, f)


def add_predict_trans(source_json, dest_json, save_file):
    with open(source_json, 'r') as f:
        s_json = json.load(f)

    with open(dest_json, 'r') as f:
        d_json = json.load(f)

    for s_dict, d_dict in zip(s_json, d_json):
        d_dict['predict_trans'] = s_dict['predict_trans']

    with open(save_file, 'w') as f:
        json.dump(d_json, f)


def measure_sdf_accuracy(infer_sdf_file, gt_sdf_file):
    from scipy.special import expit
    i_sdf = nib.load(infer_sdf_file)
    i_sdf = i_sdf.get_fdata()
    g_sdf = nib.load(gt_sdf_file)
    g_sdf = g_sdf.get_fdata()

    i_sdf = expit(i_sdf)

    g_sdf[g_sdf < 0] = 0
    g_sdf[g_sdf > 0] = 1

    ce = np.log(i_sdf + 0.0001) * g_sdf + np.log(1 - i_sdf + 0.0001) * (1 - g_sdf)
    ce = -1 * np.mean(ce)

    print('CE', ce)
