import os, glob
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

# from encoder import MultiModelEncoder
from dataset import Affine_w_Mtx
from encoder import WrapperResidualEncoder
from implicitshapes.infer_shape import infer_mean_aligned_sdf


def predict_params_channel(model_path, volume_file, mean_sdf, do_rotate, do_pca, latent_vecs=None):
    if do_pca:
        model = WrapperResidualEncoder_pca(num_outputs=6, in_channels=2, f_maps=32, do_pca=do_pca)
    else:
        model = WrapperResidualEncoder(num_outputs=6, in_channels=2, f_maps=32, do_pca=do_pca)
    if do_rotate:
        rotate_classifier = nn.Linear(model._model.output_size, 6)
        rotate_classifier.eval()
    if do_pca:
        latent_reduce = nn.Linear(256, 128).cuda()
        pca_classifier = nn.Linear(768, 28).cuda()
        pca_classifier.eval()
        latent_reduce.eval()
    model.eval()

    loaded = torch.load(model_path)
    # import pdb;pdb.set_trace()
    model_weights = loaded['state_dict']
    model_dict = {k[15:]: v for k, v in model_weights.items() if 'encoder' in k}

    model.load_state_dict(model_dict)
    model = model.cuda()
    model.eval()
    if do_rotate:
        model_dict = {k.split('.')[-1]: v for k, v in model_weights.items() if '_rotate_classifier' in k}
        rotate_classifier.load_state_dict(model_dict)
        rotate_classifier = rotate_classifier.cuda()
        rotate_classifier.eval()
    if do_pca:
        model_dict = {'.'.join(k.split('.')[-1:]): v for k, v in model_weights.items() if 'pca_classifier' in k}

        pca_classifier.load_state_dict(model_dict)
        pca_classifier.eval()
        # model_dict = {'.'.join(k.split('.')[-1:]): v for k, v in model_weights.items() if 'latent_reduce' in k}
        # latent_reduce.load_state_dict(model_dict)
        # latent_reduce.eval()

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
        if do_pca:
            if latent_vecs is None:
                latent_vecs = torch.zeros((im_np.shape[0], 256)).cuda()
            data_dict['latent_vecs'] = latent_vecs

        data_dict = model(data_dict)
        if do_rotate:
            data_dict['rotate'] = rotate_classifier(model.features)
        if do_pca:
            data_dict['pca_coeff'] = pca_classifier(model.local_features)
    return data_dict


def inverse_transform(scalar, pca_components):
    batch_size = scalar.shape[0]
    pca_components = pca_components.repeat(batch_size, 1, 1)
    return torch.matmul(scalar.unsqueeze(1), pca_components).squeeze(1)


def predict_sdf_steps(model_encoder_path, model_decoder_path, volume_file, mean_sdf_file,
                      init_affine, steps=5, do_scale=True, do_rotate=False, do_pca=False, pca_components=None,
                      Apx2sdf=None):
    Asdf2px = np.linalg.inv(Apx2sdf)

    # if init_affine is None:
    #     init_affine = np.eye(4)
    #     init_affine[0, 3] = 77.4
    #     init_affine[1, 3] = 68.7
    #     init_affine[2, 3] = 118.2
    #     # account for the initial location at the center of the image
    init_affine[:3, 3] -= Asdf2px[:3, 3]

    mean_sdf = nib.load(mean_sdf_file)
    mean_sdf = mean_sdf.get_fdata()

    mean_latent_vec = torch.mean(torch.load(model_decoder_path)['component.latent_vector']['weight'], 0).unsqueeze(
        0).cuda()

    # mean_sdf = np.zeros_like(mean_sdf)
    affine_mtx = init_affine
    latent_vecs = None
    pca_coeffs = []
    for cur_step in range(steps):
        im_affine_mtx = np.linalg.inv(affine_mtx)
        affine_trans = Affine_w_Mtx(affine_mtx=torch.tensor(im_affine_mtx), padding_mode='border')
        new_mean_np = affine_trans(np.expand_dims(mean_sdf, 0))
        new_mean_np = new_mean_np[0, :]
        # if do_rotate:
        #     import pdb;
        #     pdb.set_trace()
        #     mean = nib.Nifti1Image(new_mean_np, np.eye(4))
        #     mean.to_filename('mean_test_{}.nii.gz'.format(cur_step))
        #

        data_dict = predict_params_channel(model_encoder_path, volume_file, new_mean_np, do_rotate, do_pca,
                                           latent_vecs=latent_vecs)
        if do_pca:
            pca_coeffs.append(data_dict['pca_coeff'])
        if 'pca_coeff' in data_dict.keys():
            scalar = data_dict['pca_coeff']

            latent_vecs = mean_latent_vec + inverse_transform(scalar, pca_components=pca_components)

        theta = data_dict['theta'][0, :].detach().cpu().numpy()
        # trans = data_dict['trans'][0, :].detach().cpu().numpy()
        # scale = data_dict['scale'][0, :].detach().cpu().numpy()

        cur_refine_aff = np.eye(4)
        if do_rotate:
            rotate = data_dict['rotate'].detach().cpu()
            rotate = compute_rotation_matrix_from_orthod(rotate)
            cur_refine_aff[:3, :3] = cur_refine_aff[:3, :3] @ rotate[0].numpy()
        if do_scale:
            cur_refine_aff[0, :] *= (theta[3] + 1)
            cur_refine_aff[1, :] *= (theta[4] + 1)
            cur_refine_aff[2, :] *= (theta[5] + 1)

        cur_refine_aff[0, 3] = -theta[0]
        cur_refine_aff[1, 3] = -theta[1]
        cur_refine_aff[2, 3] = -theta[2]
        # we describe an affine goign from canonical to image space
        affine_mtx = np.linalg.solve(cur_refine_aff, affine_mtx)
        # affine_mtx = cur_refine_aff @ affine_mtx

        print('cur_refine_aff', np.linalg.inv(cur_refine_aff))
        print('cur_aff', affine_mtx)

    print('Final Affine', affine_mtx)
    if do_pca:
        return affine_mtx, pca_coeffs
    return affine_mtx


def render_sdf(model_encoder_path_trans, model_decoder_path, decoder_config, volume_file, mean_sdf_file, save_path,
               Apx2sdf,
               init_affine=None, steps=5, model_encoder_path_scale=None, model_encoder_path_rotate=None,
               model_encoder_path_pca=None, pca_components_file=None,
               data_dict=None):
    trans_affine_mtx = predict_sdf_steps(model_encoder_path_trans, model_decoder_path, volume_file,
                                         mean_sdf_file, init_affine, steps=steps, do_scale=False, Apx2sdf=Apx2sdf)

    infer_affine_mtx = trans_affine_mtx
    if model_encoder_path_pca:
        pca_components = torch.from_numpy(np.load(pca_components_file)).float().cuda()

    if model_encoder_path_scale:
        scale_affine_mtx = predict_sdf_steps(model_encoder_path_scale, model_decoder_path, decoder_config, volume_file,
                                             mean_sdf_file, infer_affine_mtx, steps=steps, do_scale=True,
                                             Apx2sdf=Apx2sdf)
        infer_affine_mtx = scale_affine_mtx
    if model_encoder_path_rotate:
        rotate_affine_mtx = predict_sdf_steps(model_encoder_path_rotate, model_decoder_path, decoder_config,
                                              volume_file,
                                              mean_sdf_file, infer_affine_mtx, steps=steps, do_scale=True,
                                              do_rotate=True,
                                              Apx2sdf=Apx2sdf)
        infer_affine_mtx = rotate_affine_mtx
    if model_encoder_path_pca:
        # pca_affine_mtx, pca_coeffs = predict_sdf_steps(model_encoder_path_pca, model_decoder_path, decoder_config,
        #                                       volume_file,
        #                                       mean_sdf_file, infer_affine_mtx, steps=steps, do_scale=True,
        #                                       do_rotate=True,do_pca=True, pca_components=pca_components,
        #                                       Apx2sdf=Apx2sdf)
        pca_affine_mtx, pca_coeffs = predict_sdf_steps(model_encoder_path_pca, model_decoder_path, decoder_config,
                                                       volume_file,
                                                       mean_sdf_file, infer_affine_mtx, steps=steps, do_scale=True,
                                                       do_rotate=True, do_pca=True, pca_components=pca_components,
                                                       Apx2sdf=Apx2sdf)
        infer_affine_mtx = pca_affine_mtx

    # trans = infer_affine_mtx[:, 3][:3]
    # trans += [63.5, 63.5, 100.5]
    # basename = os.path.basename(volume_file).replace('.nii.gz', '')
    # if data_dict is not None:
    #     data_dict[basename] = {}
    #     data_dict[basename]['predict_trans'] = trans.tolist()
    #     data_dict[basename]['predict_scale'] = np.diag(infer_affine_mtx)[:3].tolist()
    #     data_dict[basename]['predict_rotate'] = infer_affine_mtx[:3, :3].tolist()
    if model_encoder_path_pca is not None:
        # import pdb;pdb.set_trace()
        loaded = torch.load(model_decoder_path)
        mean_latent_vec = torch.mean(loaded['component.latent_vector']['weight'], 0).cuda()
        latent_vec = mean_latent_vec + torch.matmul(pca_coeffs[1], pca_components).squeeze()
        # latent_vec = loaded['component.latent_vector']['weight'].numpy()[83]
        infer_sdf_from_latent(model_decoder_path, decoder_config, save_path, volume_file, infer_affine_mtx, Apx2sdf,
                              latent_vec)
    else:
        infer_mean_aligned_sdf(model_decoder_path, decoder_config, save_path, volume_file, infer_affine_mtx, Apx2sdf)





def render_sdfs(model_encoder_path_trans, model_decoder_path, decoder_config, volume_folder, mean_sdf_file, save_path,
               Apx2sdf,
               init_affine=None, steps=5, model_encoder_path_scale=None, model_encoder_path_rotate=None,
               model_encoder_path_pca=None, pca_components_file=None,
               data_dict=None):

    volume_files = glob.glob(os.path.join(volume_folder,'*.nii.gz'))
    os.makedirs(save_path, exist_ok=True)
    for volume_file in volume_files:
        global_init_mtx = np.copy(init_affine)

        print('reading {}'.format(volume_file))
        basename = os.path.basename(volume_file)
        render_sdf(model_encoder_path_trans, model_decoder_path, decoder_config, volume_file, mean_sdf_file, os.path.join(save_path,
                                                                                                                          basename),
                   Apx2sdf, global_init_mtx, steps, model_encoder_path_scale, model_encoder_path_rotate,
                   model_encoder_path_pca, pca_components_file)







def map_trans_to_original_res(model_encoder_path_trans, model_decoder_path, decoder_config, resampled_volume_file,
                              original_volume_file,
                              mean_sdf_file, save_path, Apx2sdf, init_affine=None, steps=5):
    infer_affine_mtx = predict_sdf_steps(model_encoder_path_trans, model_decoder_path, resampled_volume_file,
                                         mean_sdf_file, init_affine, steps=steps, do_scale=False, Apx2sdf=Apx2sdf)
    import pdb;
    pdb.set_trace()

    # reduced_mid_trans = [161 / 2, 161 / 2, 611 / 2]

    ref_im = sitk.ReadImage(original_volume_file)
    sdf_size = ref_im.GetSize()
    original_mid_trans = [(sdf_size[0] - 1) / 2, (sdf_size[1] - 1) / 2, (sdf_size[2] - 1) / 2]

    Apx2sdf = np.eye(4)
    Apx2sdf[0, 3] = original_mid_trans[0]
    Apx2sdf[1, 3] = original_mid_trans[1]
    Apx2sdf[2, 3] = original_mid_trans[2]
    Apx2sdf[0, :] *= -1 / 5
    Apx2sdf[1, :] *= -1 / 5
    Apx2sdf[2, :] *= 1 / 20 * (1.5 / 4)

    infer_mean_aligned_sdf(model_decoder_path, decoder_config, save_path, original_volume_file, infer_affine_mtx,
                           Apx2sdf)
