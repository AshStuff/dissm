import numpy as np
import torch
from dlt_utils import load_model_from_ckpt
from torch import nn

from networks.deep_sdf_decoder import create_decoder_model
from networks.encoder import WrapperResidualEncoder


class Encoder(nn.Module):
    def __init__(self, num_data, trans_init=None):
        super().__init__()
        self.num_data = num_data
        self.trans_init = trans_init
        self.scales = nn.Parameter(torch.Tensor(num_data, 3))
        self.trans = nn.Parameter(torch.Tensor(num_data, 3))
        # *torch.FloatTensor([[77.4, 68.7, 118.2]])
        self.reset_parameters()

    def reset_parameters(self):
        if self.trans_init is not None:
            nn.init.constant_(self.trans, self.trans_init)
        else:
            nn.init.normal_(self.trans, 0, 5)
        nn.init.constant_(self.scales, 1.)
        self.scales.requires_grad = False

    def forward(self, idx):
        return {'scale': self.scales[idx],
                'trans': self.trans[idx]}


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


# poses batch*6
# poses
# def compute_rotation_matrix_from_ortho6d(poses):
#     x_raw = poses[:, 0:3]  # batch*3
#     y_raw = poses[:, 3:6]  # batch*3
#
#     x = normalize_vector(x_raw)  # batch*3
#     z = cross_product(x, y_raw)  # batch*3
#     z = normalize_vector(z)  # batch*3
#     y = cross_product(z, x)  # batch*3
#
#     x = x.view(-1, 3, 1)
#     y = y.view(-1, 3, 1)
#     z = z.view(-1, 3, 1)
#     matrix = torch.cat((x, y, z), 2)  # batch*3*3
#     return matrix

def compute_rotation_matrix_from_ortho6d(poses):
    batch_size = poses.shape[0]
    a1 = poses[:, 0:3]  # batch*3
    a2 = poses[:, 3:6]  # batch*3
    a1 = torch.FloatTensor([1, 0, 0]).repeat(batch_size, 1).to(a1.device) + a1
    a2 = torch.FloatTensor([0, 1, 0]).repeat(batch_size, 1).to(a1.device) + a2
    b1 = normalize_vector(a1)  # batch*3
    b2 = a2 - torch.sum(a2 * b1, dim=1).unsqueeze(1) * b1
    b2 = normalize_vector(b2)
    b3 = torch.cross(b1, b2)
    matrix = torch.stack((b1, b2, b3), 1)
    return matrix


class ConnectedModel(nn.Module):
    """
    Network module that connects both an image encoder and the shape embedding model
    Args:
        num_outputs: number of shape parameter outputs, here it is hardcoded to be 6 for now for translation and scale
        embed_config: config for latent space model
        embed_model_path: ckpt for shape embedding model
        cur_affine_key: the key in the batch_dict holding the current prediction of the affine transform. The encoder
            predicts a refinement of this affine transform
        do_scale: if true, we also estimate the scale
        rel_scales: global relative scales to apply to each point to transform pixel coordinates into canonical shape space.
            Should be based on the affine matrix of the images being processed
        global_scale: a global scale to transform pixel coordiantes into canonical shape space
    """

    def __init__(self, num_outputs, embed_config, embed_model_path, cur_affine_key, Apx2sdf, centering_affine,
                 do_scale=False, gt_trans_key='t', do_rotate=False, do_pca=False, pca_components=None,
                 num_pca_components=89, f_maps=32, debug=False, scale_lbound=0.9, scale_ubound=1.1):

        super().__init__()
        self.embed_config = embed_config
        self.embed_model_path = embed_model_path
        self.latent_size = embed_config.solver.latent_size
        self.do_scale = do_scale
        self.do_rotate = do_rotate
        self.do_pca = do_pca
        self.gt_trans_key = gt_trans_key
        self.debug = debug
        self.scale_lbound = scale_lbound
        self.scale_ubound = scale_ubound
        # self.encoder = Encoder(num_data)
        # self.scales = nn.Parameter(torch.Tensor((num_data,3)), requires_grad=True).cuda()
        # create the shape embedding model from the config and load the model weights
        self.decoder = create_decoder_model(embed_config)

        load_model_from_ckpt(self.decoder, embed_model_path)
        # create image embedding module
        self.encoder = WrapperResidualEncoder(num_outputs, in_channels=2, f_maps=f_maps, do_pca=do_pca)
        self._rotate_classifier = nn.Linear(self.encoder._model.output_size, 6)
        # self._latent_reduce = nn.Linear(256, 128)
        self._pca_components = pca_components
        if do_pca:
            assert self._pca_components is not None
            self.pca_classifier = nn.Linear(self.encoder._model.output_size, num_pca_components)
            # self.pca_classifier = nn.Linear(768, num_pca_components)

        # self.encoder = ConcatEncoder(num_outputs, in_channels=1, f_maps=16)
        self.cur_affine_key = cur_affine_key
        # turn the pixel to canonical space transform into row vector convention
        self.Apx2sdf = torch.tensor(np.transpose(Apx2sdf)).float().cuda()
        # create centering affine to use in transforming point coordinates (to  make the origin of rotation and scale the center
        # of the image)
        self.centering_affine = torch.tensor(np.transpose(centering_affine)).float().cuda()
        # and the inverse centering transform
        self.inv_centering_affine = torch.tensor(np.transpose(np.linalg.inv(centering_affine))).float().cuda()

    def inverse_transform(self, scalar):
        batch_size = scalar.shape[0]
        pca_components = self._pca_components.repeat(batch_size, 1, 1)
        return torch.matmul(scalar.unsqueeze(1), pca_components.to(scalar.device)).squeeze(1)

    def forward(self, data_dict):

        self.decoder.eval()
        # get the current shape parameter predictions from the image
        batch_size = data_dict['im'].shape[0]
        if self.do_pca:
            if 'latent_vecs' in data_dict.keys():
                latent_vecs = data_dict['latent_vecs']
                # latent_vecs_reduce = self._latent_reduce(latent_vecs)
            else:
                latent_vecs = torch.zeros((batch_size, 256)).cuda()
            data_dict['latent_vecs'] = latent_vecs

        data_dict = self.encoder(data_dict)
        if self.do_rotate:
            rotate_params = self._rotate_classifier(self.encoder.features)
            rotate = compute_rotation_matrix_from_ortho6d(rotate_params)
        else:
            rotate = torch.eye(3)
            rotate = rotate.reshape((1, 3, 3))
            rotate = rotate.repeat(batch_size, 1, 1).cuda()

        # get the current point/latent-space vectors
        samples = data_dict['samples']
        # suppress errors due to samples not having gradients
        samples = samples + 0
        orig_sample_shape = samples.shape

        batch_size = data_dict['theta'].shape[0]

        # get the GT translation
        json_trans = data_dict[self.gt_trans_key]
        gt_trans = torch.zeros((batch_size, 3))

        for idx, cur_trans in enumerate(json_trans):
            gt_trans[:, idx] = cur_trans.float().cuda()
        # reshape the point samples and extract the points for each image
        samples = samples.reshape((batch_size, -1, self.latent_size + 3))
        if self.do_pca:
            mean = samples[:, :, :self.latent_size][:, 0]
            if data_dict['is_step_0'][0]:
                pca_features = torch.cat((self.encoder.local_features, mean.to(self.encoder.local_features.device)), 1)
            else:
                pca_features = torch.cat(
                    (self.encoder.local_features, data_dict['latent_vecs'].to(self.encoder.local_features.device)), 1)

            pca_scalar = self.pca_classifier(pca_features)
            data_dict['pca'] = pca_scalar
            # mean.requires_grad = False
            if data_dict['is_step_0'][0]:
                per_image_recon = mean + self.inverse_transform(pca_scalar)
            else:
                per_image_recon = latent_vecs + self.inverse_transform(pca_scalar)
            data_dict['latent_vecs'] = per_image_recon
            samples[:, :, :self.latent_size] = per_image_recon.unsqueeze(1).repeat(1, samples.shape[1], 1)
        else:
            data_dict['latent_vecs'] = None
        points = samples[:, :, -3:]
        # used to create homogenous coordinates
        dummy = torch.ones(points.shape[1], 1).cuda()

        # get the current affine prediction, as embodied by the mask/sdf
        cur_affine = data_dict[self.cur_affine_key].float()
        # get the current translation guess
        trans = data_dict['theta'][:, :3]

        # set it to the default value if we are not predicting scale
        scale = torch.ones_like(trans).cuda()

        # if we are predicting scale, then use it as a modification of a no scale transform
        if self.do_scale:
            
            # update the sampled coordinates with the current refinement
            # trans = trans.unsqueeze(1)
            scale = data_dict['theta'][:, 3:6]
            # we predict deviations from 1
            scale = scale + 1
            # store the unclipped scale prediction so we can penalize deviations from the margin
            data_dict['predict_scale'] = scale + 0
            scale[scale > self.scale_ubound] = self.scale_ubound
            scale[scale < self.scale_lbound] = self.scale_lbound


        self.debug = False
        if self.debug:
            trans[:] = 0

        data_dict['rotate'] = rotate
        data_dict['scale'] = scale
        data_dict['trans'] = trans

            

        # if self.debug:
            # # just printing out some info
            # print('scale', scale)
            # print('total_trans', data_dict['cur_predict_trans'][0, :])

            # if scale.device.index == 0:
                # print('scale', scale[0, :])
                # print('scale:', 1 / scale[0, 0])
                # print('total_trans', data_dict['cur_predict_trans'][0, :])
                # print('tran', trans[0, :])
                # print('center_x:', gt_trans[0, 0])
                # print('center_y:', gt_trans[0, 1])
                # print('center_z:', gt_trans[0, 2])

        # for each image transform the points by the current affine prediction
        # if C is centering affine; T is encoder prediction, and M is the current guess
        # in ROW VECTOR convention, this is the what is being computed
        # p = i * C * T * M * inv(C) * Apx2sdf
        for i in range(batch_size):
            # get the current image's points
            cur_points = points[i, :, :]
            # make them homogenous
            cur_points = torch.cat((cur_points, dummy), 1)
            # transform them by the current prediction. Note, we use row vector convention here
            # first we center the points
            cur_points = torch.matmul(cur_points, self.centering_affine.to(cur_points.device))
            # make sure we also center the poitns before applying any affine transforms
            A = torch.matmul(cur_affine[i, :, :], self.inv_centering_affine.to(cur_points.device))
            # take into account the transform to canonical coordinates
            A = torch.matmul(A, self.Apx2sdf.to(A.device))

            # now apply the Rotation, translation and scale prediction

            # prediction_matrix = construct_affine_matrix(data_dict)
            #
            # cur_points[:, :3] = cur_points[:, :3] @ prediction_matrix

            cur_points[:, :3] = cur_points[:,:3] @ rotate[i].T
            cur_points[:, :3] *= scale[i, :]
            cur_points[:, :3] -= trans[i, :]

            # transform into canonical space
            cur_points = torch.matmul(cur_points, A)
            # make them inhomogenous

            points[i, :, :] = cur_points[:, :3]

        # update batch_dict sample by the updated points
        samples[:, :, -3:] = points

        samples = samples.reshape(orig_sample_shape)
        data_dict['samples_latent_vec'] = samples[:, :-3]
        data_dict['samples'] = samples

        # now extract the sdf values based on the given points and latent vector pairs
        data_dict = self.decoder(data_dict)
        return data_dict
