from torch import nn
import numpy as np
import torch
from networks.encoder import WrapperResidualEncoder
from networks.deep_sdf_decoder import create_decoder_model


# adapted from
# https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/utils/torchtools.py
# should be able to load a state dict that was wrapped with DataParallel. Should also be
# able to load into a wrapped DataParallel model
def load_model_from_ckpt(model, ckpt_path, partial=False, partial_ckpt=False):
    """
    When partial = True, the specified model is allowed to have weights that are not
    present in the saved ckpt. Thus it will only load the subset of weights within the
    ckpt.
    When partial_ckpt = True, this allows the reverse, i.e., the ckpt to have weights not
    present in the specified model
    """

    loaded = torch.load(ckpt_path)
    model_state_dict = loaded['component.model']

    new_state_dict = OrderedDict()
    # strip 'module.' from every key if applicable
    for k,v in model_state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module prefix
        new_state_dict[k] = v


    if partial_ckpt:
        new_state_dict = load_partial_ckpt(model, new_state_dict)

    if not partial:
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(new_state_dict)
    # if we are using partial loading
    else:
        if isinstance(model, torch.nn.DataParallel):
            # get the specified model state dict
            cur_model_dict = model.module.state_dict()
            # update it with the loaded weights
            cur_model_dict.update(new_state_dict)
            # load it into the specified model
            model.module.load_state_dict(cur_model_dict)
        else:
            cur_model_dict = model.state_dict()
            cur_model_dict.update(new_state_dict)
            model.load_state_dict(cur_model_dict)

    return model


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
    def __init__(self, num_outputs, embed_config, embed_model_path, cur_affine_key, Apx2sdf, centering_affine, do_scale=False, gt_trans_key='t'):

        super().__init__()
        self.embed_config = embed_config
        self.embed_model_path = embed_model_path
        self.latent_size = embed_config.solver.latent_size
        self.do_scale = do_scale
        self.gt_trans_key = gt_trans_key
        # create the shape embedding model from the config and load the model weights
        self.decoder = create_decoder_model(embed_config)
        
        load_model_from_ckpt(self.decoder, embed_model_path)
        # create image embedding module
        self.encoder = WrapperResidualEncoder(num_outputs, in_channels=2, f_maps=32)


        self._rotate_classifier = nn.Linear(self.encoder._model.output_size, 3)


        self.cur_affine_key = cur_affine_key
        # turn the pixel to canonical space transform into row vector convention
        self.Apx2sdf = torch.tensor(np.transpose(Apx2sdf)).float().cuda()
        # create centering affine to use in transforming point coordinates (to  make the origin of rotation and scale the center 
        # of the image)
        self.centering_affine = torch.tensor(np.transpose(centering_affine)).float().cuda()
        # and the inverse centering transform
        self.inv_centering_affine = torch.tensor(np.transpose(np.linalg.inv(centering_affine))).float().cuda()

        # APH: not sure if we need these, but these can be used to try to harmonize the SDFs if using direct SDF-based loss
        # if self.do_scale:
        self.alpha = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.beta = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
            

    def forward(self, data_dict):
        self.decoder.eval()
        # get the current shape parameter predictions from the image
        data_dict = self.encoder(data_dict)

        rotate_params = self._rotate_classifier(self.encoder.features)

        # get the current point/latent-space vectors
        samples = data_dict['samples']

        orig_sample_shape = samples.shape
        batch_size = data_dict['theta'].shape[0]

        # get the GT translation 
        json_trans = data_dict[self.gt_trans_key]
        gt_trans = torch.zeros((batch_size, 3)) 

        for idx, cur_trans in enumerate(json_trans):
            gt_trans[:,idx] = cur_trans.float().cuda()


        # reshape the point samples and extract the points for each image
        samples = samples.reshape((batch_size, -1, self.latent_size+3))
        points = samples[:,:,-3:]
        # used to create homogenous coordinates
        dummy = torch.ones(points.shape[1], 1).cuda()


                        
        # get the current affine prediction, as embodied by the mask/sdf
        cur_affine = data_dict[self.cur_affine_key].float()

        # if we are predicting scale, then use it as a modification of a no scale transform
        if self.do_scale:

            # get the current translation guess
            trans = data_dict['theta'][:,:3]
            # get the current translation guess trans = data_dict['theta'][:,:3]
            # update the current translation guess with the encoder's refinement    
            data_dict['cur_predict_trans'] = cur_affine[:, 3,:3] - trans  
            # update the sampled coordinates with the current refinement
            # trans = trans.unsqueeze(1)
            scale = data_dict['theta'][:,3:]

            
            scale = scale + 1

            # ******APH REMOVE***********
            # scale[:] = 1
            data_dict['predict_scale'] = scale
            scale = scale + 0
            scale[scale>2] = 2
            scale[scale<0.5] = 0.5
        else:

            # get the current translation guess
            trans = data_dict['theta'][:,:3]
            # update the current translation guess with the encoder's refinement    
            data_dict['cur_predict_trans'] = cur_affine[:, 3,:3] - trans  
            
            # update the sampled coordinates with the current refinement
            # trans = trans.unsqueeze(1)
            # get the current scale estimate
            scale = data_dict['theta'][:,3:]
            # set it to the default value if we are not predicting scale
            scale[:] = 1


        data_dict['scale'] = scale

        
        # just printing out some info
        if scale.device.index == 0:
            print(data_dict['copy_path'])
            print('scale', scale[0,:])
            print('scale:', 1/scale[0,0])
            print('total_trans', data_dict['cur_predict_trans'][0,:])
            print('tran', trans[0,:])
            print('center_x:', gt_trans[0,0]-127.5, gt_trans[0,0]-127.5+cur_affine[0,3,0])
            print('center_y:',gt_trans[0,1]-127.5, gt_trans[0,1]-127.5+cur_affine[0,3,1])
            print('center_z:',gt_trans[0,2]-80.5, gt_trans[0,2]-80.5+cur_affine[0,3,2])

        # alter the scale by the encoder's prediction
        # points -= trans
        # points *= scale 
        # scale[:] = 1
        # for each image transform the points by the current affine prediction
        # if C is centering affine; T is encoder prediction, and M is the current guess
        # in ROW VECTOR convention, this is the what is being computed
        # p = i * C * T * M * inv(C) * Apx2sdf
        for i in range(batch_size):

            # get the current image's points
            cur_points = points[i,:,:]
            # make them homogenous
            cur_points = torch.cat((cur_points, dummy), 1)
            # transform them by the current prediction. Note, we use row vector convention here
            # first we center the points
            cur_points = torch.matmul(cur_points, self.centering_affine)
            # make sure we also center the poitns before applying any affine transforms
            A = torch.matmul(cur_affine[i,:,:], self.inv_centering_affine)
            # take into account the transform to canonical coordinates
            A = torch.matmul(A, self.Apx2sdf)

            # now apply the translation and scale prediction

            cur_points[:,:3] *= scale[i,:]


            cur_points[:,:3] -= trans[i,:]
            
            # transform into canonical space
            cur_points = torch.matmul(cur_points, A)
            # make them inhomogenous
            points[i,:,:] = cur_points[:,:3]



        # update batch_dict sample by the updated points
        samples[:,:, -3:] = points
        samples[:,:,0] *= 100
        # adjust SDF values to overcome bias and scale
        # if self.do_scale:
        # samples[:,:,0] *= self.alpha
        # samples[:,:,0] += self.beta

        samples = samples.reshape(orig_sample_shape)

        data_dict['samples'] = samples

        # now extract the sdf values based on the given points and latent vector pairs
        data_dict = self.decoder(data_dict)
        return data_dict

