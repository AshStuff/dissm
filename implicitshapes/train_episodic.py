import argparse
import json
import os
import sys
from itertools import chain
import shutil
from copy import deepcopy

sys.path.append('config')
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from dlt_utils import read_yaml, init_workspace
from torch.utils.tensorboard import SummaryWriter

from networks.connected_module import ConnectedModel
from data_augmentation import Transforms, ScaleTransforms, RotateTransforms, PCATransforms
from dataset import SDFSamplesEpisodic, EpisodicDataloader, MetaDataset


def prepare_batch(data_dict, mean_latent_vec, embed_config):
    samples = data_dict['samples'].float()
    batch_size = samples.shape[0]
    num_scenes = samples.shape[1]

    # replicate the mean vector so we can concatenate it to every point for the embedding model
    mean_latent_vec = mean_latent_vec.repeat(batch_size, num_scenes, 1)
    mean_latent_vec = mean_latent_vec.reshape(-1, embed_config.solver.latent_size)

    data_dict['cur_latent_vecs'] = mean_latent_vec
    # extract the gt sdf val for each point
    gt = samples[:, :, 0]
    gt = gt.reshape(-1, 1)
    # reshape the points so we can cocatenate the mean vector to it
    samples = samples[:, :, 1:]
    samples = samples.reshape(-1, 3)
    # concatenate the mean vector
    input_samples = torch.cat((mean_latent_vec, samples), 1)
    # update the batch dict
    data_dict['samples'] = input_samples.cuda(non_blocking=True)
    data_dict['gt'] = gt.cuda(non_blocking=True)


class WeightedL1Loss(nn.Module):
    def __init__(self, m1, m2, thresh=0.01):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.thresh = thresh
        self.loss = nn.L1Loss()

    def forward(self, output, target):
        output = torch.where(output < self.thresh, output * self.m1, output * self.m2)
        # output[output < self.thresh] *= self.m1
        # output[output >= self.thresh] *= self.m2
        return self.loss(output, target)


def combined_loss(data_dict, l1_loss):
    # if args.do_scale:
    # get the gt sdf values
    # gt = data_dict['gt']
    # weights = gt.clone()
    # # threshold them to {0,1}
    # gt[gt<0] = 0
    # gt[gt>0] = 1
    # data_dict['gt'] = gt
    # import pdb;pdb.set_trace()
    # compute the ce loss
    loss = l1_loss(data_dict['output'], data_dict['gt'])
    # loss = loss * torch.abs(weights) + 0.1 * loss

    # loss = torch.mean(loss)
    # if neg_scale.nelement() != 0:
    # loss += neg_scale

    if args.do_scale:
        # now we also want to encourage the scale to fall within a value within [0.1, 3]
        scale = data_dict['predict_scale']
        # penalize any scale values > 3
        scale_select = scale.flatten()
        scale_select = scale_select[scale_select > 1.1]
        scale_select = scale_select - 1.1
        scale_select = torch.sum(scale_select)
        # print('scale_select', scale_select)
        if scale_select.nelement() != 0:
            loss += scale_select

        # penalized any scale values < 0.1
        scale_select_2 = scale.flatten()
        scale_select_2 = scale_select_2[scale_select_2 < 0.9]
        if scale_select_2.nelement() != 0:
            scale_select_2 = 0.9 - scale_select_2
            loss += torch.sum(scale_select_2)
        # print('scale_select_2', scale_select_2)
    return loss


def construct_refine_mtx(data_dict, init_mtx, aug_affine_mtx, cur_iter, do_scale=False):
    """
    given the current refinement which should be stored in data_dict
    and the current state which should be stored in data_dict, this will
    update the current state
    """
    affine_mtxs = []
    # fetch the current refinement
    rotate = data_dict['rotate'].detach().cpu().numpy()
    scale = data_dict['scale'].detach().cpu().numpy()
    trans = data_dict['trans'].detach().cpu().numpy()
    # fetch the matrix used to perform data augmentation, so we can undo it
    if np.isnan(trans).any():
        import pdb;
        pdb.set_trace()
    batch_size = scale.shape[0]

     
    for i in range(batch_size):
        # construct an affine from the current refinement
        cur_refine_aff = np.eye(4)
        cur_refine_aff[:3, :3] = cur_refine_aff[:3, :3] @ rotate[i]
        cur_refine_aff[0, :] *= scale[i, 0]
        cur_refine_aff[1, :] *= scale[i, 1]
        cur_refine_aff[2, :] *= scale[i, 2]
        cur_refine_aff[0, 3] = -trans[i, 0]
        cur_refine_aff[1, 3] = -trans[i, 1]
        cur_refine_aff[2, 3] = -trans[i, 2]
        # invert the current guess to make it go from canonical to pixel
        # and thus follow the conventions in the transforms
        init_mtx_inv = np.linalg.inv(init_mtx[i].T)
       
        # update the current guess with the refinement
        affine_mtx = np.linalg.solve(cur_refine_aff, init_mtx_inv)
        # if i==0:
            # import ipdb; ipdb.set_trace()

        # undo the geometric transform so we can start the next ministep fresh
        if aug_affine_mtx is not None:
            cur_aug_affine = aug_affine_mtx[i]
            affine_mtx = cur_aug_affine @ affine_mtx 

        if i == 0:
            gt_trans = [data_dict['t'][0][0].cpu().numpy(), data_dict['t'][1][0].cpu().numpy(), data_dict['t'][2][0].cpu().numpy()]
            print('Iter', cur_iter, 'Cur trans', affine_mtx[:3,3], 'gt_Trans', gt_trans)
            if do_scale:
                if data_dict['s'].dim() == 1:
                    gt_scale = [data_dict['s'][0].cpu().numpy(), data_dict['s'][0].cpu().numpy(), data_dict['s'][0].cpu().numpy()]

                else:
                    gt_scale = [data_dict['s'][0,0].cpu().numpy(), data_dict['s'][0,1].cpu().numpy(), data_dict['s'][0,2].cpu().numpy()]

                print( 'Cur Scale', [affine_mtx[0,0], affine_mtx[1,1], affine_mtx[2,2]], 'gt_Scale', gt_scale)

        affine_mtxs.append(affine_mtx)
    return np.array(affine_mtxs)


def load_pretrained_weights(current_model_dict, preloaded_model_dict):
    for key, current_model_value in current_model_dict.items():
        if key in preloaded_model_dict.keys():
            preloaded_model_value = preloaded_model_dict[key]
            if current_model_value.shape == preloaded_model_value.shape:
                current_model_dict[key] = preloaded_model_value
            else:
                shape = preloaded_model_value.shape
                if len(preloaded_model_value.shape) > 1:
                    current_model_value[:shape[0], :shape[1], :] = preloaded_model_value
                    current_model_dict[key] = current_model_value
                else:
                    current_model_value[:shape[0]] = preloaded_model_value
                    current_model_dict[key] = current_model_value
    return current_model_dict


def train(model, episodic_loader, loader, optimizer, mean_latent_vec, embed_config, tfboard, epoch, save_path, train_loss, sample_size, code_reg_lambda, mini_steps):
    model.train()
    avg_loss = []
    lr = optimizer.param_groups[0]['lr']
    length = len(loader)
    # SDF decoder is not trained here
    model.module.decoder.eval()
    # we choose specific samples
    for iter_idx, meta_data_dict in enumerate(loader):
        sub_total_loss = 0
        mini_step_size = mini_steps
        # get the actual index in the dataset of the samples
        idx = meta_data_dict['idx']
        # get any init guesses as to pose, if present
        init_mtx = meta_data_dict['init_mtx'].numpy()
        # setup random augmentation matrices
        aug_affine_mtxs = ['none'] * len(idx)
        ims = [None] * len(idx)
        gt_latent_vec_repeat = None
        latent_vecs = None
        # iterate for a set number of mini steps for the chosen images
        for step in range(mini_steps):
            torch.cuda.empty_cache()
            if step == 0:
                is_step_0 = True
            else:
                is_step_0 = False
            optimizer.zero_grad()
            # package the data needed for a batch in each mini_step
            batch_idx = [(cur_idx, cur_init_mtx, is_step_0, affine_mtx, im) for cur_idx, cur_init_mtx, affine_mtx, im in
                         zip(idx, init_mtx, aug_affine_mtxs, ims)]
            # forward it through the data loader to conduct necessary transforms
            data_dict = episodic_loader.forward(batch_idx)

            if 'gt_latent_vec' in data_dict.keys():
                gt_latent_vec = data_dict['gt_latent_vec']
                if gt_latent_vec_repeat is None:
                    gt_latent_vec_repeat = gt_latent_vec.repeat(1, sample_size, 1).view(-1, 256)
                    gt_latent_vec_repeat = gt_latent_vec_repeat.cuda(non_blocking=True)

            
            data_dict['im'] = data_dict['im'].cuda(non_blocking=True)
            # setup the  sdf coordinates to have the latent vector concatentated on top
            prepare_batch(data_dict, mean_latent_vec, embed_config)

            if latent_vecs is not None:
                data_dict['latent_vecs'] = latent_vecs.detach() + torch.normal(0, 0.001, size=latent_vecs.shape).cuda()

            # push data through the model
            data_dict = model(data_dict)

            latent_vecs = data_dict['latent_vecs']  # update the latent vecs

            # if it's the first ministep, then fetch the images for re-use later in the data loading/transforms
            # for subsequent steps
            if step == 0:
                ims = data_dict['im'][:,0,:].unsqueeze(1)

            # fetch the affine transform used to randomly augment the image for use in later steps of the data loading/transforms
            if 'aug_affine_mtx' in data_dict:
                aug_affine_mtxs = data_dict['aug_affine_mtx'].cpu().numpy()
            else:
                aug_affine_mtxs = ['none'] * len(idx)

            # compute loss
            loss = combined_loss(data_dict, train_loss)

            if 'gt_latent_vec' in data_dict.keys():
                pca_loss = torch.nn.functional.mse_loss(data_dict['samples_latent_vec'], gt_latent_vec_repeat)
                loss += pca_loss
            loss.backward()
            optimizer.step()
            print('Loss', loss.item())
            with torch.no_grad():
                sub_total_loss += loss.item()
                # update the current pose guess with the predicted refinement for use in subsequent ministeps
                init_mtx = data_dict['cur_affine_mtx'].cpu().numpy()
                init_mtx = construct_refine_mtx(data_dict, init_mtx, None, cur_iter=step, do_scale=model.module.do_scale)  # updated init matrix


        avg_loss.append(loss.item())

        print('Epoch: {} Iter: {}/{} LR: {} Loss: {}'.format(epoch, iter_idx, length, lr, loss.item()))
    avg_loss = np.mean(avg_loss)
    tfboard.add_scalar('train/loss', avg_loss, epoch)


def val(model, episodic_loader, loader, mean_latent_vec, embed_config, tfboard, epoch, val_loss, mini_steps):
    model.eval()
    avg_loss = []
    length = len(loader)

    for iter_idx, meta_data_dict in enumerate(loader):
        with torch.no_grad():
            torch.cuda.empty_cache()
            sub_total_loss = 0
            mini_batch_size = 15
            idx = meta_data_dict['idx']
            init_mtx = meta_data_dict['init_mtx'].numpy()
            affine_mtxs = ['none'] * len(idx)
            ims = [None] * len(idx)
            latent_vecs = None
            for step in range(mini_steps):
                if step == 0:
                    is_step_0 = True
                else:
                    is_step_0 = False
                batch_idx = [(i, j, is_step_0, affine_mtx, im) for i, j, affine_mtx, im in
                             zip(idx, init_mtx, affine_mtxs, ims)]
                data_dict = episodic_loader.forward(batch_idx)
                prepare_batch(data_dict, mean_latent_vec, embed_config)
                if latent_vecs is not None:
                    data_dict['latent_vecs'] = latent_vecs.detach()

                data_dict['im'] = data_dict['im'].cuda(non_blocking=True)
                data_dict = model(data_dict)
                latent_vecs = data_dict['latent_vecs']  # update the latent vecs

                ims = data_dict['im'][:, 0].data.cpu().unsqueeze(1)
                init_mtx = data_dict['cur_affine_mtx'].cpu().numpy()
                init_mtx = construct_refine_mtx(data_dict, init_mtx, None, step, do_scale=model.module.do_scale)  # updated init matrix
                loss = combined_loss(data_dict, val_loss)
                sub_total_loss += loss.item()
            print('Epoch: {} Iter: {}/{}  Loss: {}'.format(epoch, iter_idx, length, loss.item()))
            avg_loss.append(loss.item())

    avg_loss = np.mean(avg_loss)
    tfboard.add_scalar('val/loss', avg_loss, epoch)
    return avg_loss


def main(args):

    
    # torch.autograd.set_detect_anomaly(True)
    # torch.multiprocessing.set_start_method('spawn')
    torch.manual_seed(0)

    config = read_yaml(args.yaml_file)

    lr = config.solver.lr

    # set up study name depending ons ettings
    study_name = config.study.name 
    if args.do_scale:
        study_name += '_scale'
    if args.do_rotate:
        study_name += '_rotate'
    if args.do_pca:
        study_name += '_pca'
    study_name += '_' + str(lr)

    # create ckpt and log dirs
    workspace = os.path.join(args.save_path, 'runs', study_name)
    init_workspace(workspace)

    ckpt_path = os.path.join(args.save_path, 'ckpts', study_name)
    os.makedirs(ckpt_path, exist_ok=workspace)
    shutil.copyfile(args.yaml_file, os.path.join(ckpt_path, 'config.yml'))

    # grab mean shape vector
    loaded = torch.load(args.embed_model_path)
    latent_vecs = loaded['component.latent_vector']['weight']
    mean_latent_vec = torch.mean(latent_vecs, 0)
    mean_latent_vec = mean_latent_vec.unsqueeze(0)
    mean_latent_vec = mean_latent_vec.unsqueeze(0)

    # grab the config used for the shape embedding model
    embed_config = read_yaml(args.embed_yaml_file)

    im_dir = args.im_root
    # load the json list
    with open(args.train_json_list, 'r') as f:
        train_json_list = json.load(f)
    # train_json_list = train_json_list[:4]
    with open(args.val_json_list, 'r') as f:
        val_json_list = json.load(f)
   
    # grab the mean shape sdf
    mean_ni = nib.load(args.mean_sdf_file)
    mean_np = mean_ni.get_fdata()
    mean_affine = mean_ni.affine
    # get transform to center coordinates
    centering = -(np.asarray(mean_np.shape) - 1) / 2

    # create affine matrix to center pixel coordiantes (useful when conducting rigid transforms)
    center_affine = np.eye(4)
    center_affine[:3, 3] = centering

    scale = args.scale_factor

    # set up the transform from pixel to canonical coordinates
    Apx2sdf = np.eye(4)
    Apx2sdf[0, 3] = -(mean_np.shape[0] - 1)/2
    Apx2sdf[1, 3] = -(mean_np.shape[1] - 1)/2
    Apx2sdf[2, 3] = -(mean_np.shape[2] - 1)/2

    Apx2sdf[0, :] *= -mean_affine[0,0] / scale
    Apx2sdf[1, :] *= -mean_affine[1,1] / scale
    Apx2sdf[2, :] *= mean_affine[2,2] / scale

    gt_trans_key = 't'
    # global initialization will be the center of the image, so we don't pass anything for that

    # set up the transforms based on the MSL schedule
    if not args.do_scale and not args.do_rotate and not args.do_pca:
        transforms = Transforms(mean_np, Apx2sdf, gt_trans_key=gt_trans_key, im_dir=im_dir)

        mini_steps = config.solver.train_mini_steps_trans

    elif args.do_scale and not args.do_rotate and not args.do_pca:

        transforms = ScaleTransforms(mean_np, Apx2sdf, gt_trans_key=gt_trans_key, im_dir=im_dir, centering_affine=center_affine)
        mini_steps = config.solver.train_mini_steps_scale



    elif args.do_rotate and not args.do_pca:
        transforms = RotateTransforms(mean_np, Apx2sdf, gt_trans_key=gt_trans_key, im_dir=im_dir,
                                      global_init_mtx=cfg.global_init_mtx)
        mini_steps = config.solver.train_mini_steps_rotate

    elif args.do_pca:
        transforms = PCATransforms(mean_np, Apx2sdf, gt_trans_key=gt_trans_key, im_dir=im_dir,
                                   global_init_mtx=cfg.global_init_mtx)
        mini_steps = config.solver.train_mini_steps_pca

    else:
        raise ValueError('Unrecognized MSL schedule')
    train_step_0_transforms = transforms.train_step_0_transforms
    train_other_step_transforms = transforms.train_other_steps_transforms
    
    subsample = config.solver.subsample
    val_subsample = config.solver.val_subsample 


    # create dataset to load image and SDF samples
    train_dataset = SDFSamplesEpisodic(train_json_list, subsample, args.im_root, args.sdf_sample_root,
                                       step_0_transform=train_step_0_transforms,
                                       step_others_transform=train_other_step_transforms,
                                       gt_trans_key=gt_trans_key, load_ram=False)
    val_dataset = SDFSamplesEpisodic(val_json_list, val_subsample, args.im_root, args.sdf_sample_root,
                                     step_0_transform=transforms.val_step_0_transforms,
                                     step_others_transform=transforms.val_other_steps_transforms,
                                     gt_trans_key=gt_trans_key, load_ram=False)

    # create dummy dataset that simply dicates what images to load without actually doing it
    meta_train_dataset = MetaDataset(train_json_list, do_scale=args.do_scale, do_rotate=args.do_rotate,
                                     do_pca=args.do_pca,
                                     )
    meta_val_dataset = MetaDataset(val_json_list, do_scale=args.do_scale, do_rotate=args.do_rotate, do_pca=args.do_pca,
                                   )

    episodic_train_dataloader = EpisodicDataloader(train_dataset, num_workers=0)
    episodic_val_dataloader = EpisodicDataloader(val_dataset, num_workers=0)

    train_dataloader = torch.utils.data.DataLoader(
        meta_train_dataset,
        batch_size=config.solver.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    val_dataloader = torch.utils.data.DataLoader(
        meta_val_dataset,
        batch_size=config.solver.batch_size,
        shuffle=False,
        num_workers=0
    )
    pca_components = None
    if args.do_pca:
        pca_components = np.load(args.pca_components)
        pca_components = torch.from_numpy(pca_components).cuda()
        n_pca_components = pca_components.shape[0]
    else:
        n_pca_components = 28


    f_maps = config.model.f_maps
    # instantiate a model that connects an image encoder with the shape decoder
    context_model = ConnectedModel(12 + n_pca_components, embed_config, args.embed_model_path, do_scale=args.do_scale,
                                   cur_affine_key='cur_affine_mtx',
                                   Apx2sdf=Apx2sdf, centering_affine=center_affine, gt_trans_key=gt_trans_key,
                                   do_rotate=args.do_rotate, pca_components=pca_components, do_pca=args.do_pca,
                                   num_pca_components=n_pca_components, f_maps=f_maps).cuda()

    context_model = torch.nn.DataParallel(context_model).cuda()
    optimizer = torch.optim.AdamW(context_model.module.encoder.parameters(), lr=lr)

    if args.do_rotate:
        optimizer = torch.optim.AdamW(
            chain(context_model.module.encoder.parameters(),
                  context_model.module._rotate_classifier.parameters()),
            lr=lr)
    if args.do_pca:
        optimizer = torch.optim.AdamW(
            chain(context_model.module.encoder.parameters(),
                  context_model.module._rotate_classifier.parameters(),
                  context_model.module.pca_classifier.parameters(),
                  ),

            lr=lr)


    runs = os.path.join(args.save_path, 'runs')
    start_epoch = 1
    best_loss = np.Inf
    val_freq = 1
    if args.resume is not None:
        loaded = torch.load(args.resume)
        start_epoch = loaded['start_epoch']
        best_loss = loaded['best_loss']
        cur_model_dict = context_model.state_dict()
        updated_dict = load_pretrained_weights(cur_model_dict, loaded['state_dict'])
        cur_model_dict.update(updated_dict)
        context_model.load_state_dict(cur_model_dict)
    elif args.checkpoint is not None:
        loaded = torch.load(args.checkpoint)
        cur_model_dict = context_model.state_dict()
        updated_dict = load_pretrained_weights(cur_model_dict, loaded['state_dict'])
        cur_model_dict.update(updated_dict)
        context_model.load_state_dict(cur_model_dict)


    best_loss = np.Inf
    start_epoch = 1

    lr_step = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='min', patience=20)
    tfboard = SummaryWriter(log_dir=workspace)
    val_loss = nn.L1Loss()
    for cur_epoch in range(start_epoch, 20000):

        train(context_model, episodic_train_dataloader, train_dataloader, optimizer, mean_latent_vec, embed_config,
              tfboard, cur_epoch,
              args.save_path, val_loss, subsample, config.solver.code_reg_lambda, mini_steps=mini_steps)
        if val_freq and cur_epoch % val_freq == 0:
            loss = val(context_model, episodic_val_dataloader, val_dataloader, mean_latent_vec, embed_config, tfboard,
                       cur_epoch, val_loss, mini_steps=mini_steps)
            if loss < best_loss:
                print('loss: {} is better than prev loss: {}'.format(loss, best_loss))
                best_loss = loss
                torch.save({'state_dict': context_model.state_dict(),
                            'start_epoch': cur_epoch + 1,
                            'best_loss': best_loss},
                           os.path.join(ckpt_path, 'best_model.pth'))
            lr_step.step(loss)

        if cur_epoch and cur_epoch % 5 == 0:
            torch.save({'state_dict': context_model.state_dict(),
                        'start_epoch': cur_epoch + 1,
                        'best_loss': best_loss},
                       os.path.join(ckpt_path, 'epoch_{}.pth'.format(cur_epoch)))

        torch.save({'state_dict': context_model.state_dict(),
                    'start_epoch': cur_epoch + 1,
                    'best_loss': best_loss},
                   os.path.join(ckpt_path, 'last_checkpoint.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_root', type=str, help="root directory of images")
    parser.add_argument('--train_json_list', type=str, help='path to train json list file')
    parser.add_argument('--val_json_list', type=str, help='path to val json list file')

    parser.add_argument('--yaml_file', type=str, help="path to yaml file for configuration",
                        default='hyper-parameters.yml')
    parser.add_argument('--embed_model_path', type=str, help="path to saved shape embedding model")
    parser.add_argument('--pca_components', type=str, help="path to saved pca components")

    parser.add_argument('--resume', type=str, default=None, help="path to saved shape embedding model")

    parser.add_argument('--mean_sdf_file', type=str, help="path to mean shape mask file")
    parser.add_argument('--checkpoint', type=str, default="", help="path to saved total model")
    parser.add_argument('--do_scale', action='store_true', help="whether to predict scale as well ")
    parser.add_argument('--do_rotate', action='store_true', help="whether to predict rotate as well ")
    parser.add_argument('--do_pca', action='store_true', help="whether to predict pca as well ")

    parser.add_argument('--embed_yaml_file', type=str, help="path to config file for embedding model ")
    parser.add_argument('--sdf_sample_root', type=str, help="path to config file for embedding model ")
    parser.add_argument('--scale_factor', type=float, help="global scale factor for organ")
    parser.add_argument('--save_path', type=str, help="path to where you want checkpoints and logs save to",
                        default='./')
    args = parser.parse_args()

    main(args)
