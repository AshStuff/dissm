import argparse
import json
import os
import sys
from itertools import chain

sys.path.append('config')
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from implicitshapes.dlt_utils import read_yaml, init_workspace
from tensorboardX import SummaryWriter

from connected_module import ConnectedModel
from data_augmentation import Transforms, ScaleTransforms, RotateTransforms, PCATransforms
from dataset import SDFSamplesEpisodic, EpisodicDataloader, MetaDataset
from larynx_affine_config import Config


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
        scale_select = scale_select[scale_select > 2]
        scale_select = scale_select - 2
        scale_select = torch.sum(scale_select)
        # print('scale_select', scale_select)
        if scale_select.nelement() != 0:
            loss += scale_select

        # penalized any scale values < 0.1
        scale_select_2 = scale.flatten()
        scale_select_2 = scale_select_2[scale_select_2 < 0.1]
        if scale_select_2.nelement() != 0:
            scale_select_2 = 0.1 - scale_select_2
            loss += torch.sum(scale_select_2)
        # print('scale_select_2', scale_select_2)
    return loss


def construct_refine_mtx(data_dict, init_mtx, iter):
    affine_mtxs = []
    rotate = data_dict['rotate'].detach().cpu().numpy()
    scale = data_dict['scale'].detach().cpu().numpy()
    trans = data_dict['trans'].detach().cpu().numpy()
    if np.isnan(trans).any():
        import pdb;
        pdb.set_trace()
    batch_size = scale.shape[0]
    print('predict trans', trans)
    for i in range(batch_size):
        cur_refine_aff = np.eye(4)
        cur_refine_aff[:3, :3] = cur_refine_aff[:3, :3] @ rotate[i]
        cur_refine_aff[0, :] *= scale[i, 0]
        cur_refine_aff[1, :] *= scale[i, 1]
        cur_refine_aff[2, :] *= scale[i, 2]
        cur_refine_aff[0, 3] = -trans[i, 0]
        cur_refine_aff[1, 3] = -trans[i, 1]
        cur_refine_aff[2, 3] = -trans[i, 2]
        init_mtx_inv = np.linalg.inv(init_mtx[i].T)
        print('refine_arr', cur_refine_aff)
        affine_mtx = np.linalg.solve(cur_refine_aff, init_mtx_inv)
        print('Iter: {}'.format(iter), affine_mtx)
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


def train(model, episodic_loader, loader, optimizer, mean_latent_vec, embed_config, tfboard, epoch, save_path,
          train_loss, sample_size, code_reg_lambda):
    model.train()
    avg_loss = []
    lr = optimizer.param_groups[0]['lr']
    length = len(loader)
    model.module.decoder.eval()
    for iter_idx, meta_data_dict in enumerate(loader):
        print(save_path)
        sub_total_loss = 0
        mini_step_size = 15
        idx = meta_data_dict['idx']
        init_mtx = meta_data_dict['init_mtx'].numpy()
        affine_mtxs = ['none'] * len(idx)
        ims = [None] * len(idx)
        gt_latent_vec_repeat = None
        latent_vecs = None
        for step in range(mini_step_size):
            torch.cuda.empty_cache()
            if step == 0:
                is_step_0 = True
            else:
                is_step_0 = False
            optimizer.zero_grad()
            batch_idx = [(i, j, affine_mtx, is_step_0, im) for i, j, affine_mtx, im in
                         zip(idx, init_mtx, affine_mtxs, ims)]
            data_dict = episodic_loader.forward(batch_idx)

            if 'gt_latent_vec' in data_dict.keys():
                gt_latent_vec = data_dict['gt_latent_vec']
                if gt_latent_vec_repeat is None:
                    gt_latent_vec_repeat = gt_latent_vec.repeat(1, sample_size, 1).view(-1, 256)
                    gt_latent_vec_repeat = gt_latent_vec_repeat.cuda(non_blocking=True)

            data_dict['im'] = data_dict['im'].cuda(non_blocking=True)
            prepare_batch(data_dict, mean_latent_vec, embed_config)

            if latent_vecs is not None:
                data_dict['latent_vecs'] = latent_vecs.detach() + torch.normal(0, 0.001, size=latent_vecs.shape).cuda()

            data_dict = model(data_dict)

            latent_vecs = data_dict['latent_vecs']  # update the latent vecs
            ims = data_dict['im'][:, 0].data.cpu().unsqueeze(1)
            try:
                affine_mtxs = data_dict['affine_matrix'].cpu().numpy()
            except:
                affine_mtxs = ['none'] * len(idx)
            init_mtx = data_dict['cur_affine_mtx'].cpu().numpy()
            init_mtx = construct_refine_mtx(data_dict, init_mtx, iter=step)  # updated init matrix
            loss = combined_loss(data_dict, train_loss)

            if 'gt_latent_vec' in data_dict.keys():
                pca_loss = torch.nn.functional.mse_loss(data_dict['samples_latent_vec'], gt_latent_vec_repeat)
                loss += pca_loss
            loss.backward()
            sub_total_loss += loss.item()
            optimizer.step()
        avg_loss.append(sub_total_loss)
        print('Epoch: {} Iter: {}/{} LR: {} Loss: {}'.format(epoch, iter_idx, length, lr, sub_total_loss))
    avg_loss = np.mean(avg_loss)
    tfboard.add_scalar('train/loss', avg_loss, epoch)


def val(model, episodic_loader, loader, mean_latent_vec, embed_config, tfboard, epoch, val_loss):
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
            for step in range(mini_batch_size):
                if step == 0:
                    is_step_0 = True
                else:
                    is_step_0 = False
                batch_idx = [(i, j, affine_mtx, is_step_0, im) for i, j, affine_mtx, im in
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
                init_mtx = construct_refine_mtx(data_dict, init_mtx, step)  # updated init matrix
                loss = combined_loss(data_dict, val_loss)
                sub_total_loss += loss.item()
            print('Epoch: {} Iter: {}/{}  Loss: {}'.format(epoch, iter_idx, length, sub_total_loss))
            avg_loss.append(sub_total_loss)
    avg_loss = np.mean(avg_loss)
    tfboard.add_scalar('val/loss', avg_loss, epoch)
    return avg_loss


def main(args):
    # trans_center = [-(161.0 / 2), -(161.0 / 2), -(611.0 / 2)]
    # if args.do_scale:
    #     trans_center = [-164 / 2, -70 / 2,  -339 / 2]

    trans_center = [-63.5, -63.5, -100.5]
    if args.do_scale:
        trans_center = [-127. / 2, -107. / 2, -87. / 2]

    #     trans_center = [-95.0/2, -95.0/2, -95.0/2]
    cfg = Config(trans_center=trans_center, scale=30.)
    # torch.autograd.set_detect_anomaly(True)
    # torch.multiprocessing.set_start_method('spawn')
    torch.manual_seed(0)
    ###############################################################################################
    ###### Step 0. Environment Setup
    ###############################################################################################

    # Step 0.1 Read Configuration Document
    config = read_yaml(args.yaml_file)

    lr = config.solver.lr

    # here we only have two settings so far, translation and translation+scale
    study_name = config.study.name
    if args.do_scale:
        study_name += '_scale'
    study_name += '_' + str(lr)

    # create ckpt and log dirs
    workspace = os.path.join(args.save_path, 'runs', study_name)
    init_workspace(workspace)

    ckpt_path = os.path.join(args.save_path, 'ckpts', study_name)
    os.makedirs(ckpt_path, exist_ok=True)

    # grab mean shape vector
    loaded = torch.load(args.embed_model_path)
    latent_vecs = loaded['component.latent_vector']['weight']
    mean_latent_vec = torch.mean(latent_vecs, 0)
    mean_latent_vec = mean_latent_vec.unsqueeze(0)
    mean_latent_vec = mean_latent_vec.unsqueeze(0)

    # grab the config used for the shape embedding model
    embed_config = read_yaml(args.embed_yaml_file)

    ###############################################################################################
    ###### Step 1. Data Preparation
    ###############################################################################################
    im_dir = args.im_root
    # load the json list
    with open(args.train_json_list, 'r') as f:
        train_json_list = json.load(f)
    # train_json_list = [train_json_list[43]]
    with open(args.val_json_list, 'r') as f:
        val_json_list = json.load(f)
    latent_vec_dict = None
    if args.latent_vec_dict_file is not None:
        with open(args.latent_vec_dict_file) as f:
            latent_vec_dict = json.load(f)
        # latent_vec_dict = np.load(args.latent_vec_dict_file, allow_pickle=True)
        # latent_vec_dict = np.load(args.latent_vec_dict_file, allow_pickle=True)['latent_vec_dict']
        # import pdb;pdb.set_trace()

    # grab the mean shape sdf
    mean_ni = nib.load(args.mean_sdf_file)
    mean_np = mean_ni.get_fdata()
    mean_affine = mean_ni.affine
    centering = -(np.asarray(mean_np.shape) - 1) / 2

    # create affine matrix to center pixel coordiantes (useful when conducting rigid transforms)
    center_affine = np.eye(4)
    center_affine[:3, 3] = centering

    Apx2sdf = np.eye(4)
    Apx2sdf[0, 3] = -(mean_np.shape[0] - 1)/2
    Apx2sdf[1, 3] = -(mean_np.shape[1] - 1)/2
    Apx2sdf[2, 3] = -(mean_np.shape[2] - 1)/2

    Apx2sdf[0, :] *= -mean_affine[0,0] / scale
    Apx2sdf[1, :] *= -mean_affine[1,1] / scale
    Apx2sdf[2, :] *= mean_affine[2,2] / scale


    gt_trans_key = 't'
    transforms = Transforms(mean_np, cfg.Apx2sdf, gt_trans_key=gt_trans_key, im_dir=im_dir,
                            global_init_mtx=cfg.global_init_mtx)
    if args.do_scale:
        transforms = ScaleTransforms(mean_np, cfg.Apx2sdf, gt_trans_key=gt_trans_key, im_dir=im_dir,
                                     global_init_mtx=cfg.global_init_mtx)
    if args.do_rotate:
        transforms = RotateTransforms(mean_np, cfg.Apx2sdf, gt_trans_key=gt_trans_key, im_dir=im_dir,
                                      global_init_mtx=cfg.global_init_mtx)
    if args.do_pca:
        transforms = PCATransforms(mean_np, cfg.Apx2sdf, gt_trans_key=gt_trans_key, im_dir=im_dir,
                                   global_init_mtx=cfg.global_init_mtx)

    train_step_0_transforms = transforms.train_step_0_transforms
    train_other_step_transforms = transforms.train_other_steps_transforms
    subsample = 150000

    train_dataset = SDFSamplesEpisodic(train_json_list, subsample, args.im_root, args.sdf_sample_root,
                                       step_0_transform=train_step_0_transforms,
                                       step_others_transform=train_other_step_transforms,
                                       gt_trans_key=gt_trans_key, latent_vec_dict=latent_vec_dict, load_ram=False)
    val_dataset = SDFSamplesEpisodic(val_json_list, subsample, args.im_root, args.sdf_sample_root,
                                     step_0_transform=transforms.val_step_0_transforms,
                                     step_others_transform=transforms.val_other_steps_transforms,
                                     gt_trans_key=gt_trans_key, latent_vec_dict=latent_vec_dict, load_ram=False)

    meta_train_dataset = MetaDataset(train_json_list, do_scale=args.do_scale, do_rotate=args.do_rotate,
                                     do_pca=args.do_pca,
                                     trans_center=trans_center)
    meta_val_dataset = MetaDataset(val_json_list, do_scale=args.do_scale, do_rotate=args.do_rotate, do_pca=args.do_pca,
                                   trans_center=trans_center)

    episodic_train_dataloader = EpisodicDataloader(train_dataset)
    episodic_val_dataloader = EpisodicDataloader(val_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        meta_train_dataset,
        batch_size=config.solver.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.solver.batch_size,
    )

    val_dataloader = torch.utils.data.DataLoader(
        meta_val_dataset,
        batch_size=config.solver.batch_size,
        shuffle=False,
        num_workers=config.solver.num_workers,
    )
    pca_components = None
    if args.do_pca:
        pca_components = np.load(args.pca_components)
        pca_components = torch.from_numpy(pca_components).cuda()
        n_pca_components = pca_components.shape[0]
    else:
        n_pca_components = 28
    # instantiate a model that connects an image encoder with the shape decoder
    context_model = ConnectedModel(6, embed_config, args.embed_model_path, do_scale=args.do_scale,
                                   cur_affine_key='cur_affine_mtx',
                                   Apx2sdf=cfg.Apx2sdf, centering_affine=center_affine, gt_trans_key=gt_trans_key,
                                   do_rotate=args.do_rotate, pca_components=pca_components, do_pca=args.do_pca,
                                   num_pca_components=n_pca_components, f_maps=32).cuda()

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

    ckpt_path = os.path.join(args.save_path, 'ckpt')
    os.makedirs(ckpt_path, exist_ok=True)

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

    best_loss = np.Inf
    start_epoch = 1

    lr_step = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='min', patience=20)
    tfboard = SummaryWriter(logdir=runs)
    val_loss = nn.L1Loss()
    for cur_epoch in range(start_epoch, 20000):

        train(context_model, episodic_train_dataloader, train_dataloader, optimizer, mean_latent_vec, embed_config,
              tfboard, cur_epoch,
              args.save_path, val_loss, subsample, config.solver.code_reg_lambda)
        if val_freq and cur_epoch % val_freq == 0:
            loss = val(context_model, episodic_val_dataloader, val_dataloader, mean_latent_vec, embed_config, tfboard,
                       cur_epoch, val_loss)
            if loss < best_loss:
                print('loss: {} is better than prev loss: {}'.format(loss, best_loss))
                best_loss = loss
                torch.save({'state_dict': context_model.state_dict(),
                            'start_epoch': cur_epoch + 1,
                            'best_loss': best_loss},
                           os.path.join(args.save_path, 'ckpt', 'best_model.pth'))
            lr_step.step(loss)

        if cur_epoch and cur_epoch % 5 == 0:
            torch.save({'state_dict': context_model.state_dict(),
                        'start_epoch': cur_epoch + 1,
                        'best_loss': best_loss},
                       os.path.join(args.save_path, 'ckpt', 'epoch_{}.pth'.format(cur_epoch)))

        torch.save({'state_dict': context_model.state_dict(),
                    'start_epoch': cur_epoch + 1,
                    'best_loss': best_loss},
                   os.path.join(args.save_path, 'ckpt', 'last_checkpoint.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_root', type=str, help="root directory of images")
    parser.add_argument('--train_json_list', type=str, help='path to train json list file')
    parser.add_argument('--val_json_list', type=str, help='path to val json list file')

    parser.add_argument('--yaml_file', type=str, help="path to yaml file for configuration",
                        default='hyper-parameters.yml')
    parser.add_argument('--embed_model_path', type=str, help="path to saved shape embedding model")
    parser.add_argument('--pca_components', type=str, help="path to saved pca components")
    parser.add_argument('--latent_vec_dict_file', type=str, default=None,
                        help="path to saved latent vector with respective filename")

    parser.add_argument('--resume', type=str, default=None, help="path to saved shape embedding model")

    parser.add_argument('--mean_sdf_file', type=str, help="path to mean shape mask file")
    parser.add_argument('--checkpoint', type=str, default="", help="path to saved total model")
    parser.add_argument('--do_scale', action='store_true', help="whether to predict scale as well ")
    parser.add_argument('--do_rotate', action='store_true', help="whether to predict rotate as well ")
    parser.add_argument('--do_pca', action='store_true', help="whether to predict pca as well ")

    parser.add_argument('--embed_yaml_file', type=str, help="path to config file for embedding model ")
    parser.add_argument('--sdf_sample_root', type=str, help="path to config file for embedding model ")

    parser.add_argument('--save_path', type=str, help="path to where you want checkpoints and logs save to",
                        default='./')
    args = parser.parse_args()

    main(args)
