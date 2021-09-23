import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from dlt.common.utils.setup import read_yaml, init_workspace
from kornia.filters import spatial_gradient3d
from tensorboardX import SummaryWriter

from connected_module import ConnectedModule
from data_augmentation import Transforms
from dataset import SimpleRefineDataset


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


def mask_based_l1_loss(prediction, sdf, pca, refinement, mask, lr=0.1, rho=0.2):
    prediction_new = prediction * mask
    refinement_new = refinement * mask
    pca_new = pca * mask
    sdf_new = sdf * mask
    gt_new = sdf * mask
    sdf_new[sdf_new < 0] = 0
    sdf_new[sdf_new > 0] = 1
    sdf_new = 1 - sdf_new
    pca_new[pca_new < 0] = 0
    pca_new[pca_new > 0] = 1
    pca_new = 1 - pca_new
    residual = torch.abs(sdf_new - pca_new)
    residual_new = torch.zeros_like(residual)
    residual_new[residual != 0] = 1
    prediction_new[residual_new == 1] *= 100.
    gt_new[residual_new == 1] *= 100.
    loss = nn.functional.l1_loss(prediction_new, gt_new, reduction='sum')
    sum_mask = mask.sum().item()
    loss = loss / sum_mask
    # refinement_new_select = refinement_new.flatten()
    # refinement_new_select = refinement_new_select[refinement_new_select > rho]
    # refinement_new_select = refinement_new_select - rho
    # refinement_new_select = torch.sum(refinement_new_select)
    # if refinement_new_select.nelement() != 0:
    #     loss += lr * (refinement_new_select/sum_mask)
    #
    # refinement_new_select_2 = refinement_new.flatten()
    # refinement_new_select_2 = refinement_new_select_2[refinement_new_select_2 < -rho]
    # if refinement_new_select_2.nelement() != 0:
    #     refinement_new_select_2 = torch.abs(refinement_new_select_2+rho)
    #     loss += lr * (torch.sum(refinement_new_select_2)/sum_mask)

    return loss


def dirac_delta(phi_pred, eps):
    mask = torch.abs(phi_pred) <= eps
    dirac_delta = (1 / (2 * eps)) * (1 + torch.cos((np.pi * phi_pred) / eps))
    if phi_pred.dtype == torch.float32:
        dirac_delta = dirac_delta * mask.float()
    elif phi_pred.dtype == torch.float64:
        dirac_delta = dirac_delta * mask.double()
    else:

        print('strange dtype in delta')
        exit()
    # dirac_delta[dirac_delta==0] = 1e-3 #bcz the loss goes to nan.
    return dirac_delta


def get_norm3d_batch_torch(list_of_grad_torch):
    #    set_trace()
    assert type(list_of_grad_torch) is list
    batch_size = len(list_of_grad_torch)
    grid_size = list_of_grad_torch[0][0].shape[0]
    norm_batch_torch = torch.zeros((batch_size, grid_size, grid_size, grid_size), dtype=list_of_grad_torch[0][0].dtype)
    for _, c_grad_t in enumerate(list_of_grad_torch):
        c_norm_t = get_norm3d_torch(c_grad_t)
        norm_batch_torch[_] = c_norm_t
    return norm_batch_torch


def get_norm3d_torch(gradients_torch, small_nr=1e-8):
    assert type(gradients_torch) is tuple
    assert len(gradients_torch) == 3
    norm_torch \
        = gradients_torch[0] ** 2 + gradients_torch[1] ** 2 + gradients_torch[2] ** 2
    norm_torch = torch.sqrt(norm_torch + small_nr)
    return norm_torch


def implicit_loss(prediction, target, mask, refinement, eps=0.1, alpha2=1, p=2, rho=8.0, lr=.1):
    # import pdb;pdb.set_trace()
    # prediction_new = prediction * mask
    # target_new = target * mask
    # refinement_new = refinement * mask
    # sum_mask = mask.sum().item()

    delta = dirac_delta(prediction, eps=eps)
    chamf_loss = (delta * torch.square(target))  # BS X side**3
    chamf_loss = chamf_loss.sum(dim=1) + 1e-4 # BS
    # print(chamf_loss.min())
    chamf_loss = torch.sqrt(chamf_loss[chamf_loss > 0])  # BS, do abs?
    chamf_loss = chamf_loss.mean()
    # chamf_loss = chamf_loss

    gradient = spatial_gradient3d(prediction)
    gradient = gradient.div(4.0)  # divide by the resolution
    gradient_norm = torch.norm(gradient, dim=2)
    sdf_loss = (gradient_norm - 1) ** 2
    sdf_loss = sdf_loss.sum(dim=1)  # BS
    # sdf_loss = sdf_loss * mask

    sdf_loss = sdf_loss.mean()
    # sdf_loss = sdf_loss / sum_mask

    loss = chamf_loss + sdf_loss

    refinement_new_select = refinement.flatten()
    refinement_new_select = refinement_new_select[refinement_new_select > rho]
    refinement_new_select = refinement_new_select - rho
    refinement_new_select = torch.sum(refinement_new_select)
    if refinement_new_select.nelement() != 0:
        loss += lr * (refinement_new_select)

    refinement_new_select_2 = refinement.flatten()
    refinement_new_select_2 = refinement_new_select_2[refinement_new_select_2 < -rho]
    if refinement_new_select_2.nelement() != 0:
        refinement_new_select_2 = torch.abs(refinement_new_select_2+rho)
        refinement_new_select_2 = torch.sum(refinement_new_select_2)
        loss += lr* (refinement_new_select_2)
    return loss


def train(model, loader, optimizer, tfboard, epoch):
    model.train()
    lr = optimizer.param_groups[0]['lr']
    length = len(loader)
    avg_loss = []
    for iter_idx, data_dict in enumerate(loader):
        optimizer.zero_grad()
        data_dict['im'] = data_dict['im'].cuda(non_blocking=True)
        data_dict['sdf'] = data_dict['sdf'].cuda(non_blocking=True)
        # data_dict['mask'] = data_dict['mask'].cuda()

        data_dict = model(data_dict)
        # loss = mask_based_l1_loss(data_dict['output'], data_dict['sdf'], data_dict['pca'], data_dict['r'],
        #                           data_dict['mask'])
        loss = implicit_loss(data_dict['output'], data_dict['sdf'], None, data_dict['r'], eps=0.1, alpha2=1, p=2, rho=0.2)
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        torch.cuda.empty_cache()
        print('Epoch: {} Iter: {}/{} LR: {} Loss: {}'.format(epoch, iter_idx, length, lr, loss))
    avg_loss = np.mean(avg_loss)
    tfboard.add_scalar('train/loss', avg_loss, epoch)


def val(model, loader, tfboard, epoch):
    model.eval()
    length = len(loader)
    avg_loss = []
    with torch.no_grad():
        for iter_idx, data_dict in enumerate(loader):
            data_dict['im'] = data_dict['im'].cuda(non_blocking=True)
            data_dict['sdf'] = data_dict['sdf'].cuda(non_blocking=True)
            # data_dict['mask'] = data_dict['mask'].cuda()
            data_dict = model(data_dict)
            #loss = mask_based_l1_loss(data_dict['output'], data_dict['sdf'], data_dict['pca'], data_dict['r'],
            #                          data_dict['mask'])
            loss = implicit_loss(data_dict['output'], data_dict['sdf'], None, data_dict['r'], eps=0.1, alpha2=1, p=2, rho=0.2)

            avg_loss.append(loss.item())
            print('Epoch: {} Iter: {}/{} Loss: {}'.format(epoch, iter_idx, length, loss))
    avg_loss = np.mean(avg_loss)
    tfboard.add_scalar('val/loss', avg_loss, epoch)
    return avg_loss


def main(args):
    # trans_center = [-63.5, -63.5, -100.5]

    # torch.autograd.set_detect_anomaly(True)
    torch.multiprocessing.set_start_method('spawn')
    torch.manual_seed(0)
    ###############################################################################################
    ###### Step 0. Environment Setup
    ###############################################################################################

    # Step 0.1 Read Configuration Document
    config = read_yaml(args.yaml_file)

    lr = config.solver.lr

    # here we only have two settings so far, translation and translation+scale
    study_name = config.study.name

    # create ckpt and log dirs
    workspace = os.path.join(args.save_path, 'runs')
    init_workspace(workspace)

    ckpt_path = os.path.join(args.save_path, 'ckpts')
    os.makedirs(ckpt_path, exist_ok=True)

    ###############################################################################################
    ###### Step 1. Data Preparation
    ###############################################################################################
    im_dir = args.im_root
    # load the json list
    list_file = args.json_list
    with open(os.path.join(list_file, 'train_list_resize_4_4_4_crop_rts.json'), 'r') as f:
        train_json_list = json.load(f)
    # train_json_list = [train_json_list[43]]
    with open(os.path.join(list_file, 'val_list_resize_4_4_4_crop_rts.json'), 'r') as f:
        val_json_list = json.load(f)
    transforms = Transforms(im_dir=im_dir)
    train_dataset = SimpleRefineDataset(train_json_list,
                                        im_root=args.im_root,
                                        sdf_root=args.sdf_root,
                                        pca_root=args.pca_root,
                                        transforms=transforms.train_step_0_transforms)
    val_dataset = SimpleRefineDataset(val_json_list,
                                      im_root=args.im_root,
                                      sdf_root=args.sdf_root,
                                      pca_root=args.pca_root,
                                      transforms=transforms.val_step_0_transforms)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.solver.batch_size,
        shuffle=True,
        num_workers=config.solver.num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.solver.batch_size,
        shuffle=False,
        num_workers=config.solver.num_workers,
    )
    context_model = ConnectedModule(n_channels=2, n_classes=1)
    context_model = torch.nn.DataParallel(context_model).cuda()
    optimizer = torch.optim.Adam(context_model.module.parameters(), lr=lr, weight_decay=1e-4)
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
        context_model.load_state_dict(loaded['state_dict'])

    lr_step = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='min', patience=20)
    tfboard = SummaryWriter(logdir=runs)
    for cur_epoch in range(start_epoch, 20000):

        train(context_model, train_dataloader, optimizer, tfboard, cur_epoch)
        print(args.save_path)
        if val_freq and cur_epoch % val_freq == 0:
            loss = val(context_model, val_dataloader, tfboard, cur_epoch)
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
    parser.add_argument('--json_list', type=str, help='path to json list file')
    parser.add_argument('--yaml_file', type=str, help="path to yaml file for configuration",
                        default='hyper-parameters.yml')

    parser.add_argument('--resume', type=str, default=None, help="path to saved shape embedding model")

    parser.add_argument('--checkpoint', type=str, default="", help="path to saved total model")

    parser.add_argument('--pca_root', type=str, help="path to config file for embedding model ")
    parser.add_argument('--sdf_root', type=str, help="path to config file for embedding model ")

    parser.add_argument('--save_path', type=str, help="path to where you want checkpoints and logs save to",
                        default='./')
    args = parser.parse_args()

    main(args)
