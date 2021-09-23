import argparse
import json
import os

import nibabel as nib
import numpy as np
import torch
import torchvision
from dlt.common.controllers import Checkpointer, RollingAverageModelSaver
from dlt.common.core import Context
from dlt.common.layers import MetricWrapper
from dlt.common.metrics import OnlineAverageLoss
from dlt.common.monitors import TQDMConsoleLogger, TensorboardLogger
from dlt.common.trainers import SupervisedTrainer
from dlt.common.transforms import MoveToCuda, PreIterTransform
from dlt.common.transforms import NiBabelLoader, ExpandDims, Clip, CenterIntensities
from dlt.common.utils.setup import read_yaml, init_workspace, load_model_from_ckpt
from monai.transforms import RandGaussianNoised, RandAdjustContrastd, RandShiftIntensityd
from networks.connected_module import ConnectedModel

from dataset import SDFSamples, CustomRandomAffine, CopyField, ApplyAffineToPoints, AddMaskChannel, CreateAffinePath, \
    CreateScaleInitialGuess


def run_train(args):
    # torch.multiprocessing.set_start_method('spawn')
    torch.manual_seed(0)
    ###############################################################################################
    ###### Step 0. Environment Setup
    ###############################################################################################

    # Step 0.1 Read Configuration Document
    config = read_yaml(args.yaml_file)

    lr = config.solver.lr
    context = Context()

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

    context['var.mean_shape_vector'] = mean_latent_vec.cuda()

    # grab the config used for the shape embedding model
    embed_config = read_yaml(args.embed_yaml_file)

    ###############################################################################################
    ###### Step 1. Data Preparation
    ###############################################################################################
    im_dir = args.im_root
    # load the json list
    list_file = args.json_list
    with open(list_file, 'r') as f:
        json_list = json.load(f)
    train_json_list = json_list[:-13]

    val_json_list = json_list[-13:]

    # grab the mean shape sdf
    mean_ni = nib.load(args.mean_sdf_file)
    mean_np = mean_ni.get_fdata()

    latent_size = embed_config.solver.latent_size

    # create data transforms 
    train_transforms = [CopyField(source_key='im', dest_key='copy_path'), NiBabelLoader(fields='im', root_dir=im_dir)]

    val_transforms = [CopyField(source_key='im', dest_key='copy_path'), NiBabelLoader(fields='im', root_dir=im_dir)]

    # add intensity transforms to training
    train_transforms.append(RandShiftIntensityd(keys='im', offsets=10))
    train_transforms.append(RandAdjustContrastd(keys='im'))
    train_transforms.append(RandGaussianNoised(keys='im', std=5))

    # window the ct using liver window
    train_transforms.append(ExpandDims(fields='im', axis=0))
    train_transforms.append(Clip(fields='im', new_min=-160, new_max=240))
    train_transforms.append(CenterIntensities(fields='im', subtrahend=40, divisor=200))

    if args.do_scale:
        gt_trans_key = 'predict_trans'
    else:
        gt_trans_key = 't'

    # add random affine transform to image
    train_transforms.append(
        CustomRandomAffine(
            keys=('im'),
            mode=("bilinear"),
            prob=0.5,
            translate_range=(10, 10, 10),
            rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
            scale_range=(0.1, 0.1, 0.1),
            padding_mode="border"
            # device=torch.device('cuda:0')
        )
    )

    # add a transform to update the point coordinates with the image affine transsform
    train_transforms.append(
        ApplyAffineToPoints(gt_trans_key=gt_trans_key)
    )

    # create affine from pixel to canonical space
    Apx2sdf = np.eye(4)
    Apx2sdf[0, 3] = -127.5
    Apx2sdf[1, 3] = -127.5
    Apx2sdf[2, 3] = -80.5
    Apx2sdf[0, :] *= -1 / 63.5
    Apx2sdf[1, :] *= -1 / 63.5
    Apx2sdf[2, :] *= 1 / 63.5 * (5 / 2)

    # affine from sdf to pixel space
    Asdf2px = np.linalg.inv(Apx2sdf)
    # let's create a global affine transform that describes the initial coordinates of the mean mask/sdf
    # Here, this is hardcoded, which should be changed 
    global_init_mtx = np.eye(4)
    global_init_mtx[0, 3] = 148.5
    global_init_mtx[1, 3] = 132.8
    global_init_mtx[2, 3] = 93.2
    # account for the initial location at the center of the image
    global_init_mtx[:3, 3] -= Asdf2px[:3, 3]

    # create a centering transform to allow rotations to be properly applied
    centering = -(np.asarray(mean_np.shape) - 1) / 2
    center_affine = np.eye(4)
    center_affine[:3, 3] = centering

    do_scale = args.do_scale
    if do_scale:
        train_transforms.append(CreateScaleInitialGuess(Asdf2px=Asdf2px, gt_trans_key=gt_trans_key))

        # add a transform that randomly jitters the starting coordinates of the mean mesh/sdf and adds it as
        # an extra channel to the image
        # this transform also updates the starting coordinates to be refined
        train_transforms.append(
            AddMaskChannel(
                mean_np=mean_np,
                global_init_mtx=None,
                init_affine_key='cur_starting_aff',
                translate_range=(5, 5, 2.5),
                scale_range=(.1, .1, .1),
                # device=torch.device('cuda:0')
            )
        )

    else:
        train_transforms.append(
            CreateAffinePath(global_init_mtx=global_init_mtx, Asdf2px=Asdf2px, gt_trans_key=gt_trans_key))
        # add a transform that randomly jitters the starting coordinates of the mean mesh/sdf and adds it as
        # an extra channel to the image
        # this transform also updates the starting coordinates to be refined
        train_transforms.append(
            AddMaskChannel(
                mean_np=mean_np,
                global_init_mtx=global_init_mtx,
                init_affine_key='cur_starting_aff',
                translate_range=(10, 10, 10)
                # device=torch.device('cuda:0')
            )
        )

    val_transforms.append(ExpandDims(fields='im', axis=0))
    val_transforms.append(Clip(fields='im', new_min=-160, new_max=240))
    val_transforms.append(CenterIntensities(fields='im', subtrahend=40, divisor=200))
    if do_scale:
        val_transforms.append(CreateScaleInitialGuess(Asdf2px=Asdf2px, gt_trans_key=gt_trans_key))
        val_transforms.append(
            AddMaskChannel(
                mean_np=mean_np,
                global_init_mtx=None,
                init_affine_key='cur_starting_aff'
                # translate_range = (5.0, 5.0, 2.5)
                # device=torch.device('cuda:0')
            )
        )
    else:
        val_transforms.append(
            AddMaskChannel(
                mean_np=mean_np,
                global_init_mtx=global_init_mtx
                # device=torch.device('cuda:0')
            )
        )

    subsample = 150000
    train_transforms = torchvision.transforms.Compose(train_transforms)
    val_transforms = torchvision.transforms.Compose(val_transforms)
    train_dataset = SDFSamples(train_json_list, subsample, args.im_root, transforms=train_transforms)
    val_dataset = SDFSamples(val_json_list, subsample, args.im_root, transforms=val_transforms)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.solver.batch_size,
        shuffle=True,
        num_workers=config.solver.num_workers,
        pin_memory=False
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.solver.batch_size,
        shuffle=False,
        num_workers=config.solver.num_workers,
        pin_memory=True,
    )
    context['component.train_loader'] = train_dataloader
    context['component.val_loader'] = val_dataloader

    def prepare_batch(context):
        """
        
        """
        batch_data = context['var.batch_data']

        # grab the mean latent vector
        mean_latent_vec = context['var.mean_shape_vector']
        samples = batch_data['samples'].float()
        batch_size = samples.shape[0]
        num_scenes = samples.shape[1]

        # replicate the mean vector so we can concatenate it to every point for the embedding model
        mean_latent_vec = mean_latent_vec.repeat(batch_size, num_scenes, 1)
        mean_latent_vec = mean_latent_vec.reshape(-1, embed_config.solver.latent_size)

        batch_data['cur_latent_vecs'] = mean_latent_vec
        # extract the gt sdf val for each point
        gt = samples[:, :, 0]
        gt = gt.reshape(-1, 1)
        # reshape the points so we can cocatenate the mean vector to it
        samples = samples[:, :, 1:]
        samples = samples.reshape(-1, 3)

        # concatenate the mean vector 
        input_samples = torch.cat((mean_latent_vec, samples), 1)
        # update the batch dict
        batch_data['samples'] = input_samples
        batch_data['gt'] = gt

    # set up the function as a component step to perform before each iteration
    batch_preparer = PreIterTransform(
        prepare_batch
    )

    context['component.batch_preparer'] = batch_preparer

    # Step 1.4 Move data to cuda
    cuda_mover = PreIterTransform(
        MoveToCuda(
            fields=['var.batch_data.im', 'var.batch_data.samples']
        )
    )
    context['component.cuda_mover'] = cuda_mover

    # make sure the cuda mover executes after the prepare batch

    batch_preparer.add_dependency('callback_pre_train_iter', cuda_mover)
    batch_preparer.add_dependency('callback_pre_val_iter', cuda_mover)

    # instantiate a model that connects an image encoder with the shape decoder
    context_model = ConnectedModel(6, embed_config, args.embed_model_path, do_scale=args.do_scale,
                                   cur_affine_key='cur_affine_mtx',
                                   Apx2sdf=Apx2sdf, centering_affine=center_affine, gt_trans_key=gt_trans_key).cuda()

    if args.checkpoint != "":
        load_model_from_ckpt(context_model, args.checkpoint, partial=True)

    context_model = torch.nn.DataParallel(context_model)

    context['component.model'] = context_model

    margin = .1

    def hinge_loss(preds, gt):
        gt[gt < 0] = -1
        gt[gt > 0] = 1
        mult = margin - gt * preds
        mult[mult < 0] = 0
        return 10 * torch.mean(mult)

    # here as a mask-based loss, we treat the sdf predictions as a logit value, and threshold the gt sdfs
    # to {0,1}
    loss_ce_wrapper = MetricWrapper(label_field='gt', prediction_field='output',
                                    layer_instance=torch.nn.BCEWithLogitsLoss())

    def loss_ce_wrapper_gt(data_dict):
        gt = data_dict['gt']
        gt[gt < 0] = 0
        gt[gt > 0] = 1
        data_dict['gt'] = gt
        loss = loss_ce_wrapper(data_dict)
        return loss

    # alternatively, we may work with an L1 Loss
    loss_l1 = torch.nn.L1Loss()

    # APH we do mask-based loss for scale, not sure if this is the best way
    if args.do_scale:
        # loss_wrapper = loss_ce_wrapper_gt
        loss_wrapper = MetricWrapper(label_field='gt', prediction_field='output', layer_instance=loss_l1)
    else:
        loss_wrapper = MetricWrapper(label_field='gt', prediction_field='output', layer_instance=loss_l1)

    def combined_loss(data_dict):

        # if args.do_scale:
        # get the gt sdf values 
        gt = data_dict['gt']
        # weights = gt.clone()
        # # threshold them to {0,1}
        # gt[gt<0] = 0
        # gt[gt>0] = 1
        # data_dict['gt'] = gt

        # compute the ce loss
        loss = loss_wrapper(data_dict)
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
            print('scale_select', scale_select)
            if scale_select.nelement() != 0:
                loss += scale_select

            # penalized any scale values < 0.1
            scale_select_2 = scale.flatten()
            scale_select_2 = scale_select_2[scale_select_2 < 0.5]
            if scale_select_2.nelement() != 0:
                scale_select_2 = 0.5 - scale_select_2
                loss += torch.sum(scale_select_2)
            print('scale_select_2', scale_select_2)
        return loss

    context['component.loss'] = combined_loss

    # set up optimizer and regularizer
    weight_decay = config.solver.weight_decay
    lr = config.solver.lr

    # optimizer only optimizes the encoder parameters
    # optimizer = torch.optim.Adam(context_model.module.encoder.parameters(), 
    # lr=lr,
    # weight_decay=weight_decay)

    optim_params = context_model.module.encoder.parameters()
    if args.do_scale:
        extra_params = [context_model.module.alpha, context_model.module.beta]
        optim_params = list(optim_params) + extra_params
    # optimizer only optimizes the encoder parameters
    optimizer = torch.optim.AdamW(optim_params,
                                  lr=lr)

    context['component.optimizer'] = optimizer

    # Step 3.2 Loss metric
    context['component.metric.avg_loss'] = OnlineAverageLoss()

    # Step 5.0 Create a concole monitor
    console_monitor = TQDMConsoleLogger(
        extra_train_logs=('var.avg_train_loss', 'Avg. Train Loss %.5f'),
        extra_val_logs=('var.avg_val_loss', 'Avg. Val Loss %.5f')
    )
    context['component.monitor.console'] = console_monitor

    tensorboard_monitor = TensorboardLogger(workspace)
    tensorboard_monitor.add_to_post_val_epoch('var.avg_val_loss', 'Val Loss')
    tensorboard_monitor.add_to_post_train_epoch('var.avg_train_loss', 'Train Loss')

    tensorboard_monitor.add_dependency('callback_post_val_epoch', 'component.metric.avg_loss')
    tensorboard_monitor.add_dependency('callback_post_train_epoch', 'component.metric.avg_loss')
    context['component.monitor.tb'] = tensorboard_monitor

    os.makedirs(ckpt_path, exist_ok=True)
    model_saver = RollingAverageModelSaver(ckpt_path,
                                           metric_field='var.avg_val_loss',
                                           lower_is_better=True
                                           )

    model_ckpter = Checkpointer(
        ckpt_path,
        keep_all_ckpts=False
    )

    context['component.model_saver'] = model_saver
    context['component.model_cktper'] = model_ckpter
    model_saver.add_dependency('callback_post_val_epoch', 'component.metric.avg_loss')

    ###############################################################################################
    ###### Step 7. Start training 
    ###############################################################################################
    trainer = SupervisedTrainer(
        context=context,
        val_freq=1,
        max_epoch=config.solver.get('epochs'),
    )
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_root', type=str, help="root directory of images")
    parser.add_argument('--json_list', type=str, help='path to json list file')
    parser.add_argument('--yaml_file', type=str, help="path to yaml file for configuration",
                        default='hyper-parameters.yml')
    parser.add_argument('--embed_model_path', type=str, help="path to saved shape embedding model")

    parser.add_argument('--mean_sdf_file', type=str, help="path to mean shape mask file")
    parser.add_argument('--checkpoint', type=str, default="", help="path to saved total model")
    parser.add_argument('--do_scale', action='store_true', help="whether to predict scale as well ")
    parser.add_argument('--embed_yaml_file', type=str, help="path to config file for embedding model ")
    parser.add_argument('--save_path', type=str, help="path to where you want checkpoints and logs save to",
                        default='./')
    args = parser.parse_args()

    run_train(args)
