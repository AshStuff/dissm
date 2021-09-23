import os
import numpy as np
import json
import math
import argparse
import torch
import torchvision
import torch.nn as nn

from dlt.common.utils.setup import read_yaml, init_workspace, load_model_from_ckpt
from dlt.common.core import Context
from dlt.common.transforms import MoveToCuda, PreIterTransform, Callable
from dlt.common.monitors import TQDMConsoleLogger, TensorboardLogger
from dlt.common.metrics import OnlineAverageLoss
from dlt.common.layers import MetricWrapper
from dlt.common.trainers import SupervisedTrainer
from dlt.common.controllers import Checkpointer, OptimScheduler, ModelSaver
from dlt.common.transforms import NiBabelLoader, ExpandDims, Clip
from dlt.common.datasets import BaseDataset
from networks.encoder import WrapperResidualEncoder
from networks.deep_sdf_decoder import create_decoder_model
from dataset import SDFSamples

def run_train(args):
    ###############################################################################################
    ###### Step 0. Environment Setup
    ###############################################################################################

    # Step 0.1 Read Configuration Document
    config = read_yaml(args.yaml_file)

    lr = config.solver.lr
    # Step 0.2 Setup the context variable
    context = Context()
    study_name = config.study.name
    study_name += '_' + str(lr)
    workspace = os.path.join(args.save_path, 'runs', study_name)
    init_workspace(workspace)

    ckpt_path = os.path.join(args.save_path, 'ckpts', study_name)
    os.makedirs(ckpt_path, exist_ok=True)



    

    # here we arbitrarily set the last 13 as validation
    # TODO: make a randomized val and train json
    im_dir = args.im_root
    list_file = args.json_list
    with open(list_file ,'r') as f:
        json_list = json.load(f)

    train_json_list = json_list[:-13]

    val_json_list = json_list[-13:]
   

    def copy_path(data_dict):
        data_dict['copy_path'] = data_dict['im']
        return data_dict



    def prepare_gt(data_dict):
        # collect ground truth loaded from the json
        json_trans = data_dict['t']
        json_scale = data_dict['s']
        trans = np.asarray(json_trans)
        data_dict['t'] = trans
        return data_dict




    # TODO : set up data augmentation 
    transforms = [copy_path, prepare_gt, NiBabelLoader(fields='im', root_dir=im_dir), ExpandDims(fields='im', axis=0), Clip(fields='im', new_min=-160, new_max=240)]
    transforms = torchvision.transforms.Compose(transforms)
    train_dataset = BaseDataset(train_json_list,  transforms=transforms)
    val_dataset = BaseDataset(val_json_list,   transforms=transforms)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = config.solver.batch_size, 
        shuffle = True, 
        num_workers = config.solver.num_workers,
        pin_memory = True,
        )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size = config.solver.batch_size, 
        shuffle = False, 
        num_workers = config.solver.num_workers,
        pin_memory = True,
        )
    context['component.train_loader'] = train_dataloader
    context['component.val_loader'] = val_dataloader



                    

    # Step 1.4 Move data to cuda
    cuda_mover = PreIterTransform(
        MoveToCuda(
            fields=['var.batch_data.im', 'var.batch_data.t', 'var.batch_data.s']
        )
    )
    context['component.cuda_mover'] = cuda_mover




    class ConnectedModel(nn.Module):
        def __init__(self, num_outputs):

            super().__init__()
            self.encoder = WrapperResidualEncoder(num_outputs)

        def forward(self, data_dict):
            data_dict = self.encoder(data_dict)

           

            trans = data_dict['theta'][:,:3]
            # roughly the mean location is here, so we predict the deviation from it (should  not be hardcoded in the future)
            trans[:,0] += 155
            trans[:,1] += 140
            trans[:,2] += 80
            data_dict['predict_trans'] = trans

            scale = data_dict['theta'][:,3]

            # similiarly we predict deviations from the mean scale

            data_dict['predict_scale'] = 60 + scale


            

            # data_dict = self.decoder(data_dict)
            return data_dict


            
        
    # model = WrapperModel().cuda()
    context_model = ConnectedModel(4).cuda()

    if args.checkpoint != "":
        load_model_from_ckpt(context_model, args.checkpoint)

    context_model = torch.nn.DataParallel(context_model)

    context['component.model'] = context_model

    loss_l1 = torch.nn.L1Loss(reduction="mean")


    # compute L1Loss across translation and scale
    def direct_loss(data_dict):
        diff = data_dict['t'] - data_dict['predict_trans']
        batch_size = diff.shape[0]
        print('gt_trans', data_dict['t'].detach().cpu().numpy())
        print('predict_trans', data_dict['predict_trans'].detach().cpu().numpy())

        print('gt_scale', data_dict['s'].detach().cpu().numpy())
        print('predict_scale', data_dict['predict_scale'].detach().cpu().numpy())

        # penalize translations in z direction more heavily due to sampling differences
        # TODO: un hardcode this!
        diff[:,2] *= 5/2
        diff = torch.abs(diff)
        diff = torch.sum(diff)
        diff_scale = data_dict['s'] - data_dict['predict_scale']
        diff_scale = torch.abs(diff_scale)
        diff_scale = torch.sum(diff_scale)
        loss = (diff + diff_scale) / (4 * batch_size)
        return loss

    context['component.loss'] = direct_loss

    weight_decay = config.solver.weight_decay 
    lr = config.solver.lr

    optimizer = torch.optim.Adam(context_model.module.encoder.parameters(), 
            lr=lr,
            weight_decay=weight_decay)
    

    context['component.optimizer'] = optimizer



    # Step 3.2 Loss metric
    context['component.metric.avg_loss'] = OnlineAverageLoss()



    # Step 5.0 Create a concole monitor
    console_monitor = TQDMConsoleLogger(
        extra_train_logs = ('var.avg_train_loss', 'Avg. Train Loss %.5f'),
        extra_val_logs = ('var.avg_val_loss', 'Avg. Val Loss %.5f')
        )
    context['component.monitor.console'] = console_monitor

    tensorboard_monitor = TensorboardLogger(workspace)
    tensorboard_monitor.add_to_post_val_epoch('var.avg_val_loss', 'Val Loss')
    tensorboard_monitor.add_to_post_train_epoch('var.avg_train_loss', 'Train Loss')

    tensorboard_monitor.add_dependency('callback_post_val_epoch', 'component.metric.avg_loss')
    tensorboard_monitor.add_dependency('callback_post_train_epoch', 'component.metric.avg_loss')
    context['component.monitor.tb'] = tensorboard_monitor


    os.makedirs(ckpt_path, exist_ok = True)
    model_saver = ModelSaver(ckpt_path,
            metric_field='var.avg_val_loss',
            lower_is_better=True
            )

    context['component.model_saver'] = model_saver
    model_saver.add_dependency('callback_post_val_epoch', 'component.metric.avg_loss')

    ###############################################################################################
    ###### Step 7. Start training 
    ###############################################################################################
    trainer = SupervisedTrainer(
        context = context, 
        val_freq = 1,
        max_epoch = config.solver.get('epochs'),
        )
    trainer.run()
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_root', type=str, help="root directory of images")
    parser.add_argument('--json_list', type=str, help='path to json list file')
    parser.add_argument('--yaml_file', type=str, help="path to yaml file for configuration", default='hyper-parameters.yml')

    parser.add_argument('--checkpoint', type=str, default="", help="path to saved total model")
    parser.add_argument('--do_scale', action='store_true', help="whether to predict scale as well ")
    parser.add_argument('--save_path', type=str, help="path to where you want checkpoints and logs save to", default='./')
    args = parser.parse_args()


    run_train(args)
