import os
import json
import math
import argparse
import torch

from dlt_utils import (read_yaml, init_workspace,
                       Context, MoveToCuda, PreIterTransform, Callable,
                       TQDMConsoleLogger, OnlineAverageLoss, SupervisedTrainer,
                       Checkpointer, OptimScheduler)
from dataset import SDFSamples
from networks.deep_sdf_decoder import Decoder

def run_train(args):
    ###############################################################################################
    ###### Step 0. Environment Setup
    ###############################################################################################

    # Step 0.1 Read Configuration Document
    config = read_yaml(args.yaml_file)

    # Step 0.2 Setup the context variable
    context = Context()


    ###############################################################################################
    ###### Step 1. Data Preparation
    ###############################################################################################
    im_dir = args.im_root
    list_file = os.path.join(im_dir, 'json_list.json')
    with open(list_file ,'r') as f:
        json_list = json.load(f)


    # we use a special dataset specifically for SDF samples
    dataset = SDFSamples(json_list, config.data.subsample, args.im_root)

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = config.solver.batch_size, 
        shuffle = True, 
        num_workers = config.solver.num_workers,
        pin_memory = True,
        )

    context['component.train_loader'] = dataloader

    # number of shapes to embed
    num_scenes = len(dataset)
    latent_size = config.solver.latent_size
    # magnitude of latent vector for initialization
    code_bound = config.solver.code_bound
    code_std_dev = config.solver.code_std_dev
    if code_bound == -1:
        code_bound = None

    # initialize the latent vectors with normal distribution
    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        code_std_dev / math.sqrt(latent_size),
    )
    # store the latent space as a `global` variable in the context
    context['component.latent_vector'] = lat_vecs



    # after being loaded, the batch must be prepared
    def prepare_batch(context):

        # get the batch data from the context
        batch_data = context['var.batch_data']
        # get the set of shape indices for each samples
        indices = batch_data['idx']
        # fetch the latent vectors
        lat_vecs = context['component.latent_vector']
        # fetch the set of point/SDF pairs
        samples = batch_data['samples']
        # collect the latent vector corresponding to each scene
        cur_lat_vecs = lat_vecs(indices)
        latent_size = cur_lat_vecs.shape[1]
        num_scenes = samples.shape[1]
        cur_lat_vecs = cur_lat_vecs.unsqueeze(1)
        # now for each sample in each scene, repeat the latent vector
        cur_lat_vecs = cur_lat_vecs.repeat(1, num_scenes, 1)
        cur_lat_vecs = cur_lat_vecs.reshape(-1, latent_size)
        batch_data['cur_latent_vecs'] = cur_lat_vecs
        # extract the gt SDF and point coordinates from the samples
        gt = samples[:,:,0]
        gt = gt.reshape(-1, 1)
        samples = samples[:,:,1:]
        samples = samples.reshape(-1, 3)
        # concatenate the repeated latent vectors with the point coordinates
        input_samples = torch.cat((cur_lat_vecs, samples), 1)
        batch_data['samples'] = input_samples
        # store the gt as a seperate filed in the batch_dict
        batch_data['gt'] = gt


    # set up the batch preperar to run before every iteration
    batch_preparer = PreIterTransform(
            prepare_batch
            )
    context['component.batch_preparer'] = batch_preparer
            

    # also set up moving samples and gt variables to cuda 
    cuda_mover = PreIterTransform(
        MoveToCuda(
            fields=['var.batch_data.samples', 'var.batch_data.gt']
        )
    )
    context['component.cuda_mover'] = cuda_mover

    # make sure the batch_preparer runs before the cuda_mover
    cuda_mover.add_dependency('callback_pre_train_iter', batch_preparer)
    cuda_mover.add_dependency('callback_pre_val_iter', batch_preparer)

    #******
    # NOTE: DeepSDF clamps the SDF to be between -.1 to .1. I think this should not be done for SSMs, so this is 
    # commented out
    #*****
    # clamp_dist = config.solver.clampingdistance
    # minT = -clamp_dist
    # maxT = clamp_dist

    # def clamp(context):
        # context['var.batch_data.gt'] = torch.clamp(context['var.batch_data.gt'], minT,  maxT)
        

    # clamper = PreIterTransform(
            # clamp)

    # context['component.clamper'] = clamper

    # clamper.add_dependency('callback_pre_train_iter', cuda_mover)

    #*****

    # set up the DeepSDF model based on the configuration parameters
    model_layers_dims = config.model.dims
    model_dropout = config.model.dropout
    norm_layers = config.model.norm_layers
    dropout_prob = config.model.dropout_prob
    weight_norm = config.model.weight_norm
    latent_in = config.model.latent_in
    xyz_in_all = config.model.xyz_in_all
    use_tanh = config.model.use_tanh
    latent_dropout = config.model.latent_dropout
    model = Decoder(latent_size, model_layers_dims, dropout=model_dropout, norm_layers=norm_layers, 
            weight_norm=weight_norm, dropout_prob=dropout_prob, latent_in=latent_in, xyz_in_all=xyz_in_all,
            use_tanh=use_tanh, latent_dropout=latent_dropout)

    print(model)
    
    model = torch.nn.DataParallel(model).cuda()

    context['component.model'] = model

   
    # set up an L1Loss, and also regularize the latent space magnitudes

    code_reg_lambda = config.solver.code_reg_lambda
    loss_l1 = torch.nn.L1Loss(reduction="mean")
    def embed_loss(batch_data):
        output = batch_data['output']
        gt = batch_data['gt']
        loss = loss_l1(output, gt)
        cur_lat_vecs = batch_data['cur_latent_vecs']

        l2_size_loss = torch.mean(torch.pow(cur_lat_vecs, 2)) * code_reg_lambda
        loss += l2_size_loss 
        return loss


    context['component.loss'] = embed_loss


    
    # we use two LRs, one for latent space and one for the model, as recommended by DeepSDF
    lr_latent = config.solver.latent_lr
    lr_model = config.solver.model_lr
    optimizer_all = torch.optim.Adam(
        [
            {
                "params": model.parameters(),
                "lr": lr_model,
            },
            {
                "params": lat_vecs.parameters(),
                "lr":  lr_latent
            },
        ]
    )

    context['component.optimizer'] = optimizer_all

    # after a 1000 iterations, we step down the LR
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_all, 1000)

    optim_scheduler = OptimScheduler(scheduler)

    context['component.scheduler'] = optim_scheduler


    # set up a monitor for the average loss
    context['component.metric.avg_loss'] = OnlineAverageLoss()


    # Create a console monitor to print the average loss
    console_monitor = TQDMConsoleLogger(
        extra_train_logs = ('var.avg_train_loss', 'Avg. Train Loss %.5f')
        )
    context['component.monitor.console'] = console_monitor

    # set up a model saver that saves the latest model
    ckpt_path = os.path.join(args.save_path, 'ckpts')
    os.makedirs(ckpt_path, exist_ok = True)
    model_saver = Checkpointer(
            ckpt_path,
            keep_all_ckpts=False
            )

    context['component.model_saver'] = model_saver



    # start training. we set val_freq to 0, b/c we have no validation set
    trainer = SupervisedTrainer(
        context = context, 
        val_freq = 0,
        max_epoch = config.solver.get('epochs'),
        )
    trainer.run()
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_root', type=str, help="root directory of images")
    parser.add_argument('--yaml_file', type=str, help="path to yaml file for configuration", default='hyper-parameters.yml')
    parser.add_argument('--save_path', type=str, help="path to where you want checkpoints and logs save to", default='./')
    args = parser.parse_args()


    run_train(args)
