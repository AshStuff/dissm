#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
# DIRECTLY COPIED FROM THE DEEPSDF REPO
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_decoder_model(config):
    model_layers_dims = config.model.dims
    model_dropout = config.model.dropout
    norm_layers = config.model.norm_layers
    dropout_prob = config.model.dropout_prob
    weight_norm = config.model.weight_norm
    latent_in = config.model.latent_in
    xyz_in_all = config.model.xyz_in_all
    use_tanh = config.model.use_tanh
    latent_dropout = config.model.latent_dropout

    model = Decoder(config.solver.latent_size, model_layers_dims, dropout=model_dropout, norm_layers=norm_layers,
                    weight_norm=weight_norm, dropout_prob=dropout_prob, latent_in=latent_in, xyz_in_all=xyz_in_all,
                    use_tanh=use_tanh, latent_dropout=latent_dropout)

    return model


class Decoder(nn.Module):
    def __init__(
            self,
            latent_size,
            dims,
            dropout=None,
            dropout_prob=0.0,
            norm_layers=(),
            latent_in=(),
            weight_norm=False,
            xyz_in_all=None,
            use_tanh=False,
            latent_dropout=False,
            sample_key='samples',
            output_key='output'
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        self.sample_key = sample_key
        self.output_key = output_key
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                    (not weight_norm)
                    and self.norm_layers is not None
                    and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, batch_dict):
        cur_input = batch_dict[self.sample_key]
        xyz = cur_input[:, -3:]

        if cur_input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = cur_input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = cur_input

        for layer in range(0, self.num_layers - 1):
            if layer == self.num_layers - 2:
                dummy = 0

            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, cur_input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                        self.norm_layers is not None
                        and layer in self.norm_layers
                        and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)
        batch_dict[self.output_key] = x

        if x.device.index == 0:
            if self.training:
                with torch.no_grad():
                    gt = batch_dict['gt']
                    print(torch.min(x[:30000]).cpu().item(), torch.min(batch_dict['gt']).item())
                    print(torch.max(x[:30000]).cpu().item(), torch.max(batch_dict['gt']).item())
                    print(torch.mean(torch.abs(x)).cpu().item(), torch.mean(gt).cpu().item())

        return batch_dict
