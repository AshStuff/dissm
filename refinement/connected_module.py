import torch

# from unet import U_Net
from unet3d import UNet3D


class ConnectedModule(torch.nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.unet = UNet3D(in_channels=2, out_channels=1, is_segmentation=False, final_sigmoid=False)

    def forward(self, data_dict):
        im = data_dict['im']
        output = self.unet(im)
        data_dict['r'] = output

        pca = data_dict['pca']

        output = output + pca

        data_dict['output'] = output
        return data_dict
