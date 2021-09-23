import random
from typing import Optional, Sequence, Union

import nibabel as nib
import numpy as np
import torch
from batchgenerators.augmentations.color_augmentations import augment_brightness_additive
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_blur
from dlt.common.transforms import ExpandDims, Clip, CenterIntensities
from monai.transforms import RandAffined
from monai.transforms import RandGaussianNoised, RandShiftIntensityd
from monai.transforms import Resample
from monai.transforms.compose import MapTransform
from monai.transforms.compose import Transform
from monai.transforms.utils import create_rotate, create_translate, create_scale, create_grid
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    fall_back_tuple,
)
from torchvision.transforms import Compose

from dataset import ConcatImPCA, ComputeMask


class GaussianBlurTransform():
    def __init__(self, blur_sigma=(1, 5), keys=['image'], different_sigma_per_channel=True,
                 p_per_channel=1., p_per_sample=1.):
        self.blur_sigma = blur_sigma
        self.keys = keys
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample

    def __call__(self, data_dict):
        if np.random.uniform() < self.p_per_sample:
            for key in self.keys:
                data_dict[key] = augment_gaussian_blur(data_dict[key], self.blur_sigma,
                                                       self.different_sigma_per_channel,
                                                       self.p_per_channel)
        return data_dict


class GaussianBrightnessTransform():
    def __init__(self, mu, sigma, per_channel=True, keys=['image'], p_per_sample=1., p_per_channel=1.):
        self.p_per_sample = p_per_sample
        self.keys = keys
        self.mu = mu
        self.sigma = sigma
        self.p_per_channel = p_per_channel
        self.per_channel = per_channel

    def __call__(self, data_dict):
        if np.random.uniform() < self.p_per_sample:
            for key in self.keys:
                data_dict[key] = augment_brightness_additive(data_dict[key], self.mu, self.sigma, self.per_channel,
                                                             self.p_per_channel)
        return data_dict


class NiBabelLoader():
    def __init__(self, fields):
        self.fields = fields

    def __call__(self, data_dict):
        for field in self.fields:
            data_dict[field] = nib.load(data_dict[field]).get_fdata().astype(np.float32)

        return data_dict


class CustomRandAffine():
    def __init__(self, field, translate_range=None, scale_range=None, rotate_range=None):
        self.field = field
        self.translate_range = translate_range
        self.scale_range = scale_range
        self.rotate_range = rotate_range

    def __call__(self, data_dict):
        if np.random.uniform() > 0.5:
            pca = data_dict[self.field]
            translate_params = None
            scale_params = None
            rotate_params = None

            if self.translate_range:
                translate_params = [random.uniform(-f, f) for f in self.translate_range if f is not None]

            if self.scale_range:
                scale_params = [random.uniform(-f, f) + 1.0 for f in self.scale_range if f is not None]

            if self.rotate_range:
                rotate_params = [random.uniform(-f, f) for f in self.rotate_range if f is not None]

            rand_affine = np.eye(4)
            if rotate_params:
                rand_affine = rand_affine @ create_rotate(3, rotate_params)
            if translate_params:
                rand_affine = rand_affine @ create_translate(3, translate_params)
            if scale_params:
                rand_affine = rand_affine @ create_scale(3, scale_params)
            cur_affine_for_image = np.linalg.inv(rand_affine)

            # transform the image mask/sdf using the image-specific affine matrix
            affine_trans = Affine_w_Mtx(affine_mtx=torch.tensor(cur_affine_for_image), device=None,
                                        padding_mode='border')
            new_pca_np = torch.tensor(affine_trans(pca))
            new_pca_np = new_pca_np.float()
            data_dict[self.field] = new_pca_np.numpy()
        return data_dict


class ImageCustomRandAffine(MapTransform):
    def __init__(self, keys, mode, prob,
                 scale_range=None,
                 rotate_range=None,
                 padding_mode='border',
                 device=None):
        super().__init__(keys)
        self.rand_affine = RandAffined(keys, rotate_range=rotate_range,
                                       scale_range=scale_range, mode=mode, prob=prob,
                                       padding_mode=padding_mode,
                                       device=device)

    def __call__(self, data_dict):
        data_dict = self.rand_affine(data_dict)
        return data_dict

class PCAFix():
    def __call__(self, data_dict):
        pca = data_dict['pca']
        pca[pca == 100] = 1
        data_dict['pca'] = pca
        sdf = data_dict['sdf']
        sdf[sdf == 100] = 1
        data_dict['sdf'] = sdf
        return data_dict


class Transforms():
    def __init__(self, im_dir):
        self.im_dir = im_dir

    @property
    def train_step_0_transforms(self):
        tr = [
            NiBabelLoader(fields=['im', 'sdf', 'pca']),

            # NiBabelLoader(fields='sdf'),
            # NiBabelLoader(fields='pca'),
            RandShiftIntensityd(keys='im', offsets=10, prob=0.5),
            RandGaussianNoised(keys='im', std=5, prob=0.5),
            GaussianBlurTransform(blur_sigma=(1, 5),
                                  keys=['im'],
                                  different_sigma_per_channel=True,
                                  p_per_channel=0.5,
                                  p_per_sample=0.5),
            # PCAFix(), #ToDo: fix this later

            GaussianBrightnessTransform(mu=0.0,
                                        sigma=2,
                                        keys=['im'],
                                        p_per_sample=0.5),
            ExpandDims(fields=['im', 'pca', 'sdf'], axis=0),
            # CustomRandAffine(field='pca',
            #                  translate_range=(4, 4, 4),
            #                  scale_range=(.03, .03, .03),
            #                  rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36)
            #                  ),
            # ExpandDims(fields='pca', axis=0),
            # ExpandDims(fields='sdf', axis=0),
            Clip(fields='im', new_min=-160, new_max=240),
            CenterIntensities(fields='im', subtrahend=40, divisor=200),
            # ImageCustomRandAffine(keys=('im', 'pca', 'sdf'),
            #                       mode=("bilinear"),
            #                       prob=1.0,
            #                       rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
            #                       scale_range=(0.1, 0.1, 0.1),
            #                       padding_mode="border",
            #                       ),
            ComputeMask(margin=.25),

            ConcatImPCA(),


        ]
        return Compose(tr)

    @property
    def train_other_steps_transforms(self):
        tr = [
            #ComputeMask(margin=0.25),
            ConcatImPCA(),

        ]
        return Compose(tr)

    @property
    def val_step_0_transforms(self):
        vt = [
            NiBabelLoader(fields=['im', 'sdf', 'pca']),
            ExpandDims(fields=['im', 'pca', 'sdf'], axis=0),
            Clip(fields='im', new_min=-160, new_max=240),
            CenterIntensities(fields='im', subtrahend=40, divisor=200),
            ComputeMask(margin=.25),
            ConcatImPCA(),

        ]
        return Compose(vt)

    @property
    def val_other_steps_transforms(self):
        vt = [
            ConcatImPCA(),
            ComputeMask(margin=15),

        ]
        return Compose(vt)


# Adapted from MONAI
class Affine_w_Mtx(Transform):
    """
    Transform ``img`` given the affine parameters.
    APH: modifies MONAI Affine to accept an affine matrix directly
    """

    def __init__(
            self,
            affine_mtx,
            spatial_size: Optional[Union[Sequence[int], int]] = None,
            mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
            padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
            as_tensor_output: bool = False,
            device: Optional[torch.device] = None,
    ) -> None:
        """
        The affine transformations are applied in rotate, shear, translate, scale order.

        Args:
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if the components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device: device on which the tensor will be allocated.
        """
        self.affine_grid = AffineGrid_w_Mtx(
            affine_mtx,
            as_tensor_output=True,
            device=device,
        )
        self.resampler = Resample(as_tensor_output=as_tensor_output, device=device)
        self.spatial_size = spatial_size
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)

    def __call__(
            self,
            img: Union[np.ndarray, torch.Tensor],
            spatial_size: Optional[Union[Sequence[int], int]] = None,
            mode: Optional[Union[GridSampleMode, str]] = None,
            padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        sp_size = fall_back_tuple(spatial_size or self.spatial_size, img.shape[1:])
        grid = self.affine_grid(spatial_size=sp_size)
        return self.resampler(
            img=img, grid=grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode
        )


class AffineGrid_w_Mtx(Transform):
    """
    Affine transforms on the coordinates.
    APH: modifies MONAI's AffineGrid to accept an affine matrix directly

    Args:
        as_tensor_output: whether to output tensor instead of numpy array.
            defaults to True.
        device: device to store the output grid data.

    """

    def __init__(
            self,
            affine_mtx,
            as_tensor_output: bool = True,
            device: Optional[torch.device] = None,
    ) -> None:

        self.as_tensor_output = as_tensor_output
        self.device = device
        self.affine = affine_mtx

    def __call__(
            self, spatial_size: Optional[Sequence[int]] = None, grid: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.

        Raises:
            ValueError: When ``grid=None`` and ``spatial_size=None``. Incompatible values.

        """
        if grid is None:
            if spatial_size is not None:
                grid = create_grid(spatial_size)
            else:
                raise ValueError("Incompatible values: grid=None and spatial_size=None.")

        grid = torch.tensor(grid) if not torch.is_tensor(grid) else grid.detach().clone()
        if self.device:
            grid = grid.to(self.device)
            self.affine = self.affine.to(self.device)
        grid = (self.affine.float() @ grid.reshape((grid.shape[0], -1)).float()).reshape([-1] + list(grid.shape[1:]))
        if self.as_tensor_output:
            return grid
        return grid.cpu().numpy()
