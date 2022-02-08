import numpy as np
import torch
from batchgenerators.augmentations.color_augmentations import augment_brightness_additive
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_blur
from dlt_utils import NiBabelLoader, ExpandDims, Clip, CenterIntensities, WorldToVoxelAffine
from monai.transforms import RandGaussianNoised, RandShiftIntensityd
from torchvision.transforms import Compose, RandomChoice

from dataset import CustomRandomAffine, CopyField, ApplyAffineToPoints, AddMaskChannel, ToNumpy
from im_utils import crop_to_bbox


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


class FilterPoints():
    def __init__(self, volume_size):
        self.volume_size = volume_size

    def __call__(self, data_dict):
        samples = data_dict['samples']
        points = samples
        indxs = torch.where((0 <= points[:, 1]) & (points[:, 1] <= self.volume_size[0])
                            & (0 <= points[:, 2]) & (points[:, 2] <= self.volume_size[1]) & (0 <= points[:, 3]) & (
                                    points[:, 3] <= self.volume_size[2]))
        points = points[indxs]
        data_dict['samples'] = points
        return data_dict


class CropVolume():
    def __init__(self, fields, bbox, crop_size=[96, 96, 96], value=-1024):
        self.fields = fields
        self.bbox = bbox
        self.crop_size = crop_size
        self.value = value

    def __call__(self, data_dict):
        bbox = data_dict[self.bbox]
        im = data_dict[self.fields]
        im_crop = crop_to_bbox(im.squeeze(0), bbox)
        if im_crop.__class__.__name__ == 'ndarray':
            im_crop = torch.from_numpy(im_crop)

        data_dict['im'] = im_crop.unsqueeze(0)

        return data_dict


class Transforms():
    def __init__(self, mean_np, Apx2sdf, gt_trans_key, im_dir, global_init_mtx=None):
        self.Apx2sdf = Apx2sdf
        self.gt_trans_key = gt_trans_key
        self.im_dir = im_dir
        self.mean_np = mean_np
        # affine from sdf to pixel space
        self.Asdf2px = np.linalg.inv(Apx2sdf)
        self.global_init_mtx = global_init_mtx

        # create a centering transform to allow rotations to be properly applied

    @property
    def train_step_0_transforms(self):
        tr = [
            ToNumpy(source_key='s'),
            CopyField(source_key='im', dest_key='copy_path'),
            NiBabelLoader(fields='im', root_dir=self.im_dir),
            CopyField(source_key='im', dest_key='orig_im'),
            RandShiftIntensityd(keys='im', offsets=10, prob=0.5),
            RandGaussianNoised(keys='im', std=5, prob=0.5),
            GaussianBrightnessTransform(mu=0.0,
                                        sigma=2,
                                        keys=['im'],
                                        p_per_sample=0.5),
            ExpandDims(fields='im', axis=0),
            Clip(fields='im', new_min=-160, new_max=240),
            CenterIntensities(fields='im', subtrahend=40, divisor=200),
            CustomRandomAffine(
                keys=('im'),
                mode=("bilinear"),
                prob=1.0,
                translate_range=(10, 10, 10),
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border",
                device=torch.device('cuda:0'),
                output_key='aug_affine_mtx'
            ),
            ApplyAffineToPoints(gt_trans_key=self.gt_trans_key, init_key=None,affine_key='aug_affine_mtx'),
            # for the first step, we only work off the global initial guess
            RandomChoice(
                [AddMaskChannel(
                    mean_np=self.mean_np,
                    init_affine_key=None,
                    global_init_mtx=self.global_init_mtx,
                    translate_range=(10, 10, 10),
                    device=torch.device('cuda:0')),
                AddMaskChannel(
                    mean_np=self.mean_np,
                    init_affine_key=None,
                    global_init_mtx=self.global_init_mtx,
                    translate_range=(1, 1, 1),
                    device=torch.device('cuda:0'))])
            # AddMaskChannel(
                # mean_np=self.mean_np,
            # )


        ]
        return Compose(tr)

    @property
    def train_other_steps_transforms(self):
        tr = [
            # no need to load the image, since it will be forwarded in subsequent steps
            ApplyAffineToPoints(gt_trans_key=self.gt_trans_key, init_key=None,affine_key='aug_affine_mtx'),
            ToNumpy(source_key='s'),
            # # subsequent steps we work off of the previous guess, which should be encoded in data_dict[init_mtx]
            RandomChoice(
                [AddMaskChannel(
                    mean_np=self.mean_np,
                    init_affine_key='init_mtx',
                    global_init_mtx=None,
                    translate_range=(10, 10, 10),
                    device=torch.device('cuda:0')
                    ),
                AddMaskChannel(
                    mean_np=self.mean_np,
                    init_affine_key='init_mtx',
                    global_init_mtx=None,
                    translate_range=(1, 1, 1),
                    device=torch.device('cuda:0')
                    )])

        ]
        return Compose(tr)

    @property
    def val_step_0_transforms(self):
        vt = [
            ToNumpy(source_key='s'),
            CopyField(source_key='im', dest_key='copy_path'),
            NiBabelLoader(fields='im', root_dir=self.im_dir),
            ExpandDims(fields='im', axis=0),
            Clip(fields='im', new_min=-160, new_max=240),
            CenterIntensities(fields='im', subtrahend=40, divisor=200),
            AddMaskChannel(
                mean_np=self.mean_np,
            )

        ]
        return Compose(vt)

    @property
    def val_other_steps_transforms(self):
        vt = [
            ToNumpy(source_key='s'),
            AddMaskChannel(
                mean_np=self.mean_np,
                global_init_mtx=None,
                init_affine_key='init_mtx'
            )

        ]
        return Compose(vt)


class ScaleTransforms():
    def __init__(self, mean_np, Apx2sdf, gt_trans_key, im_dir, centering_affine, global_init_mtx=None):
        self.Apx2sdf = Apx2sdf
        self.gt_trans_key = gt_trans_key
        self.im_dir = im_dir
        self.mean_np = mean_np
        # affine from sdf to pixel space
        self.Asdf2px = np.linalg.inv(Apx2sdf)
        # account for the initial location at the center of the image
        self.global_init_mtx = global_init_mtx
        # create a centering transform to allow rotations to be properly applied
        self.centering_affine = centering_affine
    @property
    def train_step_0_transforms(self):
        tr = [
            CopyField(source_key='im', dest_key='copy_path'),
            ToNumpy(source_key='s'),
            NiBabelLoader(fields='im', root_dir=self.im_dir),
            WorldToVoxelAffine('post_translate_world', 'im_meta', centering_affine=self.centering_affine),
            # Test(self.mean_np),
            RandShiftIntensityd(keys='im', offsets=10, prob=0.5),
            RandGaussianNoised(keys='im', std=5, prob=0.5),
            GaussianBlurTransform(blur_sigma=(1, 5),
                                  keys=['im'],
                                  different_sigma_per_channel=True,
                                  p_per_channel=0.5,
                                  p_per_sample=0.5),
            GaussianBrightnessTransform(mu=0.0,
                                        sigma=2,
                                        keys=['im'],
                                        p_per_sample=0.5),
            ExpandDims(fields='im', axis=0),
            Clip(fields='im', new_min=-160, new_max=240),
            CenterIntensities(fields='im', subtrahend=40, divisor=200),
            CustomRandomAffine(
                keys=('im'),
                mode=("bilinear"),
                prob=1.0,
                translate_range=(10, 10, 10),
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border",
                device=torch.device('cuda:0'),
                output_key='aug_affine_mtx'

            ),
            ApplyAffineToPoints(gt_trans_key=self.gt_trans_key, affine_key='aug_affine_mtx'),
            # CropVolume(fields='im', bbox='bbox'),
            RandomChoice([
                AddMaskChannel(
                    mean_np=self.mean_np,
                    global_init_mtx=None,
                    init_affine_key='post_translate_world',
                    affine_mtx_key='aug_affine_mtx',
                    translate_range=(1, 1, 1),
                    scale_range=(.1, .1, .1),
                    # crop_sdf=(96, 96, 96),
                    device=torch.device('cuda:0')
                ),
                AddMaskChannel(
                    mean_np=self.mean_np,
                    global_init_mtx=None,
                    init_affine_key='post_translate_world',
                    affine_mtx_key='aug_affine_mtx',
                    translate_range=(1, 1, 1),
                    scale_range=(.02, .02, .02),
                    # crop_sdf=(96, 96, 96),

                    device=torch.device('cuda:0')
                ),
            ]),

        ]
        return Compose(tr)

    @property
    def train_other_steps_transforms(self):
        tr = [
            ToNumpy(source_key='s'),
            ApplyAffineToPoints(gt_trans_key=None, affine_key='aug_affine_mtx', init_key=None),
            RandomChoice([AddMaskChannel(
                mean_np=self.mean_np,
                global_init_mtx=None,
                translate_range=(1, 1, 1),
                # crop_sdf=(96, 96, 96),
                scale_range=(.1, .1, .1),
                init_affine_key='init_mtx',
                device=torch.device('cuda:0')
            ),
                AddMaskChannel(
                    mean_np=self.mean_np,
                    global_init_mtx=None,
                    init_affine_key='init_mtx',
                    translate_range=(1, 1, 1),
                    # crop_sdf=(96, 96, 96),
                    scale_range=(.02, .02, .02),
                    device=torch.device('cuda:0')
                )
            ]),
            # FilterPoints(volume_size=[120, 118, 60])
        ]
        return Compose(tr)

    @property
    def val_step_0_transforms(self):
        vt = [
            ToNumpy(source_key='s'),
            CopyField(source_key='im', dest_key='copy_path'),
            ToNumpy(source_key='s'),
            NiBabelLoader(fields='im', root_dir=self.im_dir),
            WorldToVoxelAffine('post_translate_world', 'im_meta', centering_affine=self.centering_affine),
            ExpandDims(fields='im', axis=0),
            Clip(fields='im', new_min=-160, new_max=240),
            CenterIntensities(fields='im', subtrahend=40, divisor=200),
            #CropVolume(fields='im', bbox='bbox'),
            AddMaskChannel(
                mean_np=self.mean_np,
                global_init_mtx=None,
                init_affine_key='post_translate_world',
                # crop_sdf=(96, 96, 96),
                device=torch.device('cuda:0')
            )

        ]
        return Compose(vt)

    @property
    def val_other_steps_transforms(self):
        vt = [
            ToNumpy(source_key='s'),
            # CropVolume(fields='im', bbox='bbox'),
            AddMaskChannel(
                mean_np=self.mean_np,
                global_init_mtx=None,
                # crop_sdf=(96, 96, 96),
                init_affine_key='init_mtx',

                device=torch.device('cuda:0')
            )

        ]
        return Compose(vt)


class RotateTransforms():
    def __init__(self, mean_np, Apx2sdf, gt_trans_key, im_dir, global_init_mtx):
        self.Apx2sdf = Apx2sdf
        self.gt_trans_key = gt_trans_key
        self.im_dir = im_dir
        self.mean_np = mean_np
        # affine from sdf to pixel space
        self.Asdf2px = np.linalg.inv(Apx2sdf)
        # account for the initial location at the center of the image
        global_init_mtx[:3, 3] -= self.Asdf2px[:3, 3]
        self.global_init_mtx = global_init_mtx
        # create a centering transform to allow rotations to be properly applied

    @property
    def train_step_0_transforms(self):
        tr = [
            CopyField(source_key='im', dest_key='copy_path'),
            NiBabelLoader(fields='im', root_dir=self.im_dir),

            RandShiftIntensityd(keys='im', offsets=10, prob=0.5),
            RandGaussianNoised(keys='im', std=5, prob=0.5),
            GaussianBlurTransform(blur_sigma=(1, 5),
                                  keys=['im'],
                                  different_sigma_per_channel=True,
                                  p_per_channel=0.5,
                                  p_per_sample=0.5),
            GaussianBrightnessTransform(mu=0.0,
                                        sigma=2,
                                        keys=['im'],
                                        p_per_sample=0.5),

            ExpandDims(fields='im', axis=0),
            Clip(fields='im', new_min=-160, new_max=240),
            CenterIntensities(fields='im', subtrahend=40, divisor=200),
            CustomRandomAffine(
                keys=('im'),
                mode=("bilinear"),
                prob=1.0,
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border",
                device=torch.device('cuda:0')
            ),
            ApplyAffineToPoints(gt_trans_key=self.gt_trans_key, init_key='init_mtx'),
            #CropVolume(fields='im', bbox='bbox'),
            RandomChoice([
                AddMaskChannel(
                    mean_np=self.mean_np,
                    global_init_mtx=None,
                    init_affine_key='init_mtx',
                    translate_range=(3, 3, 3),
                    scale_range=(.1, .1, .1),
                    rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
                    device=torch.device('cuda:0')
                ),
                AddMaskChannel(
                    mean_np=self.mean_np,
                    global_init_mtx=None,
                    init_affine_key='init_mtx',
                    translate_range=(1, 1, 1),
                    scale_range=(.03, .03, .03),
                    rotate_range=(np.pi / 48, np.pi / 48, np.pi / 48),
                    device=torch.device('cuda:0')
                ),
            ])

        ]
        return Compose(tr)

    @property
    def train_other_steps_transforms(self):
        tr = [
            ApplyAffineToPoints(gt_trans_key=self.gt_trans_key, affine_key='affine_matrix', init_key=None),
            RandomChoice([AddMaskChannel(
                mean_np=self.mean_np,
                global_init_mtx=None,
                init_affine_key='init_mtx',
                translate_range=(3, 3, 3),
                scale_range=(.1, .1, .1),
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),

                device=torch.device('cuda:0')
            ),
                AddMaskChannel(
                    mean_np=self.mean_np,
                    global_init_mtx=None,
                    init_affine_key='init_mtx',
                    translate_range=(1, 1, 1),
                    scale_range=(.03, .03, .03),
                    rotate_range=(np.pi / 48, np.pi / 48, np.pi / 48),

                    device=torch.device('cuda:0')
                )
            ])]
        return Compose(tr)

    @property
    def val_step_0_transforms(self):
        vt = [
            CopyField(source_key='im', dest_key='copy_path'),
            NiBabelLoader(fields='im', root_dir=self.im_dir),
            ExpandDims(fields='im', axis=0),
            Clip(fields='im', new_min=-160, new_max=240),
            CenterIntensities(fields='im', subtrahend=40, divisor=200),
            #CropVolume(fields='im', bbox='bbox'),
            AddMaskChannel(
                mean_np=self.mean_np,
                global_init_mtx=None,
                init_affine_key='init_mtx'
            )

        ]
        return Compose(vt)

    @property
    def val_other_steps_transforms(self):
        vt = [
            AddMaskChannel(
                mean_np=self.mean_np,
                global_init_mtx=None,
                init_affine_key='init_mtx'
            )

        ]
        return Compose(vt)


class PCATransforms():
    def __init__(self, mean_np, Apx2sdf, gt_trans_key, im_dir, global_init_mtx):
        self.Apx2sdf = Apx2sdf
        self.gt_trans_key = gt_trans_key
        self.im_dir = im_dir
        self.mean_np = mean_np
        # affine from sdf to pixel space
        self.Asdf2px = np.linalg.inv(Apx2sdf)
        # account for the initial location at the center of the image
        global_init_mtx[:3, 3] -= self.Asdf2px[:3, 3]
        self.global_init_mtx = global_init_mtx
        # create a centering transform to allow rotations to be properly applied

    @property
    def train_step_0_transforms(self):
        tr = [
            CopyField(source_key='im', dest_key='copy_path'),
            NiBabelLoader(fields='im', root_dir=self.im_dir),

            RandShiftIntensityd(keys='im', offsets=10, prob=0.5),
            RandGaussianNoised(keys='im', std=5, prob=0.5),
            GaussianBlurTransform(blur_sigma=(1, 5),
                                  keys=['im'],
                                  different_sigma_per_channel=True,
                                  p_per_channel=0.5,
                                  p_per_sample=0.5),
            GaussianBrightnessTransform(mu=0.0,
                                        sigma=2,
                                        keys=['im'],
                                        p_per_sample=0.5),

            ExpandDims(fields='im', axis=0),
            Clip(fields='im', new_min=-160, new_max=240),
            CenterIntensities(fields='im', subtrahend=40, divisor=200),
            CustomRandomAffine(
                keys=('im'),
                mode=("bilinear"),
                prob=1.0,
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border"
                # device=torch.device('cuda:0')
            ),
            ApplyAffineToPoints(gt_trans_key=self.gt_trans_key, init_key='init_mtx'),
            CropVolume(fields='im', bbox='bbox'),
            RandomChoice([
                AddMaskChannel(
                    mean_np=self.mean_np,
                    global_init_mtx=None,
                    init_affine_key='init_mtx',
                    translate_range=(3, 3, 3),
                    scale_range=(.1, .1, .1),
                    rotate_range=(np.pi / 48, np.pi / 48, np.pi / 48),
                    device=torch.device('cuda:0')
                ),
                AddMaskChannel(
                    mean_np=self.mean_np,
                    global_init_mtx=None,
                    init_affine_key='init_mtx',
                    translate_range=(1, 1, 1),
                    scale_range=(.01, .01, .01),
                    rotate_range=(np.pi / 96, np.pi / 96, np.pi / 96),
                    device=torch.device('cuda:0')
                ),
            ])

        ]
        return Compose(tr)

    @property
    def train_other_steps_transforms(self):
        tr = [
            ApplyAffineToPoints(gt_trans_key=self.gt_trans_key, affine_key='affine_matrix', init_key=None),
            RandomChoice([AddMaskChannel(
                mean_np=self.mean_np,
                global_init_mtx=None,
                init_affine_key='init_mtx',
                translate_range=(3, 3, 3),
                scale_range=(.1, .1, .1),
                rotate_range=(np.pi / 48, np.pi / 48, np.pi / 48),

                device=torch.device('cuda:0')
            ),
                AddMaskChannel(
                    mean_np=self.mean_np,
                    global_init_mtx=None,
                    init_affine_key='init_mtx',
                    translate_range=(1, 1, 1),
                    scale_range=(.01, .01, .01),
                    rotate_range=(np.pi / 96, np.pi / 96, np.pi / 96),

                    device=torch.device('cuda:0')
                )
            ])]
        return Compose(tr)

    @property
    def val_step_0_transforms(self):
        vt = [
            CopyField(source_key='im', dest_key='copy_path'),
            NiBabelLoader(fields='im', root_dir=self.im_dir),
            ExpandDims(fields='im', axis=0),
            Clip(fields='im', new_min=-160, new_max=240),
            CenterIntensities(fields='im', subtrahend=40, divisor=200),
            CropVolume(fields='im', bbox='bbox'),
            AddMaskChannel(
                mean_np=self.mean_np,
                global_init_mtx=None,
                init_affine_key='init_mtx',
                # translate_range = (5.0, 5.0, 2.5)
                device=torch.device('cuda:0')
            )

        ]
        return Compose(vt)

    @property
    def val_other_steps_transforms(self):
        vt = [
            AddMaskChannel(
                mean_np=self.mean_np,
                global_init_mtx=None,
                init_affine_key='init_mtx',
                # translate_range = (5.0, 5.0, 2.5)
                device=torch.device('cuda:0')
            )

        ]
        return Compose(vt)
