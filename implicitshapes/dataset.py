import torch
from copy import deepcopy

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from copy import copy
import os
import logging
import random
import numpy as np
from monai.transforms import RandAffined, Affine, Resample
from monai.transforms.utils import create_rotate, create_translate, create_scale, create_shear, create_grid
from monai.transforms.compose import MapTransform, Randomizable
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    optional_import,
)
from monai.transforms.compose import Randomizable, Transform


class CustomRandomAffine(MapTransform):
    """
    Calls MONAI's random affine transform, but saves the resulting affine matrix so we can use it to
    also update the sampled GT sdf coordinates
    Args:
        Same args as MONAI RandAffined
    """
    def __init__(self, keys, rotate_range=None, shear_range=None, translate_range=None, scale_range=None, mode='bilinear', prob=0.1, padding_mode='border', device=None):
        super().__init__(keys)
        self.rand_affine = RandAffined(keys, rotate_range=rotate_range, shear_range=shear_range,
                translate_range=translate_range, scale_range=scale_range, mode=mode, prob=prob, padding_mode=padding_mode,
                device=device)
        

    def __call__(self, data_dict):
        # exectute the random affine on the image
        data_dict = self.rand_affine(data_dict)
        # now extract the affine parameters used in this affine transform and save the resulting affine matrix
        rotate_params = self.rand_affine.rand_affine.rand_affine_grid.rotate_params
        shear_params = self.rand_affine.rand_affine.rand_affine_grid.shear_params
        translate_params = self.rand_affine.rand_affine.rand_affine_grid.translate_params
        scale_params = self.rand_affine.rand_affine.rand_affine_grid.scale_params
        affine = np.eye(4)
        if rotate_params:
            affine = affine @ create_rotate(3, rotate_params)
        if shear_params:
            affine = affine @ create_shear(3, shear_params)
        if translate_params:
            affine = affine @ create_translate(3, translate_params)
        if scale_params:
            affine = affine @ create_scale(3, scale_params)

        data_dict['affine_matrix'] = affine
        return data_dict

class ApplyAffineToPoints():
    """
    Applies an affine transform, executed by MONAI, to a set of GT sdf coordiantes
    Args:

    sample_key: the key holding the sampled sdf coordinates (should be Nx4, where the first element
        is the sdf value and the next three are the sdf coordinates)
    im_key: the key holding the transformed image
    affine_key: the key holding the affine matrix
    """
    def __init__(self, sample_key='samples', im_key='im', affine_key='affine_matrix', gt_trans_key='t'):
        self.sample_key = sample_key
        self.affine_key = affine_key
        self.im_key = im_key
        self.gt_trans_key = gt_trans_key

    def __call__(self, data_dict):
        samples = data_dict[self.sample_key]
        affine_mtx = data_dict[self.affine_key]
        im = data_dict[self.im_key]

        # extract the sdf coordinates from the samples
        points = samples[:,1:]

        # transpose them and turn them into homogenous coordinates
        points = np.transpose(points)
        dummy = np.ones((1, points.shape[1]))

        # now center the points before applying the affine transform
        points -= np.expand_dims((np.asarray(im.shape[1:]) - 1)/2,1)
        points = np.concatenate((points,dummy))

        # because MONAI affine matrices actually describe the **inverse** rigid transform applied
        # we must apply the inverse affine to our points to make them align with how the im was 
        # transformed
        points = np.linalg.solve(affine_mtx, points)

        # new grab the inhomogenous transformed points, un-centre them, and store them back into 
        # our samples field
        points = points[:3, :]

        points += np.expand_dims((np.asarray(im.shape[1:]) - 1)/2,1)
        points = np.transpose(points)

        gt_trans = np.asarray(data_dict[self.gt_trans_key])

        # now center the points before applying the affine transform
        gt_trans -= (np.asarray(im.shape[1:]) - 1)/2
        gt_trans = np.concatenate((gt_trans,np.ones(1)))

        # because MONAI affine matrices actually describe the **inverse** rigid transform applied
        # we must apply the inverse affine to our points to make them align with how the im was 
        # transformed
        gt_trans = np.linalg.solve(affine_mtx, gt_trans)
        gt_trans = gt_trans[:3]

        gt_trans += (np.asarray(im.shape[1:]) - 1)/2
        data_dict[self.gt_trans_key] = gt_trans.tolist()



        samples[:,1:] = torch.tensor(points)

        data_dict[self.sample_key] = samples

        return data_dict

class CreateAffinePath():

    def __init__(self, global_init_mtx, Asdf2px, gt_trans_key='t', init_guess_aff_key='cur_starting_aff'):
        self.global_init = global_init_mtx
        self.gt_trans_key = gt_trans_key
        self.init_guess_aff_key = init_guess_aff_key
        self.Asdf2px = Asdf2px
    def __call__(self, data_dict):
        """
        Transform to create an initial guess that sits somewhere in the path from the 
        average location and the gt location
        """

        json_trans = data_dict[self.gt_trans_key]
        gt_trans = np.asarray(json_trans)
        gt_trans = gt_trans - self.Asdf2px[:3,3]
        starting_trans = self.global_init[:3,3]
        diff = gt_trans - starting_trans
        rand_path = diff  * random.uniform(0,1)
        cur_affine_mtx = np.eye(4)
        cur_affine_mtx[:3,3] = rand_path 
        data_dict[self.init_guess_aff_key] = cur_affine_mtx
        return data_dict



class CreateScaleInitialGuess():

    def __init__(self, Asdf2px, gt_trans_key='t', init_guess_aff_key='cur_starting_aff'):
        self.gt_trans_key = gt_trans_key
        self.init_guess_aff_key = init_guess_aff_key
        self.Asdf2px = Asdf2px

    def __call__(self, data_dict):
        """
        Transform to create an initial guess that sits somewhere in the path from the 
        average location and the gt location
        """

        json_trans = data_dict[self.gt_trans_key]
        gt_trans = np.asarray(json_trans)
        gt_trans = (gt_trans - self.Asdf2px[:3,3]) 
        cur_affine_mtx = np.eye(4)
        cur_affine_mtx[:3,3] = gt_trans
        data_dict[self.init_guess_aff_key] = cur_affine_mtx
        return data_dict


class AddMaskChannel():
    """
    Adds an extra channel to the image consisting of the current sdf prediction. Optionally can also add random jitters 
    to the mask, thereby simulating different initial guesses for the encoder to refine. It will also save an affine
    matrix describing the intial guess, which can then be used to update the sample sdf points
    Args:

    mean_np: the mean sdf image in its initial configuration
    final_affine_key: the key to save the final affine matrix describing the current guess
    global_affine_mtx [Optional]: the affine matrix specifying an initial configuration that is universal to all images
    init_affine_mtx [Optional]: a refinement of the global_affine_mtx specifying an image-specific initial guess
    translate_range, scale_range, rotate_range: range of random jitters to the initial guess
    """
    def __init__(self, mean_np, im_key='im', final_affine_key='cur_affine_mtx', global_init_mtx=None, init_affine_key=None, translate_range=None, scale_range=None, rotate_range=None, device=None):
        self.mean_np = torch.tensor(np.expand_dims(mean_np,0)).float()
        # self.mean_np[self.mean_np < 0] = 0
        # self.mean_np[self.mean_np > 0] = 1

        self.translate_range = translate_range
        self.scale_range = scale_range
        self.rotate_range = rotate_range
        self.init_affine_key = init_affine_key
        self.im_key = im_key
        self.global_init_mtx = global_init_mtx
        self.device = device
        # if it's not specified, we initialize it to the identity
        if self.global_init_mtx is None:
            self.global_init_mtx = np.eye(4)
        self.final_affine_key = final_affine_key
       

    def __call__(self, data_dict):

        # if specified, calculate some random jitters of the initial guess
        translate_params = None
        scale_params = None
        rotate_params = None
        if self.translate_range:
            translate_params = [random.normalvariate(0, f) for f in self.translate_range if f is not None]

        if self.scale_range:
            scale_params = [random.normalvariate(0, f) + 1.0  for f in self.scale_range if f is not None]

        if self.rotate_range:
            rotate_params = [random.normalvariate(0, f) for f in self.rotate_range if f is not None]

        
        cur_affine = self.global_init_mtx

        # if specified add image-wise initial solution to the universal solution
        if self.init_affine_key:
            init_affine = data_dict[self.init_affine_key]
            # update the affine matrix for the sdf coordinates
            cur_affine = init_affine @ cur_affine 
        
        # create the random affine jitter of the initial solution
        rand_affine = np.eye(4)
        if rotate_params:
            rand_affine = rand_affine @ create_rotate(3, rotate_params)
        if translate_params:
            rand_affine = rand_affine @ create_translate(3, translate_params)
        if scale_params:

            rand_affine = rand_affine @ create_scale(3, scale_params)

        # update both coordinate affine and the image affine matrix
        cur_affine = rand_affine @ cur_affine 
        cur_affine_for_image = deepcopy(cur_affine)
        
        # because Monai transforms use the affine to specify an inverse mapping, we must
        # invert the given affine matrix before giving it to the transform
        cur_affine_for_image = np.linalg.inv(cur_affine_for_image)

        # transform the image mask/sdf using the image-specific affine matrix
        affine_trans = Affine_w_Mtx(affine_mtx=torch.tensor(cur_affine_for_image), device=self.device, padding_mode='border')
        new_mean_np = torch.tensor(affine_trans(self.mean_np))
        new_mean_np = new_mean_np.float()


        # since the model maps from image coordinates to canonical coordinates, we must store 
        # the inverse affine for the points
        cur_affine = np.linalg.inv(cur_affine)
        # for compatbility later, we store the transpose
        cur_affine = np.transpose(cur_affine)
        data_dict[self.final_affine_key] = cur_affine
       
        # cocatenate the extra channel to the image
        cur_im = data_dict[self.im_key]
        if not torch.is_tensor(cur_im):
            cur_im = torch.tensor(cur_im)
        new_mean_np = new_mean_np.to(cur_im.device)
        
        cur_im = torch.cat((cur_im, new_mean_np))
        data_dict[self.im_key] = cur_im
        return data_dict

class CopyField():

    def __init__(self, source_key, dest_key):
        self.source_key = source_key
        self.dest_key = dest_key

    def __call__(self, data_dict):

        data_dict[self.dest_key] = copy(data_dict[self.source_key]) 
        return data_dict

def unpack_sdf_samples(filename, subsample=None):

    # load the npz file
    npz = np.load(filename)

    # extrac the pos and negative distance values and coordinates, removing any nans
    pos_sdf, pos_points = remove_nans(torch.from_numpy(npz["pos_sdf"]), torch.from_numpy(npz["pos_points"]))
    neg_sdf, neg_points = remove_nans(torch.from_numpy(npz["neg_sdf"]), torch.from_numpy(npz["neg_points"]))

    # concatenate the coordinates the distance values ontop of each other 
    pos_sdf = torch.unsqueeze(pos_sdf, 1)
    neg_sdf = torch.unsqueeze(neg_sdf, 1)
    pos_tensor = torch.cat([pos_sdf, pos_points], 1)
    neg_tensor = torch.cat([neg_sdf, neg_points], 1)

    # randomly subsample both based on the subsample parameter
    half = int(subsample / 2)
    # half positive and half negative
    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    # concatenate both positive and negatives together
    samples = torch.cat([sample_pos, sample_neg], 0)

    return {'samples': samples}

def unpack_sdf_samples_from_ram(data, subsample):
    pos_tensor = data['pos']
    neg_tensor = data['neg']

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return {'samples': samples}

def remove_nans(sdf_tensor, points_tensor):
    tensor_nan = torch.isnan(sdf_tensor)

    sdf_tensor = sdf_tensor[~tensor_nan]
    points_tensor = points_tensor[~tensor_nan,:]
    return sdf_tensor, points_tensor

class SDFSamples(torch.utils.data.Dataset):
    """
    Adapated from the DeepSDF repo. Not set up for RAM loading, so do not use that
    json_list: list of dicts, each dict should have a 'path' field pointing to the SDF sample .npz's
    subsample: number of points to sample for each shape
    im_root: root location of where the sdf .npz's are located
    """
    def __init__(
        self,
        json_list,
        subsample,
        im_root,
        load_ram=False,
        transforms = None
    ):
        self.subsample = subsample
        self.im_root = im_root
        self.transforms = transforms
        self.json_list = json_list

        logging.debug(
            "using "
            + str(len(self.json_list))
            + " shapes from data source "
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for cur_dict in self.json_list:
                cur_file = cur_dict['path']
                filename = os.path.join(self.im_root, cur_file)
                npz = np.load(filename)
                pos_sdf, pos_points = remove_nans(torch.from_numpy(npz["pos_sdf"]), torch.from_numpy(npz["pos_points"]))
                neg_sdf, neg_points = remove_nans(torch.from_numpy(npz["neg_sdf"]), torch.from_numpy(npz["neg_points"]))
                pos_sdf = torch.unsqueeze(pos_sdf, 1)
                neg_sdf = torch.unsqueeze(neg_sdf, 1)
                pos_tensor = torch.cat([pos_sdf, pos_points], 1)
                neg_tensor = torch.cat([neg_sdf, neg_points], 1)
                self.loaded_data.append(
                    {
                        'pos': pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        'neg': neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    }                    
                )

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, idx):
        if self.load_ram:
            sample = unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample)
            sample[idx] = idx
            return sample
        else:
            # get the filename from the current dict
            filename = os.path.join(
                self.im_root, self.json_list[idx]['path']
            )
            # load the points coordinates and sdf values
            sample = unpack_sdf_samples(filename, self.subsample)
            # also record the index of the shape being loaded 
            sample['idx'] = idx
            sample['path'] = self.json_list[idx]['path']
            # if there are any additional entries in the dict, load them as well
            for k,v in self.json_list[idx].items():
                if k != 'idx' and k != 'path':
                    sample[k] = v
            # conduct any transforms, if specified
            if self.transforms:
                sample = self.transforms(sample)
            return sample



#Adapted from MONAI
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

