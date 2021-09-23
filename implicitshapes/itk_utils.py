import SimpleITK as sitk
import nibabel as nib
import numpy as np


def get_affine_from_itk(im_sitk):
    spacing = im_sitk.GetSpacing()
    direction = im_sitk.GetDirection()
    origin = im_sitk.GetOrigin()
    affine = np.eye(4)
    affine[0, :3] = np.asarray(direction[:3]) * spacing[0]
    affine[1, :3] = np.asarray(direction[3:6]) * spacing[1]
    affine[2, :3] = np.asarray(direction[6:9]) * spacing[2]
    affine[0, 3] = origin[0]
    affine[1, 3] = origin[1]
    affine[2, 3] = origin[2]
    return affine


def unpad(dens, pad):
    """
    Input:  dens   -- np.ndarray(shape=(nx,ny,nz))
            pad    -- np.array(px,py,pz)

    Output: pdens -- np.ndarray(shape=(nx-px,ny-py,nz-pz))
    """

    nx, ny, nz = dens.shape

    pdens = dens[pad[0][0]:nx - pad[0][1],
            pad[1][0]:ny - pad[1][1],
            pad[2][0]:nz - pad[2][1]]

    return pdens



def reverse_origin_adjust_from_pad(affine, pad):
    diag_elem = np.diag(affine[:3, :3])
    reverse_origin_adjust = pad / diag_elem
    return reverse_origin_adjust



def origin_adjust_from_pad(affine, pad):
    origin_adjust = np.matmul(affine[:3, :3], pad)
    return origin_adjust


def create_sitk_im(np_array, ref_im):
    new_im = sitk.GetImageFromArray(np_array)
    new_im.SetSpacing(ref_im.GetSpacing())
    new_im.SetDirection(ref_im.GetDirection())
    new_im.SetOrigin(ref_im.GetOrigin())
    return new_im


def create_eye_im(nib_im_path):
    im = nib.load(nib_im_path)
    new_ni = nib.Nifti1Image(im.get_fdata(), np.eye(4))
    return new_ni
