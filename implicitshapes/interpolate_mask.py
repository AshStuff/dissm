import itk
import shutil
import os
import nibabel as ni
import glob
import numpy as np

# expansion: the factor you want to expand by, e.g., 2 or 3
# axis: the axis you want to expand. This corresponds to the world axis. Normally this coincides to
# the image axis, but sometimes they do not, e.g., the x world axis could be the z image axis. The affine
# will indicate this

def morpho(im_path, expansion, axis, out_path):
    # assumes expansion happens on the third dimension
    if not isinstance(expansion, int) or expansion < 2:
        print('Provide a positive integer factor for expansion')
        return

    im = itk.imread(im_path)
    np_im = itk.array_view_from_image(im)
    np_im = np_im > 0

    # we define the axis based on the world coordinates. But the image coordinate axis
    # may be different, so we need to account for this.
    direction_matrix = itk.array_from_matrix(im.GetInverseDirection())
    row = direction_matrix[axis, :]

    # this is the axis in image coordinates, which we'll use for numpy
    axis_ij = np.nonzero(row)
    axis_ij = axis_ij[0][0]

    np_convention = [2, 1, 0]
    axis_ij = np_convention[axis_ij]
    

    orig_shape = list(np_im.shape)
    orig_shape[axis_ij] *= expansion
    new_np_im = np.zeros(orig_shape).astype(np.uint8)
    # fill in every <expansion> slices
    slicers = [slice(0,i) for i in new_np_im.shape]
    # replace the correct axis with a slicer object that only selects every expansion slice
    # slicers[axis_ij] = slice(0,-1, expansion)
    slicers[axis_ij] = slice(0,-1, expansion)
    new_np_im[tuple(slicers)] = np_im

    new_im = itk.image_from_array(new_np_im)
    new_im.SetDirection(im.GetDirection())
    new_im.SetOrigin(im.GetOrigin())
    new_spacing = list(im.GetSpacing())
    new_spacing[axis] /= expansion
    new_im.SetSpacing(new_spacing)

    filter_im = itk.MorphologicalContourInterpolator(Input=new_im)

    itk.imwrite(filter_im, out_path)


def interpolate_all_masks(folder_path, out_folder):
    """
    This function will interpolate all masks smoothly between slices. It is set up to interpolate to 
    that the inter-slice distance is minimally 1
    """
    query_path = os.path.join(folder_path, '*.nii.gz')
    paths = glob.glob(query_path)


    for cur_path in paths:
        print(cur_path)
        cur_filename = os.path.basename(cur_path)
        cur_im = ni.load(cur_path)
        cur_res = abs(cur_im.affine[2,2])
        cur_res = round(cur_res)
        if cur_res == 0:
            cur_res = 1
        # if we already have a 1 mm slice distance, just copy the file
        if cur_res == 1:
            shutil.copyfile(cur_path, os.path.join(out_folder, cur_filename))
        else:
            morpho(cur_path, int(cur_res), 2, os.path.join(out_folder, cur_filename))

