# Predicting shape parameters directly

With the DeepSDF model trained, the next (very big) step is to try to predict the shape parameters given an image. To do this, a translation vector, scaling vector, rotation matrix must be estimated first. Only then can the parameters of the actual shape model be fit. For translation, scale, and rotation, typically the way it's formulated is that the parameters that best fit the mean shape to the current image are estimated. 

Currently, only finding the best translation and scale has been explored. There are two ways to do this (1) train an encoder to predict translations and scales that minimize the difference between the mean SDF and each image's SDF. This is probably the best way to do it, but so far I have not got it to work properly. (2) the second way is much simpler, and basically involves training a encoder to directly regress the translations and scales. This will discuss option (2)

## Resample CTs

Because we work with full 3D volumes, we probably cannot work in the original volume resolution. So the first step is to resample them to coarser resolution. I've used [2,2,5]. 

```python
voxel_convert.resample_cts(in_folder, out_folder, out_spacing=[2.0, 2.0, 5.0])
```

The function also prints out the max size of the resampled volumes. To make constant-sized volumes, each volume should be padded to this max size. For LiTS the max size happens to be [256,256, 162].

```python
voxel_convert.pad_cts(in_folder, out_folder, out_size=[256,256,162], constant_values=-1024)
```

Probably the CTs should also be clipped to a [40, 400] window, so that is another preprocessing step that could be done. 

## Creating GT scale/translation/rotation

### Mean Mesh

We want to align our mean shape to each image. To do this, we first create a mean mesh, which is done by first creating an SDF then a mesh from it using marching cubes:

```python
infer_shape.infer_mean_mesh(deep_sdf_ckpt, model_config, out_path)
```

Once you save the mean mesh, then simplify using the Fast Quadric method used in the CONVERSION steps by a factor of `.01`. The magnitude of the mesh vertices depends on the size of the sdf you create as an intermediate step, but it doesn't matter here what size of SDF to create since we will algin the resulting mesh to each GT mesh of each image using the following function:

### Computing GT parameters
```python
align_shape.create_scale_translation_json(in_folder, im_folder, mean_mesh_file
```

This function will project all meshes in the `in_folder` to pixel space based off the corresonding image in `im_folder`. Use the folder that contains the meshes simplified by a factor of `0.01`, in order to ensure comptuational demands are reasonable. For `im_folder` use the folder containing the resampled and padded CTs. The function will then normalize the mean_mesh and then align it to each mesh in `in_folder` in **pixel space**. It will then save a `json` file, compatible with the `DLT` that contains the transformation parameters in image space, which is the space we need to operate in when we want to do predictions. 

## Predicting Transformation Parameters

Currently a **rough** training script is set up to do basic prediction of scale and translation. Based off of rough statistics the mean translation is `155`, `140`, and `80`, and the mean scale is `60`. So the training script will predict deviations from these "mean" values. Obviously, these values should not be hardcoded and they should be calculated stastically from the data. 

```bash
 python predict_params_directly.py --im_root CT_RESAMPLED_PAD --yaml_file config_predict_48.yml --save_path SAVE_PATH --json_list CT_RESAMPLED_PAD/t_scale_list.json                                  
```

It works decently, but can still fail on some cases. I think some additions may be necessary: (1) data augmentation (random scale, translation, rotation, and intensity transforms); and (2): an iterative approach, i.e, iteratively predict the best parameters, to allow the network to make and correct mistakes. 

## Observing Results:

You can predict an aligned mean-shape SDF based off of the predicted parameters and display it along with the original CT to observe the results:

```python
predict_params.predict_sdf(encoder_ckpt, deep_sdf_ckpt, deep_sdf_config, CT_volume_path, SDF_output_path, do_scale)
```

Note the function uses a hardcoded (again, bad practice) value of 5/2 for the z-scale, which will compensate for the larger z resolution of 5 (vs. 2 for x and y)

## Miminizing Differences in SDFs
Ultimately we'll need a way to predict the parameters that best minimize differences in SDFs, but for now this will probably be a good start. 

