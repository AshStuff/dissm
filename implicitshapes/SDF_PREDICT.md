# Predicting shape parameters through the SDF

Ideally, we want to predict shape parameters based on minimzing some divergence measure between a ground truth (GT) mask/sdf and the predicted SDF. In many ways, this is identical to predicting shape parameters directly, but the forward and backward passes and loss will propagate through the shape decoder. 

## Create GT SDF coordinate/value pairs

To calculate a loss through the shape decoder, we must have GT coordinate/SDF pairs. To do this, we conduct a similar workflow to how we created the resampled CTs. 

First we must pad the original masks to ensure that we don't have any areas where they are cut off (similar to what we had to do when converting them to meshes)
```python
voxel_convert.pad_cts(orig_mask_folder, padded_mask_folder, out_size=[1000,1000,1000], constant_values=0)
```
Here we pick an out size that will definitely give us enough space. With the padded masks created, we can then compute SDF versions of the masks. We also want them to resampled to the same space as our `256x256x162` resampled CTs we use as input into our prediction model

```python
voxel_convert.create_resample_sdfs(padded_mask_folder, sdf_folder, resampled_padded_ct_folder, anchor_mesh)
```
This function will compute the SDFs (difference values based on world coordinate spacing) and normalize them using the anchor mesh. This anchor mesh will typically be the first mesh in your folder of meshes, i.e., `liver_0.obj` in `Meshes_Simplify`. This function will also resample each resulting SDF to the same coordinate space as the correspoding CT found in `resampled_padded_ct_folder`, meaning it will do the padding the resampling. 

Now with appropriate SDF volumes created, we can samples the SDF values inside each volume to create a set of SDF values and coordinate pairs that are compatible with the DeepSDF decoder model. 

```python
voxel_convert.create_sdf_voxel_samples(sdf_folder, sdf_sample_folder)
```

The final step is to create a json file that we can use for training. Here we'll make sure to also include in our training json the "ground truth" translation, scales and rotations in the json file calculated using the CDF using the `align_shape.create_scale_translation_json(in_folder, im_folder, mean_mesh_file)` function seen in [DIRECT_PREDICT](DIRECT_PREDICT.md). We do this for debugging and training purposes. 

```python
voxel_convert.create_sample_jsonv2(resampled_padded_ct_folder,sdf_sample_folder, json_out_path, gt_scale_json_path)
```

The resulting training json will look like this:
```json
[
    {
    path: "path_to_SDF_npz",
    im: "path_to_resampled_padded_CT",
    t: [t_x, t_y, t_z], # Gt translations
    s: s, # gt scale
    R: [[]] # gt rotation
    },
    ...
]
```

## Creating a Mean SDF volume

Since we would like to incorporate the concept of "state" to the network, so that the network can know what the initial guess is and then refine that guess (kind of like an active contour model), we need to create an SDF volume that represents the initial guess. For now (since the extreme point guess is not incoporated), we will use a universal initial guess, which is based on the mean translation and scaling estimated from the CPD step: `t=[148.5, 132.8, 93.2]` and `s=63.5`. We then create a corresponding SDF using:
```python
ins.infer_mean_aligned_sdf(shape_embedding_ckpt, shape_embedding_config, out_path, example_CT, trans=[148.5, 132.8, 93.2], scale=[-1/63.5, -1/63.5, 1/63.5*(5/2)])
```

Here `example_CT` can be any resample and padded CT, which will allow the function to create a mean SDF sampled in the same coordinate space. We will use this mean SDF file as an extra channel for the prediction step. 

## Estimating Translation through the SDFs

### Current progress


Currently only translation has been well-tested. The basic idea is this: 
1. We ask the encoder to predict a translation of the sampled GT SDF **pixel** coordinates to center them. Then a scaling is applied (`1/63.7`) to transform them to the **canonical** shape embedding space (`[-1, 1]`)
2. An extra channel is concatenated to the image, representing the initial guess. This is the mean SDF volume (perhaps a mask would work better, I'm not sure)
2. We use the mean embedding in the shape decoder, and minimize the divergence between the shape decoder SDF output and the GT SDF values
3. Unfortunately, direct comparisons between the embedding SDF values and the GT SDFs **may** not be not compatible. It seems ok for prediction translation, but for predicting scale it is problematic. We can apply a mask-based loss, but this seems to have its own issues. This is a problem we must solve. For now the SDF distance loss is used. 
### Augmentations

Several augmentations are applied
1. In addition to intensity based augmentations, a random affine transform is also applied to the **image only** before being concatenated to the initial guess mean SDF channel. Note, any affine transformation of the image must be accounted for the in coordinate values of the GT SDF samples in the .npz files. An additional transform ensures that this is done (`ApplyAffineToPoints`)
2. In addition to augmenting the image, the initial guess should also be randomized, so the network can learn how to refine different initial guesses of the same image. The `AddMaskChannel` takes care of this. It accepts a universal starting guess applied to all images and also an image-specific starting guess. In addition, it will optionally randomly jitter the initial guess. After computing an initial guess, it applies any corresponding transforms to the mean SDF channel being concatenating it to the image. Thus, changes in the initial state are represented by changes in the mean SDF channel. In addition, the initial guess is also applied to the GT SDF coordinates. Thus, any initial guesses are accounted for in both the mean SDF channel and in how the SDF coordinates are transformed. 
3. In training, image-specific initial guesses are computed by taking a random position along the path from the global starting point and the GT translations computed via CDF. 

### Training

To train run
```bash
python predict_scale --im_root resample_padded_ct_folder --yaml_file config_predict_48.yml --save_path SAVE_PATH --json_list JSON_LIST_PATH --embed_model_path DEEPSDF_CKPT --embed_yaml_file config.yml --mean_sdf_file MEAN_SDF_NIFTY
```


