# Mask to SDF Processing Pipeline

To create densely sampled SDFs, needed for DeepSDF, requires several steps. It can be broken down into several steps to make nice meshes, then aligning the meshes, then sampling the SDFs from the meshes.

## Interpolating Masks

If you create a mesh directly from a mask, it will look very blocky and terrible due to the mask pixelation. Additionally, the large inter-slice gaps in the z direction can also cause serious quality problems. The first step is to interpolate the mask in between slices. We use the [itk.MorphologicalContourInterpolator](https://github.com/KitwareMedical/ITKMorphologicalContourInterpolation). Note, I have found it challenging to install this filter. What I have resorted to is to create a python 2.7 environment, downloading the library from github, and then manually installing it using:
```python
python -m pip install --upgrade pip
python -m pip install itk-morphologicalcontourinterpolation
```

The [morpho.yml](morpho.yml) file in the repo is an Anaconda environment that can successfully use this filter. Once you have it installed, you can run
```python
interpolate_mask.interpolate_all_masks(mask_folder, interp_mask_folder)
```
which will interpolate all masks to have at least a 1mm inter-slice distance.

## Mask to Mesh

Once you have nice masks, the next step is to convert them to meshes (in world coordinates). You can run 
```python
convert.convert_all_masks(interp_mask_folder, mesh_folder)
```
which will create a set of `.obj` mesh files in `mesh_folder`. It uses a mask smoothing followed by the marching cubes algorithm.

## Mesh Simplification

The converted meshes will be heavily over-sampled. So they need to be simplified to make them easier to work with, otherwise too much computational resources will be used up. We use the [Fast Quadric Mesh Simplification](https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification). You can download the appropriate binary from their repo for your environment. Assuming the binary is a `QBIN_PATH`, we can reduce the number of vertices by a factor of 10 using:
```python
convert.simplify_all_meshs(mesh_folder, simplified_mesh_folder, QBIN_PATH, 0.1)
```

Reducing by 10 seems to be a good compromise between simplicity and detail. But we also need even more simplified meshes to perform rigid alignment, so we will also do the same except by a factor of 100, and save it to a different folder, `super_simplified_mesh_folder`:
```python
convert.simplify_all_meshs(mesh_folder, super_simplified_mesh_folder, QBIN_PATH, 0.01)
```

## Registration

To model the shape distribution, all shapes must be rigidly aligned with each other. This is because we do not want to model differences in scale, translation, rotation. We only want to model the non-rigid differences between each shape. So, we rigidly align all meshes to an arbitrary anchor mesh using the `pycpd` library, which implements Coherent Point Drift. Here we use the super simplified meshes, because otherwise the algorithm would be too expensive
```python
convert.register_meshes(super_simplified_mesh_folder, super_simplified_registered_folder)
```

Now that we registered the super simplified meshes, we need to apply the same transformation to our less simplified meshes, so that we can actually align the meshes we want to use:
```python
convert.align_large_meshes(simplified_mesh_folder, super_simplified_registered_folder, simplified_registered_folder)
```

## SDF Sampling

All the above steps were all to construct and align nice meshes. With these nice meshes constructed, we can now finally sample the SDFs. As the DeepSDF paper explains, it is advantageous to densely sample closer to the boundary of the shape, while still having a proportion of samples uniformly sampled. I have found that the DeepSDF parameters don't work that well for modelling the liver shape. Instead, I found that sampling a higher proportion as uniformly sampled and using a larger jitter magnitude to be better. The following function will conduct this operation. Note, meshes are scaled to fit within a unit sphere before being sampled, so their world coordinate scales are completely squashed.
> **_NOTE:_** In order to use `jitter` and `uniform_proportion`, you need to use the `modified_mesh_to_sdf` package which is located in `APH_repos/modified_mesh_to_sdf`.



```python
convert.sample_sdf(simplified_registered_folder, sdf_folder, number_of_points=1000000, uniform_proportion=0.2, jitter=.1)
```
This will create `npz` files, each of which has 1 million sdf samples in which to train the DeepSDF model. 

