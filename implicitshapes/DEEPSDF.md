# Training DeepSDF


## Dataset

To shape the train embedding model a `json` file is expected the provides relative paths to each shape you want to embed.  The `json` file should follow this format: 

```json
[
    {
        'path': relative path to first npz
    },
    ...
    {
        'path': relative path to last npz
    }
]
```
Each entry should specify an npz that stores the SDF samples. 

The `convert.py` file has a util function to create such a json for you from a folder of npz files:

```python
convert.create_sample_json(sdf_folder)
```

However, you should only train a shape embedder on shapes that will not be in the validation or test set of the later pose estimation step, so you likely want to create the json file manually

## Training

`IM_ROOT` should point to the folder of SDF samples. `SAVE_PATH` should point to where you want the checkpoints saved. `config.yml` holds the network and training configurations that seem to work well
```python
python embed_shape.py --im_root IM_ROOT --yaml_file config.yml --save_path SAVE_PATH --file_list JSON_FILE
```


## Verification

After training the shape embedding model, you can check how well the model visually captures the distribution of shapes:

```python
infer_shape.infer_mean_mesh(MODEL_CKPT, config.yml, SAVE_LOC, SDF_RESOLUTION)
```
The above function will generate a uniformly sampled SDF from the mean latent vector and then perform marching cubes to obtain a mesh. The resulting file can be loaded into a mesh viewwer (MITKWorkbench, pyrender). You should see a reasonable organ shape. 

Individual shapes from the training set can also be visualized, where `LATENT_IDX` should specify the index in the training json of the shape you want ot visualize:
```python
infer_shape.infer_mesh(MODEL_CKPT, config.yml, LATENT_IDX, SAVE_LOC, SDF_RESOLUTION)
```

