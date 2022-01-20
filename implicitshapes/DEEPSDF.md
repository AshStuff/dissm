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



