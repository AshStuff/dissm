# Training DeepSDF


## Dataset

To shape the train embedding model a `json` file named `json_list.json` is expected to live in the folder of SDF samples. The `json` file should follow this format: 

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

The `convert.py` file has a util function to create such a json for you:

```python
convert.create_sample_json(sdf_folder)
```

## Training

`IM_ROOT` should point to the folder of SDF samples. `SAVE_PATH` should point to where you want the checkpoints saved. `config.yml` holds the network and training configurations that seem to work well
```python
python embed_shape.py --im_root IM_ROOT --yaml_file config.yml --save_path SAVE_PATH
```



