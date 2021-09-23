# Training DeepSDF

Training script uses the DLT, so you should be familiar with that library. And make sure to set its location to the `PYTHONPATH` before running

## Dataset

Like all DLT datasets, the dataset is expected to be specified using the `json` format.

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


## Training

`IM_ROOT` should point to the folder of SDF samples. `SAVE_PATH` should point to where you want the checkpoints saved. `config.yml` holds the network and training configurations that seem to work well
```python
python embed_shape.py --im_root IM_ROOT --yaml_file config.yml --save_path SAVE_PATH
```



