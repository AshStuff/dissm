### Deep Implicit Statistical Shape Models for 3D Medical Image Delineation

Please find the paper at [DISM](https://arxiv.org/pdf/2104.02847.pdf)

### Pre-requisties

There are three main steps to using this system:

1. Create a library of shapes from a set of masks. Please see [CONVERT.md](implicitshapes/CONVERT.md).

2. Train a deep SDF auto-decoded model based off the library of shapes. Please see [DEEPSDF.md](implicitshapes/DEEPSDF.md).

3. Now train an encoder to estimate an organ's shape given an image.


### Training DISM

To train DISM, we use MSL strategy. First we train Translation model with bigger resolution and 
then use the translated shape to crop and then perform subsequent models with the cropped resolution. 


### Training Translation

Before starting to train the translation model, we must first complete the steps mentioned in Pre-requisties [1, 2].
We must have the volumes resampled to a fixed resolution and generated sdf values for those resampled volumes using the packages mentioned 
in the pre-requisites.
    We use the resample voxel spacing as (4,4,4) for liver. <br>
    We use the resample voxel spacing as (4,4,1.5) for larynx.
    


#### Training script for translation

```
python train_episodic.py --im_root /data/decathalon/Task03_Liver/latest_update_october/imagesTr_resampled_pad_crop_new/  
--train_json_list ../data/liver_optimize_sdf/ --val_json_list ../data/liver_optimize_sdf/ --yaml_file config_predict_48.yml 
--embed_model_path /data/nas/Projects/StasticalShapeModel/ShapeEmbed/ckpts/Liver0.2_jitter/last_checkpoint.ckpt 
--mean_sdf_file /data/decathalon/Task03_Liver/latest_update_october/mean_centred_sdf.nii.gz --embed_yaml_file config.yml --save_path /data/results/liver/optimize
_through_sdf/resize_4_4_4/episodic_training_step_15_trans --sdf_sample_root /data/decathalon/Task03_Liver/latest_update_october/labelsTr_crop_samples/
```


Necessary files for training translation:

1. Deep SDF trained model -> You can use the `SDF_PREDICT.md` from pre-requisties to train the SDF model of liver.
2. Mean centred sdf file which used for translation.
3. SDF samples points
4. json list with the distance from mean centred liver to the ground truth liver


### Training script for scale

```
python train_episodic.py --im_root /data/decathalon/Task03_Liver/latest_update_october/imagesTr_resampled_pad_crop_new/  
--train_json_list ../data/liver_optimize_sdf/ --val_json_list ../data/liver_optimize_sdf/ 
--yaml_file config_predict_48.yml 
--embed_model_path /data/nas/Projects/StasticalShapeModel/ShapeEmbed/ckpts/Liver0.2_jitter/last_checkpoint.ckpt 
--mean_sdf_file /data/decathalon/Task03_Liver/latest_update_october/mean_centred_sdf_crop.nii.gz 
--embed_yaml_file config.yml --save_path /data/results/liver/optimize_through_sdf/resize_4_4_4/episodic_training_step_15_trans_scale_crop_v1 
--sdf_sample_root /data/decathalon/Task03_Liver/latest_update_october/labelsTr_crop_samples/ --resume /data/results/liver/optimize
_through_sdf/resize_4_4_4/episodic_training_step_7_trans/ckpt/best_model.pth --do_scale
```

Necessary files/things to have for training scale model.

1. Unlike Translation, for scale we need to crop the data so that we can focus on the organ alone. 
   ##### Steps to crop the organ using the trained translation model.
  


### Training script for rotation

```
python train_episodic.py --im_root /data/decathalon/Task03_Liver/latest_update_october/imagesTr_resampled_pad_crop_new/  
--train_json_list ../data/liver_optimize_sdf/ --val_json_list ../data/liver_optimize_sdf/ 
--yaml_file config_predict_48.yml 
--embed_model_path /data/nas/Projects/StasticalShapeModel/ShapeEmbed/ckpts/Liver0.2_jitter/last_checkpoint.ckpt 
--mean_sdf_file /data/decathalon/Task03_Liver/latest_update_october/mean_centred_sdf_crop.nii.gz 
--embed_yaml_file config.yml --save_path /data/results/liver/optimize_through_sdf/resize_4_4_4/episodic_training_step_15_trans_scale_crop_v1 
--sdf_sample_root /data/decathalon/Task03_Liver/latest_update_october/labelsTr_crop_samples/ --resume /data/results/liver/optimize
_through_sdf/resize_4_4_4/episodic_training_step_7_trans/ckpt/best_model.pth --do_scale --do_rotate
```

### Training script for PCA

```
python train_episodic.py --im_root /data/decathalon/Task03_Liver/latest_update_october/imagesTr_resampled_pad_crop_new/  
--train_json_list ../data/liver_optimize_sdf/ --val_json_list ../data/liver_optimize_sdf/ 
--yaml_file config_predict_48.yml 
--embed_model_path /data/nas/Projects/StasticalShapeModel/ShapeEmbed/ckpts/Liver0.2_jitter/last_checkpoint.ckpt 
--mean_sdf_file /data/decathalon/Task03_Liver/latest_update_october/mean_centred_sdf_crop.nii.gz 
--embed_yaml_file config.yml --save_path /data/results/liver/optimize_through_sdf/resize_4_4_4/episodic_training_step_15_trans_scale_crop_v1 
--sdf_sample_root /data/decathalon/Task03_Liver/latest_update_october/labelsTr_crop_samples/ --resume /data/results/liver/optimize
_through_sdf/resize_4_4_4/episodic_training_step_7_trans/ckpt/best_model.pth --do_scale --do_rotate --do_pca --pca_components liver_pca_28.npy
```
