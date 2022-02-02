### Deep Implicit Statistical Shape Models for 3D Medical Image Delineation

Please find the paper at [DISSM](https://arxiv.org/pdf/2104.02847.pdf)

Training requires the MONAI library

### Pre-requisties

There are three main steps to using this system:

1. Create a library of shapes from a set of masks. Please see [CONVERT.md](implicitshapes/CONVERT.md).

2. Train a deep SDF auto-decoded model based off the library of shapes. Please see [DEEPSDF.md](implicitshapes/DEEPSDF.md).

3. Now train an encoder to estimate an organ's shape given an image.


### Training DISM

To train DISM, we use MSL strategy. First we train Translation model with bigger resolution and 
then use the translated shape to crop and then perform subsequent models with the cropped resolution. 


### Training Translation

See [PREDICT_TRANSLATION.md](implicitshapes/PREDICT_TRANSLATION.md)

    




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
