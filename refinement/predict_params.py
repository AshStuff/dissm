import os

import nibabel as nib
import numpy as np
import torch

from connected_module import ConnectedModule


def load_pretrained_weights(current_model_dict, preloaded_model_dict):
    for key, current_model_value in current_model_dict.items():
        if key in preloaded_model_dict.keys():
            preloaded_model_value = preloaded_model_dict[key]
            if current_model_value.shape == preloaded_model_value.shape:
                current_model_dict[key] = preloaded_model_value
            else:
                shape = preloaded_model_value.shape
                if len(preloaded_model_value.shape) > 1:
                    current_model_value[:shape[0], :shape[1], :] = preloaded_model_value
                    current_model_dict[key] = current_model_value
                else:
                    current_model_value[:shape[0]] = preloaded_model_value
                    current_model_dict[key] = current_model_value
    return current_model_dict


class ComputeMask():
    def __init__(self, margin):
        self.margin = margin

    def __call__(self, data_dict):
        pca = data_dict['pca']
        mask = np.ones_like(pca)
        mask[pca > self.margin] = 0
        mask[pca < -self.margin] = 0
        data_dict['mask'] = mask
        return data_dict


def predict(model_path, volume_file, sdf_file, pca_file, save_path):
    cm = ComputeMask(margin=0.10)

    os.makedirs(save_path, exist_ok=True)
    loaded = torch.load(model_path)
    context_model = ConnectedModule(n_channels=2, n_classes=1)
    context_model = torch.nn.DataParallel(context_model).cuda()
    context_model.load_state_dict(loaded['state_dict'])

    im_np = nib.load(volume_file).get_fdata()
    pca_np = nib.load(pca_file).get_fdata()
    sdf_np = nib.load(sdf_file).get_fdata()
    # pca_np[pca_np == 100] = 1.
    im_np = np.clip(im_np, -160, 240)
    im_np -= 40
    im_np /= 200
    im_np = np.stack((im_np, pca_np))
    im_np = np.expand_dims(im_np, 0)
    context_model.eval()
    with torch.no_grad():
        dict_ = cm({'pca': pca_np})
        mask = dict_['mask']
        pca = torch.from_numpy(pca_np).float().unsqueeze(0).unsqueeze(0).cuda()
        im_np = torch.from_numpy(im_np).float().cuda()
        output = context_model({'im': im_np,
                                'pca': pca})
        refinement = output['r'].cpu().numpy()

        #refinement *= mask
        print(refinement.min(), refinement.max())
        output = output['output'][0][0].data.cpu().numpy()
        pca_np[mask == 1] = sdf_np[mask == 1]
        output = nib.Nifti1Image(pca_np, np.eye(4))
        output.to_filename(os.path.join(save_path, os.path.basename(volume_file)))


def predict_steps(model_path, volume_file, pca_file, save_path):
    cm = ComputeMask(margin=0.25)

    os.makedirs(save_path, exist_ok=True)
    loaded = torch.load(model_path)
    context_model = ConnectedModule(n_channels=2, n_classes=1,)
    context_model = torch.nn.DataParallel(context_model).cuda()
    cur_model_dict = context_model.state_dict()
    updated_dict = load_pretrained_weights(cur_model_dict, loaded['state_dict'])
    cur_model_dict.update(updated_dict)
    context_model.load_state_dict(cur_model_dict)

    im_np = nib.load(volume_file).get_fdata()
    pca_np = nib.load(pca_file).get_fdata()
    pca_np[pca_np == 100] = 1.
    im_np = np.clip(im_np, -160, 240)
    im_np -= 40
    im_np /= 200

    step_size = 7
    context_model.eval()
    with torch.no_grad():
        for mini_step in range(step_size):
            import pdb;pdb.set_trace()
            im_np = np.stack((im_np, pca_np.astype(np.float32)))
            im_np = np.expand_dims(im_np, 0)
            dict_ = cm({'pca': pca_np})
            mask = dict_['mask']
            pca = torch.from_numpy(pca_np).float().unsqueeze(0).unsqueeze(0).cuda()
            im = torch.from_numpy(im_np).float().cuda()
            output = context_model({'im': im,
                                    'pca': pca})
            refinement = output['r'].cpu().numpy()

            refinement *= mask
            print(refinement.min(), refinement.max())
            output = output['output'][0][0].data.cpu().numpy()

            pca_np[mask == 1] = output[mask == 1]
            im_np = im_np[0][0]

        output = nib.Nifti1Image(pca_np, np.eye(4))
        output.to_filename(os.path.join(save_path, os.path.basename(volume_file)))

