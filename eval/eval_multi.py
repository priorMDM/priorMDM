import os

import torch
import numpy as np

from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from model.comMDM import ComMDM
import utils.rotation_conversions as geometry
from model.cfg_sampler import UnconditionedModel
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import load_model, load_model_wo_clip
from utils.parser_util import evaluation_multi_parser
from diffusion import logger


def extract_motions(_sample0, _sample1, dataset):
    canon0, _sample0 = torch.split(_sample0, [1, _sample0.shape[-1] - 1], dim=-1)
    canon0 = dataset.dataset.t2m_dataset.rebuilt_canon(canon0[:, :4, 0, 0])
    canon1, _sample1 = torch.split(_sample1, [1, _sample1.shape[-1] - 1], dim=-1)
    canon1 = dataset.dataset.t2m_dataset.rebuilt_canon(canon1[:, :4, 0, 0])
    diff_trans = canon1[:, -3:] - canon0[:, -3:]
    _rot0 = geometry.rotation_6d_to_matrix(canon0[:, :6])
    _rot1 = geometry.rotation_6d_to_matrix(canon1[:, :6])
    diff_rot = torch.matmul(_rot0, _rot1.permute(0, 2, 1)).float().cpu()

    n_joints = 22 if _sample0.shape[1] == 263 else 21
    _sample0 = dataset.dataset.t2m_dataset.inv_transform(_sample0.cpu().permute(0, 2, 3, 1)).float()
    _sample0 = recover_from_ric(_sample0, n_joints)
    _sample0 = _sample0.view(-1, *_sample0.shape[2:]).permute(0, 2, 3, 1)

    # _sample1 = model_kwargs['y']['other_motion']
    if _sample1 is not None:
        _sample1 = dataset.dataset.t2m_dataset.inv_transform(
            _sample1.cpu().permute(0, 2, 3, 1)).float()
        _sample1 = recover_from_ric(_sample1, n_joints)
        _sample1 = torch.matmul(diff_rot.view(-1, 1, 1, 1, 3, 3), _sample1.unsqueeze(-1)).squeeze(-1)
        _sample1 = _sample1.view(-1, *_sample1.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
        _sample1 += diff_trans.view(-1, 1, 3, 1).cpu().numpy()
    return _sample0, _sample1


def sample_with_prefix(model, diffusion, data):
    iterator = iter(data)
    input_motions, model_kwargs = next(iterator)
    predict_two_person = True
    
    gt_frames_per_sample = {}
    model_kwargs['y']['inpainted_motion_multi'] = [input_motions,
                                                   model_kwargs['y']['other_motion'].to(input_motions.device)]

    model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool,
                                                           device=input_motions.device)  # True means use gt motion
    for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
        # start_idx int(args.prefix_end * length)
        start_idx = 20
        gt_frames_per_sample[i] = list(range(0, start_idx))
        model_kwargs['y']['inpainting_mask'][i, :, :,
        start_idx:] = False  # do inpainting in those frames

    model_device = next(model.parameters()).device
    model_kwargs['y'] = {key: val.to(model_device) if torch.is_tensor(val) else val for key, val in
                         model_kwargs['y'].items()}
    model_kwargs['y']['inpainted_motion_multi'][0] = model_kwargs['y']['inpainted_motion_multi'][0].to(
        model_device)
    model_kwargs['y']['inpainted_motion_multi'][1] = model_kwargs['y']['inpainted_motion_multi'][1].to(
        model_device)

    sample_fn = diffusion.p_sample_loop

    gt, gt1 = extract_motions(input_motions.cpu(), model_kwargs['y']['other_motion'].cpu(), data)

    del model_kwargs['y']['other_motion']  # not necessary - just to make sure

    sample = sample_fn(
        model,
        model_kwargs['y']['inpainting_mask'].shape,
        # (args.batch_size, model.njoints, model.nfeats, n_frames + 1),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        predict_two_person=predict_two_person,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )

    if predict_two_person:
        sample, sample1 = sample
    else:
        sample1 = model_kwargs['y']['other_motion']

    sample, sample1 = extract_motions(sample, sample1, data)

    # save pred + gt
    def process_to_save(_sample0, _sample1):
        sample_save = np.concatenate((_sample0[None], _sample1[None]), axis=0).transpose(1, 0, 4, 2, 3)
        return sample_save.reshape(*sample_save.shape[:3], -1)

    gt_save = process_to_save(gt, gt1)
    sample_save = process_to_save(sample, sample1)
    return {'pred': sample_save, 'gt': gt_save}

def evaluate_multi(model, diffusion, data, vis_dir=None):
    uncond_model = UnconditionedModel(model)  # make sure we eval unconditioned model
    samples = sample_with_prefix(uncond_model, diffusion, data)
    fps = 20
    n_joints = 22

    results = torch.tensor(samples['pred'][:, :, fps:])
    output_seq = torch.tensor(samples['gt'][:, :, fps:])

    def _slice(_tensor, _time):
        slice = _tensor[:, :, :fps*_time, :].view(_tensor.shape[0], _tensor.shape[1], -1, n_joints, 3)
        root = slice[:, :, :, [0], :]
        slice_no_root = (slice - root)[:, :, :, 1:, :]
        return slice, root, slice_no_root

    prediction_1, prediction_root_1, prediction_no_root_1 = _slice(results, 1)
    prediction_2, prediction_root_2, prediction_no_root_2 = _slice(results, 2)
    prediction_3, prediction_root_3, prediction_no_root_3 = _slice(results, 3)
    gt_1, gt_root_1, gt_no_root_1 = _slice(output_seq, 1)
    gt_2, gt_root_2, gt_no_root_2 = _slice(output_seq, 2)
    gt_3, gt_root_3, gt_no_root_3 = _slice(output_seq, 3)

    def _l2_loss(a, b):
        _loss = torch.sqrt(((a - b) ** 2).sum(dim=-1)).numpy()
        return _loss.mean(), _loss.std()

    loss1, loss1_std = _l2_loss(prediction_1, gt_1)
    loss2, loss2_std = _l2_loss(prediction_2, gt_2)
    loss3, loss3_std = _l2_loss(prediction_3, gt_3)

    loss_root1, loss_root1_std = _l2_loss(prediction_root_1, gt_root_1)
    loss_root2, loss_root2_std = _l2_loss(prediction_root_2, gt_root_2)
    loss_root3, loss_root3_std = _l2_loss(prediction_root_3, gt_root_3)

    loss_no_root1, loss_no_root1_std = _l2_loss(prediction_no_root_1, gt_no_root_1)
    loss_no_root2, loss_no_root2_std = _l2_loss(prediction_no_root_2, gt_no_root_2)
    loss_no_root3, loss_no_root3_std = _l2_loss(prediction_no_root_3, gt_no_root_3)

    return {
        'loss1': {'mean': loss1},
        'loss2': {'mean': loss2},
        'loss3': {'mean': loss3},
        'loss_root1': {'mean': loss_root1},
        'loss_root2': {'mean': loss_root2},
        'loss_root3': {'mean': loss_root3},
        'loss_no_root1': {'mean': loss_no_root1},
        'loss_no_root2': {'mean': loss_no_root2},
        'loss_no_root3': {'mean': loss_no_root3},
    }

if __name__ == '__main__':
    args = evaluation_multi_parser()
    fixseed(args.seed)
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_file = os.path.join(os.path.dirname(args.model_path), 'eval_prefix_{}_{}'.format(name, niter))
    n_samples = 1000
    log_file += f'_{args.eval_mode}'
    log_file += f'_{n_samples}samples'
    log_file += '.log'

    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")

    eval_data = get_dataset_loader(name=args.multi_dataset, batch_size=n_samples, num_frames=None,
                                   split=args.multi_eval_splits, load_mode=args.multi_train_mode)

    logger.log("Creating model and diffusion...")
    args.guidance_param = 0
    model, diffusion = load_model(args, eval_data, dist_util.dev(), ModelClass=ComMDM)
    
    eval_dict = evaluate_multi(model, diffusion, eval_data, vis_dir=args.model_path.replace('.pt', '_vis'))
    with open(log_file, 'w') as f:
        # json.dump(eval_dict, f, indent=4)
        f.write(str(eval_dict))
    print(eval_dict)
    print(f'Saved to [{log_file}]')
