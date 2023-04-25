# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from model.DoubleTake_MDM import doubleTake_MDM
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import load_model
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from utils.sampling_utils import unfold_sample_arb_len, double_take_arb_len
import pandas as pd


def calc_frame_colors(handshake_size, blend_size, step_sizes, lengths):
    for ii, step_size in enumerate(step_sizes):
        if ii == 0:
            frame_colors = ['orange'] * (step_size - handshake_size - blend_size) + \
                           ['blue'] * blend_size + \
                           ['purple'] * (handshake_size // 2)
            continue
        if ii == len(step_sizes) - 1:
            frame_colors += ['purple'] * (handshake_size // 2) + \
                            ['blue'] * blend_size + \
                            ['orange'] * (lengths[ii] - handshake_size - blend_size)
            continue
        frame_colors += ['purple'] * (handshake_size // 2) + ['blue'] * blend_size + \
                        ['orange'] * (lengths[ii] - 2 * handshake_size - 2 * blend_size) + \
                        ['blue'] * blend_size + \
                        ['purple'] * (handshake_size // 2)
    return frame_colors


def main():
    print(f"generating samples")
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    fps = 30 if args.dataset == 'babel' else 20
    n_frames = 150
    is_using_data = not args.input_text
    dist_util.setup_dist(args.device)
    is_csv, is_txt = False, False
    assert (args.double_take)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'DoubleTake_samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.input_text != '':
            if ".txt" in args.input_text:
                out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')
                is_txt = True
            elif ".csv" in args.input_text:
                out_path += '_' + os.path.basename(args.input_text).replace('.csv', '').replace(' ', '_').replace('.', '')
                is_csv = True
            else:
                raise TypeError("Incorrect text file type, use csv or txt")
        if args.sample_gt:
            out_path += "_gt"
        out_path += f"_handshake_{args.handshake_size}"
        if args.double_take:
            out_path += "_double_take"
            out_path += f"_blend_{args.blend_len}"
            out_path += f"_skipSteps_{args.skip_steps_double_take}"

    # this block must be called BEFORE the dataset is loaded
    if args.input_text != '':
        assert os.path.exists(args.input_text)
        if is_txt:
            with open(args.input_text, 'r') as fr:
                texts = fr.readlines()
            texts = [s.replace('\n', '') for s in texts]
            args.num_samples = len(texts)
        elif is_csv:
            df = pd.read_csv(args.input_text)
            args.num_samples = len(list(df['text']))

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = load_model(args, data, dist_util.dev(), ModelClass=doubleTake_MDM)

    if is_using_data:
        iterator = iter(data)
        gt_motion, model_kwargs = next(iterator)
    elif is_csv:
        model_kwargs = {'y': {
            'mask': torch.ones((len(list(df['text'])), 1, 1, 196)), #196 is humanml max frames number
            'lengths': torch.tensor(list(df['length'])),
            'text': list(df['text']),
            'tokens': [''],
            'scale': torch.ones(len(list(df['text'])))*2.5
        }}
    elif is_txt:
        model_kwargs = {'y': {
            'mask': torch.ones((len(texts), 1, 1, 196)), # 196 is humanml max frames number
            'lengths': torch.tensor([n_frames]*len(texts)),
            'text': texts,
            'tokens': [''],
            'scale': torch.ones(len(texts))*2.5
        }}
    else:
        raise TypeError("Only text to motion is availible atm")

    all_motions = []
    all_lengths = []
    all_text = []
    all_captions = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}

        max_arb_len = model_kwargs['y']['lengths'].max()
        min_arb_len = 2 * args.handshake_size + 2*args.blend_len + 10

        for ii, len_s in enumerate(model_kwargs['y']['lengths']):
            if len_s > max_arb_len:
                model_kwargs['y']['lengths'][ii] = max_arb_len
            if len_s < min_arb_len:
                model_kwargs['y']['lengths'][ii] = min_arb_len
        samples_per_rep_list, samples_type = double_take_arb_len(args, diffusion, model, model_kwargs, max_arb_len)

        step_sizes = np.zeros(len(model_kwargs['y']['lengths']), dtype=int)
        for ii, len_i in enumerate(model_kwargs['y']['lengths']):
            if ii == 0:
                step_sizes[ii] = len_i
                continue
            step_sizes[ii] = step_sizes[ii-1] + len_i - args.handshake_size

        final_n_frames = step_sizes[-1]

        for sample_i, samples_type_i in zip(samples_per_rep_list, samples_type):

            sample = unfold_sample_arb_len(sample_i, args.handshake_size, step_sizes, final_n_frames, model_kwargs)

            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
            if args.dataset == 'babel':
                from data_loaders.amass.transforms import SlimSMPLTransform
                transform = SlimSMPLTransform(batch_size=args.batch_size, name='SlimSMPLTransform', ename='smplnh', normalization=True)

                all_feature = sample #[bs, nfeats, 1, seq_len]
                all_feature_squeeze = all_feature.squeeze(2) #[bs, nfeats, seq_len]
                all_feature_permutes = all_feature_squeeze.permute(0, 2, 1) #[bs, seq_len, nfeats]
                splitted = torch.split(all_feature_permutes, all_feature.shape[0]) #[list of [seq_len,nfeats]]
                sample_list = []
                for seq in splitted[0]:
                    all_features = seq
                    Datastruct = transform.SlimDatastruct
                    datastruct = Datastruct(features=all_features)
                    sample = datastruct.joints

                    sample_list.append(sample.permute(1, 2, 0).unsqueeze(0))
                sample = torch.cat(sample_list)
            else:
                rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
                if args.dataset == 'babel':
                    rot2xyz_pose_rep = 'rot6d'
                rot2xyz_mask = None

                sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                       jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                       get_rotations_back=False)

            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'

            all_text += model_kwargs['y'][text_key]
            all_captions += model_kwargs['y'][text_key]

            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

            print(f"created {len(all_motions) * args.batch_size} samples")

    # param update for unfolding visualization
    # out of for rep_i
    old_num_samples = args.num_samples
    args.num_samples = 1
    args.batch_size = 1
    n_frames = final_n_frames

    num_repetitions = args.num_repetitions

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = [n_frames] * num_repetitions

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    frame_colors = calc_frame_colors(args.handshake_size, args.blend_len, step_sizes, model_kwargs['y']['lengths'])
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': num_repetitions, 'frame_colors': frame_colors})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
    if args.dataset == 'babel':
        skeleton = paramUtil.t2m_kinematic_chain
    sample_files = []
    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i, samples_type_i in zip(range(num_repetitions), samples_type):
            caption = [f'{samples_type_i} {all_text[0]}'] * (model_kwargs['y']['lengths'][0] - int(args.handshake_size/2))
            for ii in range(1, old_num_samples):
                caption += [f'{samples_type_i} {all_text[ii]}'] * (int(model_kwargs['y']['lengths'][ii])-args.handshake_size)
            caption += [f'{samples_type_i} {all_text[ii]}'] * (int(args.handshake_size/2))
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            print(f'[({sample_i}) "{set(caption)}" | Rep #{rep_i} | -> {save_file}]')
            plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps,
                           vis_mode='gt' if args.sample_gt else 'unfold_arb_len', handshake_size=args.handshake_size,
                           blend_size=args.blend_len,step_sizes=step_sizes, lengths=model_kwargs['y']['lengths'])
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)
        if num_repetitions > 1:
            all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
            ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
            hstack_args = f' -filter_complex hstack=inputs={num_repetitions}' if num_repetitions > 1 else ''
            ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
            os.system(ffmpeg_rep_cmd)
            print(f'[({sample_i}) "{set(caption)}" | all repetitions | -> {all_rep_save_file}]')
            sample_files.append(all_rep_save_file)


    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')

def load_dataset(args, n_frames):
    if args.dataset == 'babel':
        args.num_frames = (args.min_seq_len, args.max_seq_len)
    else:
        args.num_frames = n_frames
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=args.num_frames,
                              split='val',
                              load_mode='text_only',
                              short_db=args.short_db,
                              cropping_sampler=args.cropping_sampler)
    data.fixed_length = n_frames
    return data

if __name__ == "__main__":
    main()
