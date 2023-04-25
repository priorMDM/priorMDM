# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from data_loaders import humanml_utils
from eval.eval_multi import extract_motions
from model.comMDM import ComMDM
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import edit_multi_args
from utils.model_util import load_model
from utils import dist_util
from model.cfg_sampler import UnconditionedModel
from data_loaders.get_data import get_dataset_loader
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil


def main():
    print(f"generating samples")
    args = edit_multi_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    fps = 20
    n_frames = 80
    args.num_repetitions = 1  # Hardcoded - prefix completion has limited diversity.
    args.edit_mode = 'prefix'  # Prefix completion script.
    max_frames = n_frames + 1  # for global root pose
    sample1 = None  # a place holder for two characters, do not delete
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'prefix_completion_{}_{}_{}_seed{}'.format(name, niter, args.edit_mode, args.seed))
        if args.text_condition != '':
            out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    args.guidance_param = 0
    model, diffusion = load_model(args, data, dist_util.dev(), ModelClass=ComMDM)

    iterator = iter(data)
    input_motions, model_kwargs = next(iterator)

    input_motions = input_motions[..., :n_frames+1]
    model_kwargs['y']['mask'] = model_kwargs['y']['mask'][..., :n_frames+1]
    model_kwargs['y']['other_motion'] = model_kwargs['y']['other_motion'][..., :n_frames+1]

    gt_frames_per_sample = {}
    model_kwargs['y']['inpainted_motion_multi'] = [input_motions, model_kwargs['y']['other_motion'].to(input_motions.device)]
    model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool,
                                                           device=input_motions.device)  # True means use gt motion
    for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
        start_idx, end_idx = int(args.prefix_end * length), int(args.suffix_start * length)
        gt_frames_per_sample[i] = list(range(0, start_idx))
        model_kwargs['y']['inpainting_mask'][i, :, :,
        start_idx+1:] = False  # do inpainting in those frames

    all_motions = []
    all_lengths = []
    all_text = []
    all_captions = []

    gt, gt1 = extract_motions(input_motions.cpu(), model_kwargs['y']['other_motion'].cpu(), data)

    def process_to_save(_sample0, _sample1):
        sample_save = np.concatenate((_sample0[None], _sample1[None]), axis=0).transpose(1, 0, 4, 2, 3)
        return sample_save.reshape(*sample_save.shape[:3], -1)

    gt_save = process_to_save(gt, gt1)

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}
        model_kwargs['y']['inpainted_motion_multi'][0] = model_kwargs['y']['inpainted_motion_multi'][0].to(dist_util.dev())
        model_kwargs['y']['inpainted_motion_multi'][1] = model_kwargs['y']['inpainted_motion_multi'][1].to(dist_util.dev())

        
        sample_fn = diffusion.p_sample_loop
        # del model_kwargs['y']['other_motion']  # not necessary - just to make sure

        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames+1),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=(input_motions, model_kwargs['y']['other_motion']),
            progress=True,
            predict_two_person=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        sample, sample1 = sample
        sample, sample1 = extract_motions(sample, sample1, data)

        text_key = 'text'
        all_text += model_kwargs['y'][text_key]
        all_captions += model_kwargs['y'][text_key]

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'gt': gt_save, 'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})#,
             # 'all_captions': all_captions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
    sample_files = []
    num_samples_in_out_file = 7

    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = f'Prefix Completion'
            length = all_lengths[rep_i*args.batch_size + sample_i] - 1
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            motion1 = None
            if sample1 is not None:
                motion1 = sample1[sample_i].transpose(2, 0, 1)[:length]
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
            plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps, vis_mode=args.edit_mode,
                           gt_frames=gt_frames_per_sample.get(sample_i, []),
                           joints2=motion1)#, captions=captions)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

            rep_files.append(animation_save_path)
        if args.num_repetitions > 1:
            all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
            ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
            hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
            ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
            os.system(ffmpeg_rep_cmd)
            print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')
            sample_files.append(all_rep_save_file)

            if (sample_i+1) % num_samples_in_out_file == 0 or sample_i+1 == args.num_samples:
                all_sample_save_file = os.path.join(out_path, f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4')
                ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
                vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
                ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_file}'
                os.system(ffmpeg_rep_cmd)
                print(f'[(samples {(sample_i - len(sample_files) + 1):02d} to {sample_i:02d}) | all repetitions | -> {all_sample_save_file}]')
                sample_files = []


    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name='pw3d',  # args.multi_dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='validation',  # args.multi_eval_splits,
                              load_mode='prefix')  # for GT vis
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
