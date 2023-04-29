import itertools

import torch
from data_loaders.humanml.networks.modules import *
from data_loaders.humanml.networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
from data_loaders.humanml_utils import get_inpainting_mask

from utils.sampling_utils import double_take_arb_len, unfold_sample_arb_len
from utils import dist_util

def build_models(opt):
    if opt.text_enc_mod == 'bigru':
        text_encoder = TextEncoderBiGRU(word_size=opt.dim_word,
                                        pos_size=opt.dim_pos_ohot,
                                        hidden_size=opt.dim_text_hidden,
                                        device=opt.device)
        text_size = opt.dim_text_hidden * 2
    else:
        raise Exception("Text Encoder Mode not Recognized!!!")

    seq_prior = TextDecoder(text_size=text_size,
                            input_size=opt.dim_att_vec + opt.dim_movement_latent,
                            output_size=opt.dim_z,
                            hidden_size=opt.dim_pri_hidden,
                            n_layers=opt.n_layers_pri)


    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    movement_enc = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

    len_estimator = MotionLenEstimatorBiGRU(opt.dim_word, opt.dim_pos_ohot, 512, opt.num_classes)

    # latent_dis = LatentDis(input_size=opt.dim_z * 2)
    checkpoints = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_est_bigru', 'model', 'latest.tar'), map_location=opt.device)
    len_estimator.load_state_dict(checkpoints['estimator'])
    len_estimator.to(opt.device)
    len_estimator.eval()

    # return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
    return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec, len_estimator

class CompV6GeneratedDataset(Dataset):

    def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        print(opt.model_dir)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, len_estimator = build_models(opt)
        trainer = CompTrainerV6(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)
        epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        min_mov_length = 10 if opt.dataset_name == 't2m' else 6
        # print(mm_idxs)

        print('Loading model: Epoch %03d Schedule_len %03d' % (epoch, schedule_len))
        trainer.eval_mode()
        trainer.to(opt.device)
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
                tokens = tokens[0].split('_')
                word_emb = word_emb.detach().to(opt.device).float()
                pos_ohot = pos_ohot.detach().to(opt.device).float()

                pred_dis = len_estimator(word_emb, pos_ohot, cap_lens)
                pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False

                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)

                    m_lens = mov_length * opt.unit_length
                    pred_motions, _, _ = trainer.generate(word_emb, pos_ohot, cap_lens, m_lens,
                                                          m_lens[0]//opt.unit_length, opt.dim_pose)
                    if t == 0:
                        # print(m_lens)
                        # print(text_data)
                        sub_dict = {'motion': pred_motions[0].cpu().numpy(),
                                    'length': m_lens[0].item(),
                                    'cap_len': cap_lens[0].item(),
                                    'caption': caption[0],
                                    'tokens': tokens}
                        generated_motion.append(sub_dict)

                    if is_mm:
                        mm_motions.append({
                            'motion': pred_motions[0].cpu().numpy(),
                            'length': m_lens[0].item()
                        })
                if is_mm:
                    mm_generated_motions.append({'caption': caption[0],
                                                 'tokens': tokens,
                                                 'cap_len': cap_lens[0].item(),
                                                 'mm_motions': mm_motions})

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.opt = opt
        self.w_vectorizer = w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length < self.opt.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), []

class CompMDMGeneratedDataset(Dataset):

    def __init__(self, args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()


        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):

                    sample = sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'is_transition': model_kwargs['y']['is_transition'][bs_i]
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens, transitions = data['motion'], data['length'], data['caption'], data['tokens'], data['is_transition']
        sent_len = data['cap_len']

        if self.dataset.load_mode == 'eval' and self.dataset.dataset_name != 'babel':  # re-norm is not needed for babel dataset
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # FIXME: I deleted the extra return value ([]), should check whether it breakes anything or not
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), []

class CompMDMInpaintingGeneratedDataset(CompMDMGeneratedDataset):
    def __init__(self, args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()


        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale
                
                model_kwargs['y']['inpainted_motion'] = motion.to(dist_util.dev())
                model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, motion.shape)).float().to(dist_util.dev())

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):

                    sample = sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=motion.to(dist_util.dev()),
                        progress=True,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'is_transition': model_kwargs['y']['is_transition'][bs_i]
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer



class CompMDMUnfoldingGeneratedDataset2(Dataset):
    def __init__(self, args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1., num_unfoldings=10):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length

        self.num_steps_to_generate = num_unfoldings

        # Will be changed later by the evaluation script for each copy of this dataset
        self.step_to_eval = 1
        self.transition = False

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()

        n_frames = 100  # FIXME - using fixed length

        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                # tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):

                    sample = []
                    all_texts = []
                    all_lengths = []
                    all_tokens = []

                    for bs_i in range(dataloader.batch_size):
                        _tokens = [model_kwargs['y']['tokens'][bs_i - ii].split('_') for ii in
                                   reversed(range(self.num_steps_to_generate))]
                        texts = [model_kwargs['y']['text'][bs_i - ii] for ii in
                                 reversed(range(self.num_steps_to_generate))]
                        lengths = [n_frames - args.handshake_size] * self.num_steps_to_generate
                        # lengths = [model_kwargs['y']['lengths'][bs_i - ii] for ii in
                        #            reversed(range(self.num_steps_to_generate))]
                        all_texts.append(texts)
                        all_lengths.append(lengths)
                        all_tokens.append(_tokens)

                    new_model_kwargs = {
                        'y': {
                            'text': list(itertools.chain.from_iterable(all_texts)),
                            'lengths': list(itertools.chain.from_iterable(all_lengths)),
                            # TODO - support more than one sample in batch
                        }
                    }
                    new_batch_size = len(new_model_kwargs['y']['text'])

                    transition = torch.zeros(n_frames)
                    transition[:args.handshake_size] = 1.
                    transition[-args.handshake_size:] = 1.
                    transition = torch.tile(transition.unsqueeze(0), dims=(new_batch_size, 1))
                    transition[0, :args.handshake_size] = 0
                    transition[-1, -args.handshake_size:] = 0
                    new_model_kwargs['y']['is_transition'] = transition

                    # add CFG scale to batch
                    if scale != 1.:
                        new_model_kwargs['y']['scale'] = torch.ones(new_batch_size, device=dist_util.dev()) * scale
                    samples_per_rep_list, samples_type = double_take_arb_len(args, diffusion, model, new_model_kwargs,
                                                                     n_frames=n_frames, eval_mode=True) # TODO: check if possible using Doubletake arblen instead
                    all_samples = samples_per_rep_list[0]  # we only do one rep
                    sample = [[all_samples[bs_i*self.num_steps_to_generate + step_i, :, :, args.handshake_size:].squeeze().permute(1,0).cpu().numpy() for step_i in range(self.num_steps_to_generate)] for bs_i in range(dataloader.batch_size)]

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i],
                                    'length': all_lengths[bs_i],
                                    'caption': all_texts[bs_i],
                                    'tokens': all_tokens[bs_i],
                                    'cap_len': [len(e) for e in all_tokens[bs_i]],
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i],
                                        'length': all_lengths[bs_i],
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': all_texts[bs_i],
                                    'tokens': all_tokens[bs_i],
                                    'cap_len': [len(e) for e in all_tokens[bs_i]],
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'][self.step_to_eval], data['length'][self.step_to_eval], data['caption'][self.step_to_eval], data['tokens'][self.step_to_eval]
        sent_len = data['cap_len'][self.step_to_eval]

        if self.transition:
            max_tran_length = 40
            prev_motion = data['motion'][self.step_to_eval-1]
            cur_motion = data['motion'][self.step_to_eval]

            prev_motion_end = data['length'][self.step_to_eval-1]
            prev_motion_start = max(prev_motion_end - (max_tran_length//2), 0)

            cur_motion_start = 0
            cur_motion_end = min(max_tran_length // 2, data['length'][self.step_to_eval])

            # print(f'prev motion [{prev_motion_start}:{prev_motion_end}]')
            # print(f'cur motion [{cur_motion_start}:{cur_motion_end}]')

            motion = np.concatenate([prev_motion[prev_motion_start:prev_motion_end],
                                     cur_motion[cur_motion_start:cur_motion_end]], axis=0)
            m_length = motion.shape[0]
            # print(f'transition length [{motion.shape[0]}], max is [{max_tran_length}]')
            pad = np.zeros((self.max_motion_length - motion.shape[0], prev_motion.shape[1]), dtype=prev_motion.dtype)
            motion = np.concatenate([motion, pad], axis=0)
            assert motion.shape[0] == self.max_motion_length, f'motion.shape[0]={motion.shape[0]}'


        if self.dataset.load_mode == 'eval' and self.dataset.dataset_name != 'babel':  # re-norm is not needed for babel dataset
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), []

import  numpy as np
def pad_sample_with_zeros(sample, max_len=250):
    # pad inp, change lenghts, and pad is transition
    seq_len, n_feats = sample.shape
    len_to_pad = max_len - seq_len
    np.zeros_like(sample)
    sample_padding = np.zeros((len_to_pad, n_feats))
    sample = np.concatenate((sample, sample_padding))
    return sample

class CompMDMUnfoldingGeneratedDataset(Dataset):
    def __init__(self, args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1., num_unfoldings=10):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False
        clip_denoised = False
        self.max_motion_length = max_motion_length

        self.num_steps_to_generate = num_unfoldings

        # Will be changed later by the evaluation script for each copy of this dataset
        self.step_to_eval = 1
        self.transition = False

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()


        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                # max_arb_len = model_kwargs['y']['lengths'].max() #+ 2 * args.handshake_size
                # min_arb_len = model_kwargs['y']['lengths'].min()
                #
                # # assert (min_arb_len > 2 * args.blend_len)
                #
                # for ii, len_s in enumerate(model_kwargs['y']['lengths']):
                #     # model_kwargs['y']['lengths'][ii] += 2 * args.handshake_size
                #     if args.dataset == 'humanml':
                #         model_kwargs['y']['lengths'][ii] = min(
                #             model_kwargs['y']['lengths'][ii] + 2 * args.handshake_size, 196)
                #     if args.dataset =='babel':
                #         model_kwargs['y']['lengths'][ii] = min(
                #             model_kwargs['y']['lengths'][ii] + 2 * args.handshake_size, 250)

                    # model_kwargs['y']['lengths'][ii] = min(model_kwargs['y']['lengths'][ii] + 2 * args.handshake_size, 196 if args.dataset == 'humanml' else 250)
                    
                # model_kwargs['y']['lengths'][0] -= args.handshake_size #first and last.
                # model_kwargs['y']['lengths'][-1] -= args.handshake_size


                # Old version:
                max_arb_len = model_kwargs['y']['lengths'].max()
                min_arb_len = 2 * args.handshake_size + 2 * args.blend_len + 10

                for ii, len_s in enumerate(model_kwargs['y']['lengths']):
                    if len_s > max_arb_len:
                        model_kwargs['y']['lengths'][ii] = max_arb_len
                    if len_s < min_arb_len:
                        model_kwargs['y']['lengths'][ii] = min_arb_len
                max_arb_len = model_kwargs['y']['lengths'].max() #+ 2 * args.handshake_size

                n_frames = max_arb_len

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                # tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):

                    sample = []
                    all_texts = []
                    all_lengths = []
                    all_tokens = []

                    batch_size = len(model_kwargs['y']['text'])

                    if scale != 1.:
                        model_kwargs['y']['scale'] = torch.ones(batch_size, device=dist_util.dev()) * scale
                    samples_per_rep_list, samples_type = double_take_arb_len(args, diffusion, model, model_kwargs,
                                                                     n_frames=n_frames, eval_mode=True)
                    # if args.double_take:
                    #     all_samples = samples_per_rep_list[1]  # we only do one rep
                    # else:
                    #     all_samples = samples_per_rep_list[0]  # we only do one rep
                    all_samples = samples_per_rep_list[0]  # we only do one rep
                    sample = all_samples
                    step_sizes = np.zeros(len(model_kwargs['y']['lengths']), dtype=int)
                    for ii, len_i in enumerate(model_kwargs['y']['lengths']):
                        if ii == 0:
                            step_sizes[ii] = len_i
                            continue
                        step_sizes[ii] = step_sizes[ii - 1] + len_i - args.handshake_size
                    final_n_frames = step_sizes[-1]
                    unfolded = unfold_sample_arb_len(sample, args.handshake_size, step_sizes, final_n_frames, model_kwargs)

                    tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                    max_motion_length = int(model_kwargs['y']['lengths'].max())

                    if t == 0:
                        if args.eval_on == "motion":
                            sub_dicts = [{
                                        # eval on seq
                                        'motion': pad_sample_with_zeros(unfolded[..., step_sizes[bs_i] - model_kwargs['y']['lengths'][bs_i] + args.handshake_size:
                                                                step_sizes[bs_i] - args.handshake_size].squeeze().permute(1, 0).cpu().numpy(), max_motion_length),
                                        'length': model_kwargs['y']['lengths'][bs_i] - 2*args.handshake_size,
                                        'caption': model_kwargs['y']['text'][bs_i],
                                        'tokens': tokens[bs_i],
                                        'cap_len': len(tokens[bs_i]),
                                        'is_transition': model_kwargs['y']['is_transition'][bs_i][:args.handshake_size]
                                        } for bs_i in range(1, dataloader.batch_size - 1)] #-1)] uncomment the -1 for transitions
                            sub_dicts += [{
                                'motion': pad_sample_with_zeros(unfolded[..., :step_sizes[0] - args.handshake_size].squeeze().permute(1, 0).cpu().numpy(), max_motion_length),
                                'length': model_kwargs['y']['lengths'][0] - args.handshake_size,
                                'caption': model_kwargs['y']['text'][0],
                                'tokens': tokens[0],
                                'cap_len': len(tokens[0]),
                                'is_transition': model_kwargs['y']['is_transition'][0][:args.handshake_size]
                            }]
                            sub_dicts += [{
                                'motion': pad_sample_with_zeros(unfolded[..., step_sizes[-1] - model_kwargs['y']['lengths'][-1] + args.handshake_size:
                                                                              ].squeeze().permute(1, 0).cpu().numpy(), max_motion_length),
                                'length': model_kwargs['y']['lengths'][-1] - args.handshake_size,
                                'caption': model_kwargs['y']['text'][-1],
                                'tokens': tokens[-1],
                                'cap_len': len(tokens[-1]),
                                'is_transition': model_kwargs['y']['is_transition'][-1][:args.handshake_size]
                            }]
                        elif args.eval_on == "transition":
                            sub_dicts = [{
                                        'motion': unfolded[..., step_sizes[bs_i]-args.handshake_size-(args.transition_margins//2):
                                                                step_sizes[bs_i]+(args.transition_margins//2)].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': args.handshake_size + args.transition_margins,
                                        'caption': model_kwargs['y']['text'][bs_i],
                                        'tokens': tokens[bs_i],
                                        'cap_len': len(tokens[bs_i]),
                                        'is_transition': model_kwargs['y']['is_transition'][bs_i][:args.handshake_size]
                                        } for bs_i in range(0, dataloader.batch_size - 1)] #uncomment the -1 for transitions
                        else:
                            print("Error")
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i],
                                        'length': all_lengths[bs_i],
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': all_texts[bs_i],
                                    'tokens': all_tokens[bs_i],
                                    'cap_len': [len(e) for e in all_tokens[bs_i]],
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens, transitions = data['motion'], data['length'], data['caption'], data['tokens'], data['is_transition']
        sent_len = data['cap_len']

        if self.dataset.load_mode == 'eval' and self.dataset.dataset_name != 'babel':  # re-norm is not needed for babel dataset
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), []#np.zeros(1)
