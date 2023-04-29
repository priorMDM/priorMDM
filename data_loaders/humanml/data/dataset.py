import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy
import itertools

from torch.utils.data._utils.collate import default_collate
from data_loaders.amass.sampling import FrameSampler
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt

from data_loaders.amass.babel import BABEL


MOTION_TYPES = [
    '_0',
    '_1',
    '_0_with_transition',
    '_1_with_transition',
]

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text': [text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text':text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(['single', 'single', 'double'])
                else:
                    coin2 = 'single'
                if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx+self.max_length]
                else:
                    if coin2 == 'single':
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'

            if coin2 == 'double':
                m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length


'''For use of training text motion matching model, and evaluations'''
class Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, num_frames, size=None, **kwargs):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.num_frames = num_frames if num_frames else False
        self.max_motion_length = opt.max_motion_length
        if (self.num_frames == False) or type(self.num_frames)==int:
            min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        else:
            min_motion_len = self.num_frames[0]
            self.max_motion_length = self.num_frames[1]
        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        id_list = id_list[:size]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                if self.num_frames != False:
                                    if len(n_motion) >= self.max_motion_length:
                                        bias = random.randint(0, len(n_motion) - self.max_motion_length)
                                        data_dict[new_name] = {'motion': n_motion[bias: bias+self.max_motion_length],
                                                               'length': self.max_motion_length,
                                                               'text': [text_dict]}
                                        length_list.append(self.max_motion_length)

                                    else:
                                        data_dict[new_name] = {'motion': n_motion,
                                                               'length': len(n_motion),
                                                               'text': [text_dict]}
                                        length_list.append(len(n_motion))

                                else:
                                    data_dict[new_name] = {'motion': n_motion,
                                                           'length': len(n_motion),
                                                           'text':[text_dict]}
                                    length_list.append(len(n_motion))

                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    if self.num_frames != False:
                        if len(motion) >= self.max_motion_length:
                            bias = random.randint(0, len(motion) - self.max_motion_length)
                            data_dict[name] = {'motion': motion[bias: bias + self.max_motion_length],
                                                   'length': self.max_motion_length,
                                                   'text': [text_dict]}
                            length_list.append(self.max_motion_length)

                        else:
                            data_dict[name] = {'motion': motion,
                                               'length': len(motion),
                                               'text': text_data}
                            length_list.append(len(motion))

                    else:
                        data_dict[name] = {'motion': motion,
                                           'length': len(motion),
                                           'text': text_data}
                        length_list.append(len(motion))

                    new_name_list.append(name)
            except Exception as e:
                print(e)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        # FIXME: I removed the extra return value ([]) at the end
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), []



'''For use of training text motion matching model, and evaluations'''
class PW3D_Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, splits, w_vectorizer, **kwargs):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.min_motion_len = 80 if self.opt.dataset_name =='t2m' else 24
        self.canon_relevant_entries = [0, 2, 6, 8]

        data_dict = {}
        base_dir = './dataset/3dpw/new_joint_vecs'
        splits = splits.split(',')
        group_path = [[pjoin(base_dir, s, f) for f in os.listdir(pjoin(base_dir, s)) if (f.endswith('_p0.npy') or f.endswith('_p0_M.npy'))] for s in splits]
        id_list = list(itertools.chain.from_iterable(group_path))
        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            name0 = name
            name1 = name.replace('_p0', '_p1')
            motion0 = np.load(name0)
            motion1 = np.load(name1)
            def get_canon(_name):
                _canon = np.load(_name.replace('new_joint_vecs', 'canon_data'))[self.canon_relevant_entries] / 10.
                return np.concatenate((_canon, np.zeros(self.opt.dim_pose-len(self.canon_relevant_entries))))
            canon0 = get_canon(name0)
            canon1 = get_canon(name1)
            if not 'test' in splits:  # test is not yet annotated
                with open(name0.replace('new_joint_vecs', 'text').replace('p0_M', 'p0').replace('.npy', '.txt'), 'r') as fr:
                    texts = [t.strip() for t in fr.readlines()]
            else:
                texts = [''] * 5
            assert len(texts) == 5
            text_data = [{'caption': texts, 'tokens':['sos/OTHER', 'eos/OTHER']}]  # FIXME - parse tokens
            new_name = os.path.basename(name).replace('.npy', '')
            new_name_list.append(new_name)
            assert len(motion0) == len(motion1)
            length_list.append(len(motion0))
            # print('canon0', canon0)
            # print('canon1', canon1)
            data_dict[new_name] = {'motion0': motion0, 'motion1': motion1,
                               'length0': len(motion0), 'length1': len(motion1),
                               'text': text_data, 'canon0': canon0, 'canon1': canon1,}

        # replicate data for beter utilization
        n_replications = 1000
        rep_data_dict = {}
        rep_name_list = []
        rep_length_list = []
        for rep_i in range(n_replications):
            rep_name_list += [e+'_{:04d}'.format(rep_i) for e in new_name_list]
            rep_length_list += length_list
            rep_data_dict.update({
                k+'_{:04d}'.format(rep_i): v for k, v in data_dict.items()
            })


        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        name_list, length_list = zip(*sorted(zip(rep_name_list, rep_length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = rep_data_dict  # data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def rebuilt_canon(self, canon_pred):
        # asuumes canon data in the first 4 entries of the last dim
        # canon_pred *= 10.
        canon_pred_scaled = canon_pred * 10.
        first_entry = canon_pred_scaled[..., [0]]
        zeros = torch.zeros_like(first_entry)
        ones = torch.ones_like(first_entry)
        return torch.cat((canon_pred_scaled[..., [0]], zeros, canon_pred_scaled[..., [1]], zeros, ones, zeros,
                          canon_pred_scaled[..., [2]], zeros, canon_pred_scaled[..., [3]]), dim=-1)

    def inv_transform(self, data):
        return data * self.std + self.mean

    def inv_transform_torch(self, data):
        return data * torch.tensor(self.std, dtype=data.dtype, device=data.device) + torch.tensor(self.mean, dtype=data.dtype, device=data.device)

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        person_i = random.randint(0,1)
        motion, m_length, text_list, canon = data[f'motion{person_i}'], data[f'length{person_i}'], data['text'], data[f'canon{person_i}']
        other_motion = data[f'motion{1-person_i}']
        other_canon = data[f'canon{1-person_i}']

        orig_length = m_length
        if self.opt.load_mode == 'prefix':
            m_length = 80
        elif self.opt.load_mode == 'text':
            m_length = random.randint(self.min_motion_len, min(self.max_motion_length, orig_length))
        offset = random.randint(0, orig_length-m_length-1)
        motion = motion[offset:offset+m_length]
        other_motion = other_motion[offset:offset+m_length]

        # Randomly select a caption
        assert len(text_list) == 1
        text_data = text_list[0]
        caption, tokens = random.choice(text_data['caption']), text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.load_mode == 'text':
            # Crop the motions in to times of 4, and introduce small variations
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'

            if coin2 == 'double':
                m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]
            other_motion = other_motion[idx:idx+m_length]


        "Z Normalization"
        motion = (motion - self.mean) / self.std
        other_motion = (other_motion - self.mean) / self.std

        if self.opt.load_mode == 'text':
            if m_length < self.max_motion_length:
                motion = np.concatenate([motion,
                                         np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                         ], axis=0)
                other_motion = np.concatenate([other_motion,
                                         np.zeros((self.max_motion_length - m_length, other_motion.shape[1]))
                                         ], axis=0)

        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        # return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

        # Concat canon data
        motion = np.concatenate((canon[None], motion), axis=0)
        other_motion = np.concatenate((other_canon[None], other_motion), axis=0)
        m_length += 1 # since we added the cannon in the begining

        return other_motion, pos_one_hots, caption, person_i, motion, m_length, '_'.join(tokens), []


'''For training BABEL text2motion evaluators'''
class BABEL_Text2MotionDatasetV2(BABEL):

    def __init__(self, split, datapath, transforms, opt, mean, std, w_vectorizer, sampler, mode, **kwargs):
        BABEL.__init__(self, datapath=datapath, transforms=transforms, split=split, sampler=sampler,
                       parse_tokens=True, mode=mode, short_db=kwargs.get('short_db', False),
                       cropping_sampler=kwargs.get('cropping_sampler', False))  # tokens are needed for training
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.mean = mean
        self.std = std

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __getitem__(self, item):
        keyid = self._split_index[item]
        batch = self.load_keyid(keyid, mode='train')

        # Randomly choose a motion from batch
        caption = batch['text']
        tokens = batch['tokens']
        motion = batch['features']
        m_length = batch['length']
        tansition_seq = batch['is_transition']


        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length

        idx = random.randint(0, abs(len(motion) - m_length))

        motion = motion[idx:idx+m_length]
        tansition_seq = tansition_seq[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length <= self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
            tansition_seq = np.concatenate([tansition_seq,
                                     np.zeros(self.max_motion_length - m_length)
                                     ])
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), tansition_seq


'''For use of training baseline'''
class Text2MotionDatasetBaseline(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'
            if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == 'single':
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx: s_idx + m_length]
        tgt_motion = motion[s_idx: s_idx + self.max_length]

        "Z Normalization"
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            src_motion = np.concatenate([src_motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(m_length, src_motion.shape, tgt_motion.shape)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


class MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx+self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion

'''For training BABEL movement encoder (used by evaluators)'''
class BABEL_MotionDatasetV2(BABEL):

    def __init__(self, split, datapath, transforms, opt, mean, std, sampler, mode, **kwargs):
        BABEL.__init__(self, datapath=datapath, transforms=transforms, split=split,  sampler=sampler,
                       parse_tokens=False, mode=mode, **kwargs)  # Tokens are not needed
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.mean = mean
        self.std = std

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __getitem__(self, item):

        keyid = self._split_index[item]
        batch = self.load_keyid(keyid, mode='train')
        motion_type = MOTION_TYPES[random.randint(0, len(MOTION_TYPES)-1)]
        motion = batch['features' + motion_type]
        m_length = batch['length' + motion_type]
        assert m_length >= self.opt.window_size

        idx = random.randint(0, m_length - self.opt.window_size)
        _motion = motion[idx:idx + self.opt.window_size]

        "Z Normalization"
        _motion = (_motion - self.mean) / self.std

        return _motion

class RawTextDataset(data.Dataset):
    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load('en_core_web_sm')

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = ['%s/%s'%(word_list[i], pos_list[i]) for i in range(len(word_list))]
                self.data_dict.append({'caption':line.strip(), "tokens":tokens})

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))


    def process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data['caption'], data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len

class TextOnlyDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, size=None, **kwargs):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120


        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        id_list = id_list[:size]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'text':[text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'text': text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        return None, None, caption, None, np.array([0]), self.fixed_length, None
        # fixed_length can be set from outside before sampling

# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):
    def __init__(self, load_mode, datapath='./dataset/humanml_opt.txt', split="train", **kwargs):
        self.load_mode = load_mode
        
        self.dataset_name = 't2m'
        self.dataname = 't2m'
        self.split = split

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        opt.load_mode = load_mode
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        if load_mode == 'gt':
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif load_mode in ['train', 'eval', 'text_only', 'prefix', 'text']:
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        if load_mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        if load_mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file, **kwargs)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')

            if hasattr(opt, 'dataset_type') and opt.dataset_type == 'pw3d':
                self.t2m_dataset = PW3D_Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split,
                                                             self.w_vectorizer)
            else:
                self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, **kwargs)
            self.num_actions = 1 # dummy placeholder

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

# A wrapper class for t2m original dataset for MDM purposes
class KIT(HumanML3D):
    def __init__(self, load_mode, datapath='./dataset/kit_opt.txt', split="train", **kwargs):
        super(KIT, self).__init__(load_mode, datapath, split, **kwargs)

# A wrapper class for t2m original dataset for MDM purposes
class PW3D(HumanML3D):
    def __init__(self, load_mode, datapath='./dataset/pw3d_opt.txt', split="train", **kwargs):
        super(PW3D, self).__init__(load_mode, datapath, split, **kwargs)


# A wrapper class for t2m original dataset for MDM purposes
class BABEL_eval(data.Dataset):
    def __init__(self, load_mode, datapath, transforms, sampler, mode, opt, split="train", **kwargs):
        self.load_mode = load_mode

        self.split = split
        self.datapath = datapath
        abs_base_path = f'.'

        if opt is None:
            self.opt_path = './dataset/humanml_opt.txt'
            # Configurations of T2M dataset and KIT dataset is almost the same
            dataset_opt_path = pjoin(abs_base_path, self.opt_path)
            device = None  # torch.device('cuda:4') # This param is not in use in this context
            opt = get_opt(dataset_opt_path, device)
            opt.data_root = pjoin('dataset', 'babel')
            opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
            opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
            opt.text_dir = pjoin(abs_base_path, opt.text_dir)
            opt.model_dir = None
            opt.checkpoints_dir = '.'
            opt.data_root = pjoin(abs_base_path, opt.data_root)
            opt.save_root = pjoin(abs_base_path, opt.save_root)
            opt.meta_dir = './dataset'
            opt.dim_pose = 135
            opt.foot_contact_entries = 0
            opt.dataset_name = 'babel'
            opt.decomp_name = 'Decomp_SP001_SM001_H512_babel_2700epoch'
            opt.meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, 'motion1', 'meta')
            opt.min_motion_length = sampler.min_len # must be at least window size
            opt.max_motion_length = sampler.max_len
        self.opt = opt

        print('Loading dataset %s ...' % opt.dataset_name)

        self.dataset_name = opt.dataset_name
        self.dataname = opt.dataset_name
        self.sampler = sampler
        self.transforms = transforms
        self.mean = np.zeros([opt.dim_pose], dtype=np.float32)  # data is already normalized
        self.std = np.ones([opt.dim_pose], dtype=np.float32)  # data is already normalized

        DATA = BABEL_MotionDatasetV2 if load_mode == 'movement_train' else BABEL_Text2MotionDatasetV2

        self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
        self.t2m_dataset = DATA(
            split=self.split,
            datapath=self.datapath,
            transforms=self.transforms,
            mode=mode,
            opt=self.opt,
            mean=self.mean, std=self.std, w_vectorizer=self.w_vectorizer, sampler=self.sampler,
            short_db=kwargs.get('short_db', False), cropping_sampler=kwargs.get('cropping_sampler', False)
        )
        self.num_actions = 1  # dummy placeholder

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()