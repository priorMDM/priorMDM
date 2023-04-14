# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import json
from operator import itemgetter
import os
from glob import glob
from platform import architecture
from re import A
from typing import Dict, List, Optional, Tuple
import logging
import joblib

import numpy as np
import pandas
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import random


from .file_io import read_json
from .nlp_consts import fix_spell
from .transforms import Transform

logger = logging.getLogger(__name__)

SPLITS = ["train", "val", "test", "all", "subset"]
EXCLUDED_ACTIONS = ['t-pose', 'a-pose', 'a pose', 't pose',
                    'tpose', 'apose', 'transition']
EXCLUDED_ACTIONS_WO_TR = ['t-pose', 'a-pose', 'a pose', 't pose', 'tpose', 'apose']

import spacy
nlp = spacy.load('en_core_web_sm')
# Tokenizer according to https://github.com/EricGuo5513/HumanML3D/blob/main/text_process.py
def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
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


def get_split(path: str, split: str, subset: Optional[str] = ''):
    assert split in SPLITS
    filepath = Path(path) / f'{split}{subset}.pth.tar'
    split_data = joblib.load(filepath)
    return split_data


def get_babel_keys(path: str):
    filepath = Path(path) / f'../babel-teach/id2fname/amass-path2babel.json'
    amass2babel = read_json(filepath)
    return amass2babel


def separate_actions(pair: Tuple[Tuple]):
    if len(pair) == 3:
        if pair[0][1] < pair[2][0]:
            # a1 10, 15 t 14, 18 a2 17, 25
            # a1 10, 15 t 16, 16 a2 17, 25
            # transition only --> transition does not matter
            final_pair = [(pair[0][0], pair[1][0]),
                          (pair[1][0] + 1, pair[2][0] - 1),
                          (pair[2][0], pair[2][1])]
        else:
            # overlap + transition --> transition does not matter
            over = pair[2][0] - pair[0][1]
            final_pair = [(pair[0][0], int(pair[0][1] + over / 2)),
                          (int(pair[0][1] + over / 2 + 1), pair[2][1])]
    else:
        # give based on small or long
        # p1_prop = (pair[0][1] - pair[0][0]) / (eoq - soq)
        # p2_prop = (pair[1][1] - pair[1][0]) / (eoq - soq)
        # over = pair[1][0] - pair[0][1]
        # final_pair = [(pair[0][0], int(p1_prop*over) + pair[0][1]),
        #               (int(p1_prop*over) + pair[0][1] + 1, pair[1][1])]

        # no transition at all
        over = abs(pair[0][1] - pair[1][0])
        # final_pair = [(pair[0][0], pair[0][1]), (pair[1][0], pair[1][0]+1), (pair[1][0] ,pair[1][1])]
        # final_pair = [(pair[0][0], pair[0][1]), sorted((pair[0][1], pair[1][0])), (pair[1][0], pair[1][1])]
        # final_pair = [(pair[0][0], int(pair[0][1] + over / 2)),
        #               (int(pair[0][1] + over / 2 + 1), pair[1][1])]
        final_pair = [(pair[0][0],pair[0][1]),
                      (pair[1][0], pair[1][1])]

    return final_pair


def timeline_overlaps(arr1: Tuple, arr2: List[Tuple]) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    '''
    Returns the intervals for which:
    (1) arr1 has overlap with
    (2) arr1 is a subset of
    (3) arr1 is a superset of
    '''
    l = arr1[0]
    r = arr1[1]
    inter_sub = []
    inter_super = []
    inter_before = []
    inter_after = []
    for s in arr2:

        if (s[0] > l and s[0] > r) or (s[1] < l and s[1] < r):
            continue
        if s[0] <= l and s[1] >= r:
            inter_sub.append(s)
        if s[0] >= l and s[1] <= r:
            inter_super.append(s)
        if s[0] < l and s[1] < r and s[1] >= l:
            inter_before.append(s)
        if s[0] > l and s[0] <= r and s[1] > r:
            inter_after.append(s)

    return inter_before, inter_after


def segments_sorted(segs_fr: List[List], acts: List) -> Tuple[List[List], List]:
    assert len(segs_fr) == len(acts)
    if len(segs_fr) == 1: return segs_fr, acts
    L = [(segs_fr[i], i) for i in range(len(segs_fr))]
    L.sort()
    sorted_segs_fr, permutation = zip(*L)
    sort_acts = [acts[i] for i in permutation]

    return list(sorted_segs_fr), sort_acts


def extract_frame_labels(babel_labels, fps, seqlen, datatype):
    seg_ids = []
    seg_acts = []
    is_valid = True
    babel_key = babel_labels['babel_sid']
    if datatype == 'seq' and babel_labels['frame_ann'] is not None:
        is_valid = False
    if datatype == 'seg' and babel_labels['frame_ann'] is None:
        is_valid = False
    if datatype == 'pairs_only' and babel_labels['frame_ann'] is None:
        is_valid = False
    if datatype == 'separate_pairs' and babel_labels['frame_ann'] is None:
        is_valid = False

    if is_valid:
        if babel_labels['frame_ann'] is None:

            # 'transl' 'pose''betas'
            action_label = babel_labels['seq_ann']['labels'][0]['proc_label']
            seg_ids.append([0, seqlen])
            seg_acts.append(fix_spell(action_label))
        else:
            # Get segments
            for seg_an in babel_labels['frame_ann']['labels']:
                action_label = fix_spell(seg_an['proc_label'])

                st_f = int(seg_an['start_t'] * fps)
                end_f = int(seg_an['end_t'] * fps)
                if end_f > seqlen:
                    end_f = seqlen
                seg_ids.append((st_f, end_f))
                seg_acts.append(action_label)
            # Process segments
            assert len(seg_ids) == len(seg_acts)
            # TODO make this pairs and then a subconfig for pairs
            if datatype == 'pairs' or datatype == 'pairs_only' or datatype == 'separate_pairs':
                import itertools

                seg_ids, seg_acts = segments_sorted(seg_ids, seg_acts)

                # remove a/t pose for pair calculation
                seg_acts_for_pairs = [a for a in seg_acts if a not in EXCLUDED_ACTIONS_WO_TR]
                idx_to_keep = [i for i, a in enumerate(seg_acts) if a not in EXCLUDED_ACTIONS_WO_TR]
                seg_ids_for_pairs = [s for i, s in enumerate(seg_ids) if i in idx_to_keep]
                assert len(seg_acts_for_pairs) == len(seg_ids_for_pairs)

                seg2act = dict(zip(seg_ids_for_pairs, seg_acts_for_pairs))
                # plot_timeline(seg_ids, seg_acts, babel_key)

                overlaps_for_each_seg = {}
                for idx, segment in enumerate(seg_ids_for_pairs):
                    # remove the segment of interest
                    seg_ids_wo_seg = [x for x in seg_ids_for_pairs if x != segment]
                    # calculate the before and after overlaps for the segment of interest
                    ov_bef, ov_aft = timeline_overlaps(segment, seg_ids_wo_seg)

                    overlaps_for_each_seg[segment] = {}
                    overlaps_for_each_seg[segment]['before'] = ov_bef
                    overlaps_for_each_seg[segment]['after'] = ov_aft

                pairs_s = []
                pairs_a = []
                for seg_, ov_seg in overlaps_for_each_seg.items():
                    cur_act_pairs = []
                    cur_seg_pairs = []
                    cur_seg_pairs_bef = []
                    cur_seg_pairs_af = []
                    if seg2act[seg_] == 'transition':
                        # if transition is not the start
                        if not seg_[0] == 0:
                            if ov_seg['before'] and ov_seg['after']:
                                cur_seg_pairs = list(itertools.product(ov_seg['before'], ov_seg['after']))
                                cur_act_pairs = [(seg2act[x], seg2act[y]) for x, y in cur_seg_pairs]
                                if not datatype == 'separate_pairs':
                                    cur_seg_pairs = [(min(min(a), min(b), min(seg_)),
                                                      max(max(a), max(b), max(seg_))) for a, b in cur_seg_pairs]
                                else:
                                    cur_seg_pairs = [tuple(sorted(p, key=lambda item: item[0])) for p in cur_seg_pairs]
                                    cur_seg_pairs = [(a, seg_, b) for a, b in cur_seg_pairs]

                                pairs_s.append(cur_seg_pairs)
                                pairs_a.append(cur_act_pairs)

                    else:
                        ov_seg['before'] = [x for x in ov_seg['before'] if seg2act[x] != 'transition']
                        ov_seg['after'] = [x for x in ov_seg['after'] if seg2act[x] != 'transition']
                        if ov_seg['before']:
                            cur_seg_pairs_bef = list(itertools.product(ov_seg['before'], [seg_]))
                        if ov_seg['after']:
                            cur_seg_pairs_af = list(itertools.product([seg_], ov_seg['after']))

                        if ov_seg['after'] and ov_seg['before']:
                            cur_seg_pairs = cur_seg_pairs_bef + cur_seg_pairs_af
                        elif ov_seg['after']:
                            cur_seg_pairs = cur_seg_pairs_af
                        elif ov_seg['before']:
                            cur_seg_pairs = cur_seg_pairs_bef
                        else:
                            continue

                        if not datatype == 'separate_pairs':
                            cur_act_pairs = [(seg2act[x], seg2act[y]) for x, y in cur_seg_pairs]
                            cur_seg_pairs = [(min(min(a), min(b)), max(max(a), max(b))) for a, b in cur_seg_pairs]
                        else:
                            # just to be sure
                            cur_seg_pairs = [tuple(sorted(p, key=lambda item: item[0])) for p in cur_seg_pairs]
                            cur_act_pairs = [(seg2act[x], seg2act[y]) for x, y in cur_seg_pairs]

                        # separate_pairs
                        # [((),())]
                        # list of tuples for everything except separate_pairs
                        pairs_s.append(cur_seg_pairs)
                        pairs_a.append(cur_act_pairs)

                # flatten list of lists
                pairs_s = list(itertools.chain(*pairs_s))
                pairs_a = list(itertools.chain(*pairs_a))

                # remove duplicates
                from more_itertools import unique_everseen

                tmp = zip(pairs_s, pairs_a)
                uniq_tmp = unique_everseen(tmp, key=itemgetter(0))
                segment_pairs = []
                action_pairs = []
                for seg, a in list(uniq_tmp):
                    segment_pairs.append(seg)
                    action_pairs.append(a)

                assert len(segment_pairs) == len(action_pairs)
                if not datatype == 'separate_pairs':
                    # conversion of actions to pair with comma
                    action_pairs = [f'{a1}, {a2}' for a1, a2 in action_pairs]
                if datatype == 'pairs':
                    seg_ids.extend(segment_pairs)
                    seg_acts.extend(action_pairs)
                elif datatype == 'pairs_only' or datatype == 'separate_pairs':
                    if segment_pairs:
                        is_valid = True
                        return segment_pairs, action_pairs, is_valid
                    else:
                        is_valid = False
                        return segment_pairs, action_pairs, is_valid
    return seg_ids, seg_acts, is_valid


def get_all_seq_durs(duration, max_len, min_len, crop_samples):
    if crop_samples:
        dur1, dur_tr, dur2 = duration
        dur1_t = dur1 + dur_tr if dur1 > min_len else None
        dur2_t = dur2 + dur_tr if dur2 > min_len else None

        durations = {"dur1": dur1 if dur1 > min_len else None,
                     "dur2": dur2 if dur2 > min_len else None,
                     "dur1_t": dur1_t,
                     "dur2_t": dur2_t,
                     "bias_dur1": 0,
                     "bias_dur2": 0,
                     "bias_dur1_t": 0,
                     "bias_dur2_t": 0}

        for dur_key, dur in durations.items():
            if "bias" in dur_key:
                continue
            if dur != None:
                if dur > max_len:
                    durations[f'bias_{dur_key}'] = random.randint(0, dur - max_len)
                    durations[dur_key] = max_len
    else:
        dur1, dur_tr, dur2 = duration
        dur1_t = dur1 + dur_tr if ((dur1 >= min_len) and ((dur1 + dur_tr) <= (max_len + 4))) else None
        dur2_t = dur2 + dur_tr if ((dur2 >= min_len) and ((dur2 + dur_tr) <= (max_len + 4))) else None
        durations = {"dur1": dur1 if ((dur1 >= min_len) and (dur1 <= (max_len + 4))) else None,
                     "dur2": dur2 if ((dur2 >= min_len) and (dur2 <= (max_len + 4))) else None,
                     "dur1_t": dur1_t,
                     "dur2_t": dur2_t,
                     "bias_dur1": 0,
                     "bias_dur2": 0,
                     "bias_dur1_t": 0,
                     "bias_dur2_t": 0}
        for dur_key, dur in durations.items():
            if "bias" in dur_key:
                continue
            if dur != None:
                if dur > max_len:
                    bias = random.randint(dur - max_len, 4)
                    durations[f'bias_{dur_key}'] = bias
                    durations[dur_key] = dur - bias
                else:
                    bias = random.randint(0, 4)
                    durations[f'bias_{dur_key}'] = bias
                    durations[dur_key] = dur - bias

    return durations

class BABEL(Dataset):
    dataname = "BABEL"

    def __init__(self, datapath: str,
                 transforms: Transform,
                 split: str = "train",
                 transforms_xyz: Optional[Transform] = None,
                 transforms_smpl: Optional[Transform] = None,
                 sampler=None,
                 progress_bar: bool = True,
                 load_with_rot=True,
                 downsample=True,
                 tiny: bool = False,
                 walk_only: Optional[bool] = False,
                 kit_only: Optional[bool] = False,
                 dtype: str = 'separate_pairs',
                 mode: str = 'train',
                 parse_tokens: bool = False,
                 **kwargs):

        self.split = split
        self.parse_tokens = parse_tokens
        self.load_with_rot = load_with_rot
        self.downsample = downsample
        self.dtype = dtype  # seg or seq or empty string for segments or sequences
        # or all of the stuff --> 'seg', 'seq', 'pairs', 'pairs_only', ''
        self.walk_only = walk_only
        self.kit_only = kit_only
        if not self.load_with_rot:
            self.transforms_xyz = transforms_xyz
            self.transforms_smpl = transforms_smpl
            self.transforms = transforms_xyz
        else:
            self.transforms = transforms
        if self.split not in SPLITS:
            raise ValueError(f"{self.split} is not a valid split")

        assert sampler is not None, 'Must inject sampler via constructor'
        self.sampler = sampler
        super().__init__()
        if tiny:
            data_for_split = get_split(path=datapath, split=split,
                                       subset='_tiny')
            self.babel_annots = read_json(Path(datapath) / f'../babel-teach/{split}.json')
        else:
            data_for_split = get_split(path=datapath, split=split)
            self.babel_annots = read_json(Path(datapath) / f'../babel-teach/{split}.json')

        motion_data = {}
        texts_data = {}
        durations = {}
        transitions_data = {}
        if progress_bar:
            enumerator = enumerate(tqdm(data_for_split, f"Loading BABEL {split}"))
        else:
            enumerator = enumerate(data_for_split)

        if kwargs.get('short_db', False):
            maxdata = 600
        else:
            maxdata = np.inf

        crop_samples = kwargs.get('cropping_sampler', False)

        num_bad_actions = 0
        num_bad_short = 0
        valid_data_len = 0
        invalid = 0
        all_data_len = 0
        num_bad_bml = 0
        num_not_kit = 0
        self.sep_to_4 = True
        assert (self.sep_to_4==True)

        for i, sample in enumerator:
            if self.kit_only and not tiny and 'KIT/KIT' not in sample['fname']:
                num_not_kit += 1
                continue
            if len(motion_data) >= maxdata:
                break

            all_data_len += 1
            nframes_total = len(sample['poses'])
            last_framerate = sample['fps']
            babel_id = sample['babel_id']
            seg_ids, seg_acts, valid = extract_frame_labels(self.babel_annots[babel_id],
                                                            fps=last_framerate,
                                                            seqlen=nframes_total,
                                                            datatype=self.dtype)
            if not valid:
                invalid += 1
                continue

            for index, seg in enumerate(seg_ids):
                frames_0_valid, frames_1_valid = False, False
                if self.dtype == 'separate_pairs':
                    fpair = separate_actions(seg)
                    duration = [(e - s + 1) for s, e in fpair]
                    duration[-1] -= 1
                    if self.sep_to_4:
                        if len(duration) == 2:
                            duration.insert(1, 0)
                        else:
                            duration[1] -= 1
                        dur_1, dur_tr, dur_2 = duration
                        dur_tr_valid = True if dur_tr > 0 else False
                        dur_dict = get_all_seq_durs(duration, sampler.max_len, sampler.min_len, crop_samples)
                        dur1, bias_dur1 = dur_dict["dur1"], dur_dict["bias_dur1"]
                        dur2, bias_dur2 = dur_dict["dur2"], dur_dict["bias_dur2"]
                        dur1_t, bias_dur1_t = dur_dict["dur1_t"], dur_dict["bias_dur1_t"]
                        dur2_t, bias_dur2_t = dur_dict["dur2_t"], dur_dict["bias_dur2_t"]
                        if dur1 != None:
                            frames_0_valid = True
                            frames_0 = np.arange(fpair[0][0], fpair[0][1]+1)[bias_dur1: bias_dur1 + dur1]
                            is_transition_0 = torch.zeros(dur_1)[bias_dur1: bias_dur1 + dur1]
                            if (dur_tr_valid) and (dur1_t != None):
                                frames_0_T = np.arange(fpair[0][0], fpair[1][1]+1)[bias_dur1_t: bias_dur1_t + dur1_t]
                                is_transition_0_T = torch.concatenate((torch.zeros(dur_1), torch.ones(dur_tr)))[bias_dur1_t: bias_dur1_t + dur1_t]
                        if dur2 != None:
                            frames_1_valid = True
                            frames_1 = np.arange(fpair[-1][0], fpair[-1][1]+1)[bias_dur2: bias_dur2 + dur2]
                            is_transition_1 = torch.zeros(dur_2)[bias_dur2: bias_dur2 + dur2]
                            if dur_tr_valid and (dur2_t != None) :
                                frames_1_T = np.arange(fpair[1][0], fpair[-1][1]+1)[bias_dur2_t: bias_dur2_t + dur2_t]
                                is_transition_1_T = torch.concatenate((torch.ones(dur_tr), torch.zeros(dur_2)))[bias_dur2_t: bias_dur2_t + dur2_t]
                    else:
                        frames = np.arange(fpair[0][0], fpair[-1][1])

                    if (not frames_1_valid) and (not frames_0_valid):
                        continue

                    if len(duration) == 2:
                        duration.insert(1, 0)
                else:
                    frames = np.arange(seg[0], seg[1])
                    duration = len(frames)

                if self.sep_to_4:
                    if frames_0_valid:
                        smpl_data_0 = {"poses":
                                           torch.from_numpy(sample['poses'][frames_0]).float(),
                                       "trans":
                                           torch.from_numpy(sample['trans'][frames_0]).float()}
                        if dur_tr_valid and (dur1_t != None):
                            smpl_data_0_T = {"poses":
                                                 torch.from_numpy(sample['poses'][frames_0_T]).float(),
                                             "trans":
                                                 torch.from_numpy(sample['trans'][frames_0_T]).float()}
                    if frames_1_valid:
                        smpl_data_1 = {"poses":
                                           torch.from_numpy(sample['poses'][frames_1]).float(),
                                       "trans":
                                           torch.from_numpy(sample['trans'][frames_1]).float()}
                        if dur_tr_valid and (dur2_t != None):
                            smpl_data_1_T = {"poses":
                                                 torch.from_numpy(sample['poses'][frames_1_T]).float(),
                                             "trans":
                                                 torch.from_numpy(sample['trans'][frames_1_T]).float()}
                else:
                    smpl_data = {"poses":
                                     torch.from_numpy(sample['poses'][frames]).float(),
                                 "trans":
                                     torch.from_numpy(sample['trans'][frames]).float()}
                # pose: [T, 22, 3, 3]
                # if split != 'test': # maybe include this (it was there originally): split != "test"
                if not self.dtype == 'separate_pairs':
                    # Accept or not the sample, based on the duration
                    if not self.sampler.accept(duration):
                        num_bad_short += 1
                        continue
                else:
                    # check acceptance for long sequences ... TODO
                    if not self.sep_to_4:  # do not accept by duration
                        dur1, dur_tr, dur2 = duration
                        if not (self.sampler.accept(dur1) and self.sampler.accept(dur2)
                                and self.sampler.accept(dur1 + dur_tr) and self.sampler.accept(dur2 + dur_tr)):
                            num_bad_short += 1
                            continue
                valid_data_len += 1
                if seg_acts[index] in EXCLUDED_ACTIONS:
                    num_bad_actions += 1
                    continue

                if self.walk_only:
                    if not 'walk' in seg_acts[index]:
                        num_bad_actions += 1
                        continue

                from data_loaders.amass.tools.smpl import smpl_data_to_matrix_and_trans
                # here we take smpl data in poses/trans and move to rots/trans
                if self.sep_to_4:
                    if frames_0_valid:
                        smpl_data_0 = smpl_data_to_matrix_and_trans(smpl_data_0, nohands=True)
                        if dur_tr_valid and (dur1_t != None):
                            smpl_data_0_T = smpl_data_to_matrix_and_trans(smpl_data_0_T, nohands=True)
                    if frames_1_valid:
                        smpl_data_1 = smpl_data_to_matrix_and_trans(smpl_data_1, nohands=True)
                        if dur_tr_valid and (dur2_t != None):
                            smpl_data_1_T = smpl_data_to_matrix_and_trans(smpl_data_1_T, nohands=True)
                else:
                    smpl_data = smpl_data_to_matrix_and_trans(smpl_data, nohands=True)
                # Load rotation features (rfeats) data from AMASS
                if mode == 'train':
                    # if split != 'test' and split != 'val':
                    if load_with_rot:
                        if self.sep_to_4:
                            if frames_0_valid:
                                features_0 = self.transforms.rots2rfeats(smpl_data_0)
                                if dur_tr_valid and (dur1_t != None):
                                    features_0_T = self.transforms.rots2rfeats(smpl_data_0_T)
                            if frames_1_valid:
                                features_1 = self.transforms.rots2rfeats(smpl_data_1)
                                if dur_tr_valid and (dur2_t != None):
                                    features_1_T = self.transforms.rots2rfeats(smpl_data_1_T)
                        else:
                            features = self.transforms.rots2rfeats(smpl_data)
                    # Load xyz features (jfeats) data from AMASS
                    else:
                        joints = self.transforms_smpl.rots2joints(smpl_data)
                        features = self.transforms_xyz.joints2jfeats(joints)
                else:
                    joints = smpl_data

                if self.dtype == 'separate_pairs':
                    if self.sep_to_4:
                        if frames_0_valid:
                            texts_data[f'{babel_id}-{index}-{1}'] = seg_acts[index][0]
                            durations[f'{babel_id}-{index}-{1}'] = dur1
                            transitions_data[f'{babel_id}-{index}-{1}'] = is_transition_0
                            if dur_tr_valid and (dur1_t != None):
                                texts_data[f'{babel_id}-{index}-{1_1}'] = seg_acts[index][0]
                                durations[f'{babel_id}-{index}-{1_1}'] = dur1_t
                                transitions_data[f'{babel_id}-{index}-{1_1}'] = is_transition_0_T
                        if frames_1_valid:
                            texts_data[f'{babel_id}-{index}-{2}'] = seg_acts[index][1]
                            durations[f'{babel_id}-{index}-{2}'] = dur2
                            transitions_data[f'{babel_id}-{index}-{2}'] = is_transition_1
                            if dur_tr_valid and (dur2_t != None):
                                texts_data[f'{babel_id}-{index}-{2_1}'] = seg_acts[index][1]
                                durations[f'{babel_id}-{index}-{2_1}'] = dur2_t
                                transitions_data[f'{babel_id}-{index}-{2_1}'] = is_transition_1_T
                    else:
                        texts_data[f'{babel_id}-{index}'] = seg_acts[index]
                        durations[f'{babel_id}-{index}'] = duration
                else:
                    texts_data[f'{babel_id}-{index}'] = seg_acts[index]
                    durations[f'{babel_id}-{index}'] = duration

                if mode == 'train':
                    if self.sep_to_4:
                        if frames_0_valid:
                            motion_data[f'{babel_id}-{index}-{1}'] = features_0
                            if dur_tr_valid and (dur1_t != None):
                                motion_data[f'{babel_id}-{index}-{1_1}'] = features_0_T

                        if frames_1_valid:
                            motion_data[f'{babel_id}-{index}-{2}'] = features_1
                            if dur_tr_valid and (dur2_t != None):
                                motion_data[f'{babel_id}-{index}-{2_1}'] = features_1_T
                    else:
                        motion_data[f'{babel_id}-{index}'] = features
                else:
                    motion_data[f'{babel_id}-{index}'] = joints
        if split != "test" and not tiny:
            total = valid_data_len
            logger.info(f"Processed {all_data_len} sequences and found {invalid} invalid cases based on the datatype.")
            logger.info(f"{total} sequences -- datatype:{self.dtype}.")
            percentage = 100 * (num_bad_actions + num_bad_short) / (total + num_bad_short + num_bad_actions)
            logger.info(f"{percentage:.4}% of the sequences which are rejected by the sampler in total.")
            percentage = 100 * num_bad_actions / (total + num_bad_short + num_bad_actions)
            logger.info(
                f"{percentage:.4}% of the sequence which are rejected by the sampler, because of the excluded actions.")
            percentage = 100 * num_bad_short / (total + num_bad_short + num_bad_actions)
            logger.info(
                f"{percentage:.4}% of the sequence which are rejected by the sampler, because they are too short(<{self.sampler.min_len / 30} secs) or too long(>{self.sampler.max_len / 30} secs).")
            logger.info(f"Discard from BML: {num_bad_bml}")
            logger.info(f"Discard not KIT: {num_not_kit}")

        self.motion_data = motion_data
        self.texts_data = texts_data
        self.transitions_data = transitions_data

        def get_tokens(caption):
            word_list, pose_list = process_text(caption)
            return ['%s/%s' % (word_list[i], pose_list[i]) for i in range(len(word_list))]

        if self.parse_tokens:
            self.tokens_data = {}
            if self.sep_to_4:
                for k, text in tqdm(self.texts_data.items()):
                    self.tokens_data.update({k: get_tokens(text)})
            else:
                for k, (text0, text1) in tqdm(self.texts_data.items()):
                    self.tokens_data.update({k: (get_tokens(text0), get_tokens(text1))})

        # if not tiny:
        self._split_index = list(motion_data.keys())
        self._num_frames_in_sequence = durations
        # breakpoint()
        self.keyids = list(self.motion_data.keys())

        if split == 'test' or split == 'val':
            # does not matter should be removed, just for code to not break
            self.nfeats = 135
        elif self.dtype == 'separate_pairs':
            self.nfeats = 135
        else:
            self.nfeats = len(self[0]["datastruct"].features[0])

    def _load_datastruct(self, keyid, frame_ix=None):
        features = self.motion_data[keyid][frame_ix]
        datastruct = self.transforms.Datastruct(features=features)
        return datastruct

    def _load_text(self, keyid):
        sequences = self.texts_data[keyid]
        return sequences

    def _load_tokens(self, keyid):
        sequences = self.tokens_data[keyid]
        return sequences

    def _load_actions(self, keyid):
        actions_all = self.action_datas[keyid]
        return actions_all

    def load_keyid(self, keyid, mode='train'):
        num_frames = self._num_frames_in_sequence[keyid]

        text = self._load_text(keyid)
        if self.parse_tokens:
            tokens = self._load_tokens(keyid)

        if mode == 'train':
            # max_seq_len = 250
            if self.dtype == 'separate_pairs':
                if self.sep_to_4:
                    features = self.motion_data[keyid]
                    length = self._num_frames_in_sequence[keyid]
                    is_transition = self.transitions_data[keyid]

                    element = {"features": features,
                               "length": length,
                               "is_transition": is_transition,
                               "keyid": keyid,
                               "text": text,
                               }
                else:
                    features = self.motion_data[keyid]
                    length_0, length_transition, length_1 = self._num_frames_in_sequence[keyid]

                    features_0 = features[:length_0]  # lenght_0
                    features_1_with_transition = features[length_0:]  # lenght_1 + length_transition
                    features_1 = features[(length_0 + length_transition):]  # lenght_1

                    features_0_with_transition = features[:(length_0 + length_transition)]


                    element = {"features_0": features_0,
                               "features_1": features_1,
                               "features_1_with_transition": features_1_with_transition,
                               "features_0_with_transition": features_0_with_transition,
                               "length_0": length_0,
                               "length_1": length_1,
                               "length_transition": length_transition,
                               "length_1_with_transition": length_1 + length_transition,
                               "length_0_with_transition": length_0 + length_transition,
                               "keyid": keyid,
                               "text_0": text[0],
                               "text_1": text[1],
                               }

                if self.parse_tokens:
                    if self.sep_to_4:
                        element.update({
                            "tokens": tokens
                        })
                    else:
                        element.update({
                            "tokens_0": tokens[0],
                            "tokens_1": tokens[1],
                        })

            else:
                frame_ix = self.sampler(num_frames)
                datastruct = self._load_datastruct(keyid, frame_ix)
                element = {'datastruct': datastruct, 'text': text,
                           'length': len(datastruct), 'keyid': keyid}
        else:
            raise ValueError("mdm project - you should never use mode other than train in our scope")
        return element

    def load_seqid(self, seqid):

        segs_keyids = [keyid for keyid in self._split_index if keyid.split('-')[0] == seqid]
        segs_keyids = sorted([(e.split('-')[0], int(e.split('-')[1])) for e in segs_keyids], key=lambda x: x[1])
        segs_keyids = ['-'.join([seq, str(id)]) for seq, id in segs_keyids]
        current = segs_keyids[0]
        texts = []
        lens = []
        ov = False
        if len(segs_keyids) == 1:
            t0, t1 = self._load_text(current)
            l0, lt, l1 = self._num_frames_in_sequence[current]
            lens = [l0, l1 + lt]
            texts = [t0, t1]
        else:
            while True:
                t0, t1 = self._load_text(current)
                l0, lt, l1 = self._num_frames_in_sequence[current]
                if not ov:
                    texts.append(t0)
                    texts.append(t1)
                    l1t = lt + l1
                    lens.append(l0)
                    lens.append(l1t)
                else:
                    texts.append(t1)
                    l1t = lt + l1
                    lens.append(l1t)
                if current == segs_keyids[-1]:
                    break
                candidate_next = [i for i in segs_keyids[(segs_keyids.index(current) + 1):] if
                                  self._load_text(i)[0] == t1]

                if candidate_next:
                    ov = True
                    max_id = np.argmax(np.array([self._num_frames_in_sequence[cn][1] for cn in candidate_next]))
                    next_seg = candidate_next[max_id]
                    current = next_seg
                else:
                    ov = False
                    if current != segs_keyids[-1]:
                        current = segs_keyids[segs_keyids.index(current) + 1]
                    else:
                        continue
        element = {'length': lens,
                   'text': texts}
        return element

    def __getitem__(self, index):
        keyid = self._split_index[index]
        return self.load_keyid(keyid, mode='train')

    def __len__(self):
        return len(self._split_index)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"
