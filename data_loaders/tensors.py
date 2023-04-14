import random
import torch
from data_loaders.amass.tools import collate_tensor_with_padding

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'is_transition' in notnone_batches[0]:
        is_transition_batch = torch.stack([b['is_transition']for b in notnone_batches])
        cond['y'].update({'is_transition': is_transition_batch})

    if 'length_transition' in notnone_batches[0]:
        length_transition = [b['length_transition'] for b in notnone_batches]
        cond['y'].update({'length_transition': length_transition})

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'other_motion' in notnone_batches[0]:
        other_motion = [b['other_motion'] for b in notnone_batches]
        other_motion = collate_tensors(other_motion)
        cond['y'].update({'other_motion': other_motion})

    if 'person_id' in notnone_batches[0]:
        textbatch = [b['person_id'] for b in notnone_batches]
        cond['y'].update({'person_id': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    if 'action_cat' in notnone_batches[0]:
        action_cat = torch.stack([b['action_cat']for b in notnone_batches])
        action_cat_mask = torch.stack([b['action_cat_mask']for b in notnone_batches])
        act_cat_list = [b['act_cat_list']for b in notnone_batches]
        cond['y'].update({'action_cat': action_cat})
        cond['y'].update({'action_cat_mask': action_cat_mask})
        cond['y'].update({'act_cat_list': act_cat_list})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.from_numpy(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
        'is_transition': torch.zeros(1), # just for eval not really needed
    } for b in batch]
    return collate(adapted_batch)

def babel_eval_collate(batch):
    try:
        adapted_batch = [{
            'inp': torch.from_numpy(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
            'text': b[2], #b[0]['caption']
            'tokens': b[6],
            'lengths': b[5],
            'is_transition': torch.from_numpy(b[7]),
        } for b in batch]
    except TypeError:
        print(5)
    return collate(adapted_batch)

def pw3d_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'other_motion': torch.tensor(b[0].T).float().unsqueeze(1),
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'person_id': b[3],
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)

from enum import IntEnum

class motion_type(IntEnum):
    MOTION_0 = 0
    MOTION_1 = 1
    MOTION_0_W_T = 2
    MOTION_1_W_T = 3

def pad_sample_with_zeros(sample, vector_len):
    # pad inp, change lenghts, and pad is transition
    n_feats, _, seq_len = sample['inp'].shape
    len_to_pad = vector_len-seq_len
    torch.zeros_like(sample['inp'])
    is_transition_padding = torch.zeros(len_to_pad)
    inp_padding = torch.zeros((n_feats, 1, len_to_pad))
    sample['inp'] = torch.cat((sample['inp'], inp_padding), dim=2)
    sample['is_transition'] = torch.cat((sample['is_transition'], is_transition_padding))
    return sample

def babel_collate(batch):
    from data_loaders.amass.tools import collate_pairs_and_text
    batch = collate_pairs_and_text(batch)
    bs = len(batch['motion_feats'])
    adapted_batch = []
    for ii in range(bs):
        adapted_batch.append({
            'inp': batch['motion_feats'][ii].permute(1, 0).unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
            'text': batch['text'][ii],
            'lengths': batch['length'][ii],
            'is_transition': batch['is_transition'][ii]})
    return collate(adapted_batch)
