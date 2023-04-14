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

import joblib
import sys

from data_loaders.amass.file_io import read_json

sys.path.append('.')

# todo make it parametric I am too tired now
from tqdm import tqdm
import os
amass_path = './data_loaders/babel/babel-smplh-30fps-male/amass.pth.tar'
amass_data = joblib.load(amass_path)
print(f'Loading the dataset from {amass_path}')
babel_path = './data_loaders/babel-teach/id2fname/amass-path2babel.json'
amass2babel = read_json(babel_path)
dataset_db_lists = {'train': [],
                    'val': []}
num_bad = 0
for sample in tqdm(amass_data):
    if sample['fname'] not in amass2babel:
        num_bad += 1
        continue

    split_of_seq = amass2babel[sample['fname']]['split']
    babel_key = amass2babel[sample['fname']]['babel_id']
    if split_of_seq in ['train', 'val']:
    	# construct babel key fro  amass keys and utils
    	sample_babel = {}
    	for k, v in sample.items():
            sample_babel[k] = v
    	sample_babel['babel_id'] = babel_key
    	dataset_db_lists[split_of_seq].append(sample_babel)

print(f'Percentage not found: {num_bad}/{len(amass_data)}')
out_path = 'output-path/babel-smplh-30fps-male'
os.makedirs(out_path, exist_ok=True)
for k, v in dataset_db_lists.items():
    joblib.dump(v, f'{out_path}/{k}.pth.tar')
for k, v in dataset_db_lists.items():
    joblib.dump(v[:10], f'{out_path}/{k}_tiny.pth.tar')
