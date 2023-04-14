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


from typing import Optional
from dataclasses import dataclass

# from utils.parser_util import frame_sampler_parser


@dataclass
class FrameSampler:
    sampling: str = "conseq"
    sampling_step: int = 1
    request_frames: Optional[int] = None
    threshold_reject: int = 0.75
    # args = frame_sampler_parser()
    max_len: int = 250
    min_len: int = 45
    # max_len: int = 360
    # min_len: int = 15

    def __call__(self, num_frames):
        from .frames import get_frameix_from_data_index
        return get_frameix_from_data_index(num_frames,
                                           self.max_len,
                                           self.request_frames,
                                           self.sampling,
                                           self.sampling_step)

    def accept(self, duration):
        # Outputs have original lengths
        # Check if it is too long
        if self.request_frames is None:
            if duration > self.max_len:
                return False
            if duration < self.min_len:
                return False
        else:
            # Reject sample if the length is
            # too little relative to
            # the request frames
            
            # min_number = self.threshold_reject * self.request_frames
            if duration < self.min_len: # min_number:
                return False
        return True

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)
