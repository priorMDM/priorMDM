import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):
    """this is a wrapper around a model that applies classifier-free guidance when sampling.
    Note that when accessing the model's attributes, you must it returns the wrapped model's attributes.
    This does not apply to functions, though"""
    def __init__(self, model):
        super().__init__()
        vars(self)['model'] = model
        assert model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

    def __getattr__(self, name: str):
        model = vars(self)['model']
        return getattr(model, name)

    def forward(self, x, timesteps, y=None):
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))
    
    def parameters(self):
        return self.model.parameters()

class UnconditionedModel(nn.Module):
    """this is a wrapper around a model that forces unconditional sampling.
    Note that when accessing the model's attributes, you must it returns the wrapped model's attributes.
    This does not apply to functions, though"""
    def __init__(self, model):
        super().__init__()
        vars(self)['model'] = model
        assert model.cond_mask_prob > 0, 'Cannot run unconditional generation on a model that has not been trained with no conditions'

    def __getattr__(self, name: str):
        model = vars(self)['model']
        return getattr(model, name)


    def forward(self, x, timesteps, y=None):
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond
    
    def parameters(self):
        return self.model.parameters()

def wrap_model(model, args):
    if args.guidance_param not in [0., 1.]:
        return ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    elif args.guidance_param == 0:
        return UnconditionedModel(model)
    else:
        return model
    