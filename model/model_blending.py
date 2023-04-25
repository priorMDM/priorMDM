import torch
from torch import nn
import random

class ModelBlender(nn.Module):
    """
    interpolates the output of multiple models with the given weights
    """
    def __init__(self, models, weights):
        super().__init__()
        self.models = models
        # weights can also be per-frame mask for each model or any sort of list of vectors that can be multiplied with the motion
        self.weights = weights
    
    def forward(self, x, timesteps, y=None):
        out = torch.tensor([0], device=next(self.parameters()).device)
        models = vars(self)['models']
        weights = vars(self)['weights']
        for model, weight in zip(models, weights):
            out = weight * model(x, timesteps, y) + out
        return out
    
    def  __getattr__(self, name: str):
        return vars(self)['models'][0].__getattr__(name)
    
    # The following is an altentative implementation, using a random model on every step instead of interpolating them. Here weights must be a 1D vector
    # def forward(self, x, timesteps, y=None):
    #     curr_model = random.choice(self.models)
    #     return curr_model(x, timesteps, y)
    
    def parameters(self):
        return vars(self)['models'][0].parameters()



