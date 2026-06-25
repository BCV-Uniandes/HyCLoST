import random, os, timm
import numpy as np
import torch

import random
import os
import numpy as np
import torch
import torchvision

def seed_everything(seed: int = 42):
    """
    Set seed for reproducibility across different libraries: Python, NumPy, and PyTorch.
    
    Args:
        seed: Integer seed for random number generation.
    """
    random.seed(seed)  # Python RNG
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # PyTorch CPU RNG
    torch.cuda.manual_seed(seed)  # PyTorch GPU RNG for current device
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU RNG for all devices
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    os.environ['PYTHONHASHSEED'] = str(seed)

class EightSymmetry(object):
    """Returns a tuple of the eight symmetries resulting from rotation and reflection.
    
    This behaves similarly to TenCrop.
    This transform returns a tuple of images and there may be a mismatch in the number of inputs and targets your Dataset returns. See below for an example of how to deal with this.
    Example:
     transform = Compose([
         EightSymmetry(), # this is a tuple of PIL Images
         Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
     ])
    """
    # This class function was taken fron the original ST-Net repository at:
    # https://github.com/bryanhe/ST-Net/blob/43022c1cb7de1540d5a74ea2338a12c82491c5ad/stnet/transforms/eight_symmetry.py#L3
    def __call__(self, img):
        identity = lambda x: x
        ans = []
        for i in [identity, torchvision.transforms.RandomHorizontalFlip(1)]:
            for j in [identity, torchvision.transforms.RandomVerticalFlip(1)]:
                for k in [identity, torchvision.transforms.RandomRotation((90, 90))]:
                    ans.append(i(j(k(img))))
        return tuple(ans)

    def __repr__(self):
        return self.__class__.__name__ + "()"