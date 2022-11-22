import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

from transformers import ViTForImageClassification
from transformers import ViTFeatureExtractor

import torch.nn as nn
import torch.nn.functional as F



class VIT_pretrained(nn.Module):
    def __init__(self, file_path_model, device, model_type = 'google/vit-base-patch16-224-in21k'):
        super().__init__()
        self.device = device
        self.vit_backbone = torch.load(file_path_model).to(device)
        self.model_type = model_type
    def __call__(self, x_batch):
        output = self.vit_backbone(x_batch)
        return output.logits