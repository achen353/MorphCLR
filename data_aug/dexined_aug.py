"""
File for data augmentation on dexined type edge images
"""
import sys
import os
import numpy as np
import torch
from torchvision import transforms, datasets


root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root_path)

from DexiNed.model_inference import model_process, pre_process
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset

class DexinedAug(object):
    """
    apply substutute image with the edge version of the image.
    
    only used for inference and preprocessing images before training, too slow for live training augmentation
    
    Parameters
    ----------
    image : np.array of shape (H, W, 3)
        the image to be processed
    """
    def __init__(self):
        pass
    def __call__(self, image):
        image = np.array(image)
        process_image = model_process(image, "cuda")[0]
        return process_image




