"""
File for data augmentation on dexined type edge images
"""
import sys
import os
import cv2
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

class CannyAug(object):
    """
    apply canny edge detection to the image
    
    Parameters
    ----------
    image : np.array of shape (H, W, 3)
        the image to be processed
    """
    def __init__(self, canny_threshold=200, canny_threshold2=400):
        self.canny_threshold = canny_threshold
        self.canny_threshold2 = canny_threshold2
        pass
    def __call__(self, image):
        image = np.array(image, dtype=np.uint8)
        process_image = cv2.Canny(image, self.canny_threshold, self.canny_threshold2)
        return np.array(process_image)

if __name__ == "__main__":
    # load a single test image from the stl10 dataset and run it through the canny augmentation
    ds = datasets.STL10("./datasets", split="train", transform=CannyAug(), download=True)
    loader = torch.utils.data.DataLoader(
        ds,
        num_workers=1,
        pin_memory=True,
    )
    for i, (image, label) in enumerate(loader):
        
        print("image shape", image.shape)
        print("label", label)
        cv2.imshow("image", np.array(image[0]))
        cv2.waitKey(0)


