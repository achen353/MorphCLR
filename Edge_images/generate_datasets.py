"""
file for generating and using datasets for edge images.
example usage:
    from Edge_images.generate_datasets import EdgeDataset
    ds = DualDataset(CannyDataset("./datasets"), STL10("./datasets", split="train"))
    for image_canny, image, label in ds:
        ...
    # example for dexined edge images
    ds = DualDataset(DexiNedTrainingDataset, STL10("./datasets", split="train"))
    for image_dexi, image, label in ds:
        ...
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm
from data_aug.dexined_aug import CannyAug, DexinedAug
from torchvision import transforms
class DexiNedDataset(Dataset):
    def __init__(self, 
                csv_file="datasets/Edge_images/Dexi/train/labels.csv", 
                root_dir="datasets/Edge_images/Dexi/train", 
                transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.labels.iloc[index, 0])
        image = Image.open(img_path)
        y_label = eval("torch." + self.labels.iloc[index, 1])

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

class DexiNedUnlabeledDataset(DexiNedDataset):
    # all labels are -1
    def __init__(self,
                 csv_file="Edge_images/Dexi/unlabeled/labels.csv", 
                 root_dir="Edge_images/Dexi/unlabeled",
                 transform=None):
        super().__init__(csv_file, root_dir, transform) 

class DexiNedTrainDataset(DexiNedDataset):
    def __init__(self, 
                csv_file="Edge_images/Dexi/train/labels.csv", 
                root_dir="Edge_images/Dexi/train", 
                transform=None):
        super().__init__(csv_file, root_dir, transform)

class DexiNedTestDataset(DexiNedDataset):
    def __init__(self,
                 csv_file="Edge_images/Dexi/test/labels.csv",
                 root_dir="Edge_images/Dexi/test",
                 transform=None):
        super().__init__(csv_file, root_dir, transform)

class CannyDataset(datasets.STL10):
    """ Class to load the canny edge images from the stl10 dataset. done on the fly as the tranform is very fast. """
    def __init__(self, root, transform=None, download=True, **kwargs):
        transform = transforms.Compose([CannyAug()] + (transform if transform in locals() else []))
        super().__init__(root, transform=transform, download=download, **kwargs)
class DualDataset(Dataset):
    """ class for loading two datasets at once. For example DexiNed and original images"""
    def __init__(self, dataset1: Dataset, dataset2: Dataset):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        assert len(self.dataset1) == len(self.dataset2)
    def __len__(self):
        return len(self.dataset1)
    def __getitem__(self, index):
        label1 = self.dataset1[index][1]
        label2 = self.dataset2[index][1]
        assert label1 == label2
        return (self.dataset1[index][0], self.dataset2[index][0], label1)
class STL10(datasets.STL10):
    """ Class to load stl10 dataset, just to have everything in one place"""
    def __init__(self, root, download=True, **kwargs):
        super().__init__(root, download=download, **kwargs)

def preprocess_all_stl10(stl_data_path, destination_path_root, transform=DexinedAug()):
    """Convert all images from stl10 to edge versions and save them in the same directory"""
    print("Provided stl_data_path: ", stl_data_path)

    edge_dest_path_root = os.path.join(destination_path_root, "Dexi")
    orig_dest_path_root = os.path.join(destination_path_root, "STL10")

    for ds_split in ['train', 'unlabeled', 'test']:
        print("Split: ", ds_split)

        print("Applying DexiNed and saving results.")
        # Load STL10 and apply DexiNed edge detection
        edge_dataset = datasets.STL10(stl_data_path, split=ds_split, transform=transforms.ToTensor(), download=True)
        # Use a data loader with pin_memory = True for faster CPU to GPU memory transfer
        edge_data_loader = torch.utils.data.DataLoader(
            edge_dataset,
            num_workers=1,
            pin_memory=True,
        )
        # Maintain a dataframe for image labels
        labels_df = pd.DataFrame(columns=["image_num", "label"])
        # Create directory for current data split
        edge_split_dir = os.path.join(edge_dest_path_root, ds_split)
        os.makedirs(edge_split_dir, exist_ok=True)
        # Save edge images to directories
        for i, (edge_img, label) in enumerate(tqdm(edge_data_loader)):
            edge_img = edge_img[0].permute(1,2,0)
            edge_img = transform((edge_img * 255).to(torch.uint8))
            path_name = os.path.join(edge_split_dir, str(i) + ".png")
            labels_df.loc[i] = [str(i) + ".png", label]
            Image.fromarray(image).save(path_name)
        labels_df.to_csv(os.path.join(edge_split_dir, "labels.csv"), index=False)


if __name__ == "__main__":
    ds = CannyDataset("./datasets")
    ds1 = DualDataset(ds, STL10("./datasets", split="train"))
    ds2 = DualDataset(DexiNedTrainDataset(), STL10("./datasets", split="train"))
    import matplotlib.pyplot as plt
    for ds in [ds1, ds2]:
        for image_canny, image, label in ds:
            # display image, and image_canny on a single row
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(image_canny)
            ax2.imshow(image)
            print(label)
            plt.show()
        
            print(image_canny, image, label)
            break
    

    
