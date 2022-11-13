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
        y_label = torch.tensor(int(self.labels.iloc[index, 1]))

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


def preprocess_all_stl10(stl_data_path, destination_path_root, transform=DexinedAug()):
    """Convert all images from stl10 to edge versions and save them in the same directory"""
    print("Provided stl_data_path: ", stl_data_path)

    edge_dest_path_root = os.path.join(destination_path_root, "Dexi")
    orig_dest_path_root = os.path.join(destination_path_root, "STL10")

    for ds_split in ['train', 'unlabeled', 'test']:
        print("Split: ", ds_split)

        print("Applying DexiNed and saving results.")
        # Load STL10 and apply DexiNed edge detection
        # TODO (jakob-bjorner): 1. Fix the hardcoded `transform` argument
        # TODO (jakob-bjorner): 2. Generate data for all splits
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
            Image.fromarray(np.array(edge_img)).save(path_name)
        # Save data labesl
        labels_df.to_csv(os.path.join(edge_split_dir, "labels.csv"), index=False)
        
        print("Saving a copy of original images.")
        # Instantiate another dataset without DexiNed transformations
        orig_dataset = datasets.STL10(stl_data_path, split=ds_split, download=True)
        labels_df = pd.DataFrame(columns=["image_num", "label"])
        orig_split_dir = os.path.join(orig_dest_path_root, ds_split)
        os.makedirs(orig_split_dir, exist_ok=True)
        # No GPU used. So directly iterate on dataset instead of using a data loader.
        for i, (orig_img, label) in enumerate(tqdm(orig_dataset)):
            path_name = os.path.join(orig_split_dir, str(i) + ".png")
            Image.fromarray(np.array(orig_img)).save(path_name)
        # Save data labesl
        labels_df.to_csv(os.path.join(orig_split_dir, "labels.csv"), index=False)

    

if __name__ == "__main__":

    # TODO (jakob-bjorner): 3. Remove unneeded comments.
    # preprocess_all_stl10("/srv/share4/jbjorner3/datasets", "/srv/share4/jbjorner3/Edge_images/Dexi", transform=DexinedAug())
    preprocess_all_stl10("/datasets", "./Edge_images/", transform=DexinedAug())
    # preprocess_all_stl10("./datasets", "./Edge_images/Canny", transform=CannyAug()) This is not useful to do because it has hyper parameters and doesn't take time to run.

