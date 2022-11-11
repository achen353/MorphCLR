import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets
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
    print("provided stl_data_path", stl_data_path)
    from torchvision import transforms 
    for ds_split in ['train', 'unlabeled', 'test']:
        ds = datasets.STL10(stl_data_path, split=ds_split, transform=transforms.ToTensor(), download=True)
        loader = torch.utils.data.DataLoader(
            ds,
            num_workers=1,
            pin_memory=True,
            # drop_last=True, ?
        )
        labels_df = pd.DataFrame(columns=["image_num", "label"])
        split_dir = os.path.join(destination_path_root, ds_split)
        os.makedirs(split_dir, exist_ok=True)
        for i, (image, label) in enumerate(loader):
            image = image[0].permute(1,2,0)
            image = transform((image * 255).to(torch.uint8))
            path_name = os.path.join(split_dir, str(i) + ".png")
            labels_df.loc[i] = [str(i) + ".png", label]
            Image.fromarray(image).save(path_name)
        labels_df.to_csv(os.path.join(split_dir, "labels.csv"), index=False)
    

if __name__ == "__main__":

    # preprocess_all_stl10("/srv/share4/jbjorner3/datasets", "/srv/share4/jbjorner3/Edge_images/Dexi", transform=DexinedAug())
    preprocess_all_stl10("./datasets", "./Edge_images/Dexi", transform=DexinedAug())
    # preprocess_all_stl10("./datasets", "./Edge_images/Canny", transform=CannyAug()) This is not useful to do because it has hyper parameters and doesn't take time to run.

