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
from data_aug.view_generator import ContrastiveLearningViewGenerator


class DexiNedDataset(Dataset):
    def __init__(
        self,
        csv_file="datasets/Edge_images/Dexi/train/labels.csv",
        root_dir="datasets/Edge_images/Dexi/train",
        transform=None,
    ):
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
    def __init__(
        self,
        csv_file="Edge_images/Dexi/unlabeled/labels.csv",
        root_dir="Edge_images/Dexi/unlabeled",
        transform=None,
    ):
        super().__init__(csv_file, root_dir, transform)


class DexiNedTrainDataset(DexiNedDataset):
    def __init__(
        self,
        csv_file="Edge_images/Dexi/train/labels.csv",
        root_dir="Edge_images/Dexi/train",
        transform=None,
    ):
        super().__init__(csv_file, root_dir, transform)


class DexiNedTestDataset(DexiNedDataset):
    def __init__(
        self,
        csv_file="Edge_images/Dexi/test/labels.csv",
        root_dir="Edge_images/Dexi/test",
        transform=None,
    ):
        super().__init__(csv_file, root_dir, transform)


class CannyDataset(datasets.STL10):
    """Class to load the canny edge images from the stl10 dataset. done on the fly as the tranform is very fast."""

    def __init__(self, root, transform=None, download=True, **kwargs):
        if type(transform) == ContrastiveLearningViewGenerator:
            transform.base_transform = transforms.Compose(
                [CannyAug(), transforms.ToPILImage()]
                + transform.base_transform.transforms
            )
        else:
            transform = transforms.Compose(
                [CannyAug()] + (transform if transform in locals() else [])
            )
        super().__init__(root, transform=transform, download=download, **kwargs)


class DualDataset(Dataset):
    """class for loading two datasets at once. For example DexiNed and original images"""

    def __init__(self, dataset1: Dataset, dataset2: Dataset):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        assert len(self.dataset1) == len(self.dataset2)

    def __len__(self):
        return len(self.dataset1)

    def _assert_label_equal(self, label1, label2):
        if torch.is_tensor(label1):
            label1 = label1.clone().numpy()
        if torch.is_tensor(label2):
            label2 = label2.clone().numpy()

        label1 = np.array(label1)
        label2 = np.array(label2)

        if label1.ndim == 0:
            label1 = label1[np.newaxis]
        if label2.ndim == 0:
            label2 = label2[np.newaxis]

        assert np.array_equal(
            label1, label2
        ), "[ERROR] Labels Mismatched: label1 is {} but label2 is {}.".format(
            label1, label2
        )

    def __getitem__(self, index):
        label1 = self.dataset1[index][1]
        label2 = self.dataset2[index][1]

        self._assert_label_equal(label1, label2)

        return (self.dataset1[index][0], self.dataset2[index][0], label1)


class STL10(datasets.STL10):
    """Class to load stl10 dataset, just to have everything in one place"""

    def __init__(self, root, download=True, **kwargs):
        super().__init__(root, download=download, **kwargs)


class StylizedSTL10Dataset(torch.utils.data.Dataset):
    """A duplicate class for StylizedSTL10Dataset in eval.py"""

    def __init__(self, source_dir, transform=None):
        data = []
        style_labels = []
        content_labels = []
        print("[INFO] Preparing stylzed images from: {}".format(source_dir))
        for source_path in sorted(os.listdir(source_dir)):
            source_example_path = sorted(
                os.listdir(os.path.join(source_dir, source_path))
            )
            source_example_path = [
                os.path.join(source_dir, source_path, x) for x in source_example_path
            ]
            dir_style_labels = torch.tensor(
                [int(x.split("/")[-1].split("_")[3]) for x in source_example_path]
            )
            style_labels.append(dir_style_labels)
            content_style_labels = torch.tensor(
                [int(source_path) - 1] * dir_style_labels.shape[0]
            )
            content_labels.append(content_style_labels)
            tensor_images = [
                transforms.ToTensor()(Image.open(x).convert("RGB"))
                for x in source_example_path
            ]

            tensor_images = torch.stack(tensor_images)

            data.append(tensor_images)
        data = torch.cat(data, dim=0)
        style_labels = torch.cat(style_labels, dim=0).reshape(-1, 1)
        content_labels = torch.cat(content_labels, dim=0).reshape(-1, 1)
        target = torch.cat((style_labels, content_labels), dim=-1)

        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            d = self.data[idx].permute(1, 2, 0)
            d = (d * 255).to(torch.uint8)
            return self.transform(d), self.target[idx]

        return self.data[idx], self.target[idx]


class CannyStylizedDataset(StylizedSTL10Dataset):
    def __init__(self, source_dir, transform=None, **kwargs):
        if type(transform) == ContrastiveLearningViewGenerator:
            transform.base_transform = transforms.Compose(
                [CannyAug(), transforms.ToPILImage()]
                + transform.base_transform.transforms
            )
        else:
            transform = transforms.Compose(
                [CannyAug()] + (transform if transform in locals() else [])
            )
        super().__init__(source_dir, transform)


class DexiNedStylizedTestDataset(DexiNedDataset):
    def __init__(
        self,
        csv_file="Edge_images/Dexi_stylized/labels.csv",
        root_dir="Edge_images/Dexi_stylized",
        transform=None,
    ):
        super().__init__(csv_file, root_dir, transform)


def preprocess_all_stl10(stl_data_path, destination_path_root, transform=DexinedAug()):
    """Convert all images from stl10 to edge versions and save them in the same directory"""
    print("Provided stl_data_path: ", stl_data_path)

    edge_dest_path_root = os.path.join(destination_path_root, "Dexi")
    orig_dest_path_root = os.path.join(destination_path_root, "STL10")

    for ds_split in ["train", "unlabeled", "test"]:
        print("Split: ", ds_split)

        print("Applying DexiNed and saving results.")
        # Load STL10 and apply DexiNed edge detection
        edge_dataset = datasets.STL10(
            stl_data_path,
            split=ds_split,
            transform=transforms.ToTensor(),
            download=True,
        )
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
            edge_img = edge_img[0].permute(1, 2, 0)
            edge_img = transform((edge_img * 255).to(torch.uint8))
            path_name = os.path.join(edge_split_dir, str(i) + ".png")
            labels_df.loc[i] = [str(i) + ".png", label]
            Image.fromarray(edge_img).save(path_name)
        labels_df.to_csv(os.path.join(edge_split_dir, "labels.csv"), index=False)


def preprocess_all_stylized_stl10(
    stylized_data_path="../stylization/inter_class_stylized_dataset",
    destination_path_root="Dexi_stylized",
    transform=DexinedAug(),
):
    """Convert all images from stl10 to edge versions and save them in the same directory"""
    print("Provided stylized_data_path: ", stylized_data_path)

    os.makedirs(destination_path_root, exist_ok=True)

    print("Applying DexiNed and saving results.")
    # Load STL10 and apply DexiNed edge detection
    edge_dataset = StylizedSTL10Dataset(stylized_data_path)
    # Use a data loader with pin_memory = True for faster CPU to GPU memory transfer
    edge_data_loader = torch.utils.data.DataLoader(
        edge_dataset,
        num_workers=1,
        pin_memory=True,
    )
    # Maintain a dataframe for image labels
    labels_df = pd.DataFrame(columns=["image_num", "label"])

    # Save edge images to directories
    for i, (edge_img, label) in enumerate(tqdm(edge_data_loader)):
        edge_img = edge_img[0].permute(1, 2, 0)
        edge_img = transform((edge_img * 255).to(torch.uint8))
        path_name = os.path.join(destination_path_root, str(i) + ".png")
        labels_df.loc[i] = [str(i) + ".png", label.flatten()]
        Image.fromarray(edge_img).save(path_name)
    labels_df.to_csv(os.path.join(destination_path_root, "labels.csv"), index=False)


if __name__ == "__main__":
    # ds = CannyDataset("./datasets")
    # ds1 = DualDataset(ds, STL10("./datasets", split="train"))
    # ds2 = DualDataset(DexiNedTrainDataset(), STL10("./datasets", split="train"))
    # import matplotlib.pyplot as plt

    # for ds in [ds1, ds2]:
    #     for image_canny, image, label in ds:
    #         # display image, and image_canny on a single row
    #         fig, (ax1, ax2) = plt.subplots(1, 2)
    #         ax1.imshow(image_canny)
    #         ax2.imshow(image)
    #         print(label)
    #         plt.show()

    #         print(image_canny, image, label)
    #         break
    preprocess_all_stylized_stl10()
