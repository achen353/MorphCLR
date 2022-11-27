import argparse
import os
import shutil
import sys
import warnings
from collections import defaultdict
from enum import Enum

import gdown
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import cuda
from torch.utils.data import DataLoader
from torchvision import datasets, models
from tqdm import tqdm
from transformers import ViTFeatureExtractor

from Edge_images.generate_datasets import (
    STL10,
    CannyDataset,
    DexiNedTestDataset,
    CannyStylizedDataset,
    DexiNedStylizedTestDataset,
    DualDataset,
)
from models.morphclr import MorphCLRDualEval, MorphCLRSingleEval
from vit import VIT_pretrained

warnings.filterwarnings("ignore", category=UserWarning)

convert_tensor_fn = torchvision.transforms.ToTensor()

vit_model_type = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_type)


class ModelType(Enum):
    SIMCLR = 1
    MORPHCLRSINGLE = 2
    MORPHCLRDUAL = 3
    VIT = 4


torch_model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch SimCLR")
parser.add_argument(
    "-data", metavar="DIR", default="./datasets", help="path to dataset"
)
parser.add_argument(
    "-dataset-name",
    default="stl10",
    help="dataset name",
    choices=[
        "stl10",
        "stl10_canny",
        "stl10_dexined",
    ],
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=torch_model_names,
    help="model architecture: "
    + " | ".join(torch_model_names)
    + " (default: resnet18)",
)
parser.add_argument(
    "-m",
    "--model-type",
    default="simclr",
    choices=["simclr", "morphclr_single", "morphclr_dual", "vit"],
    help="model type: simclr, morphclr_single, morphclr_dual, or vit",
)
parser.add_argument(
    "-c",
    "--checkpoint",
    default="simclr_resnet50_50-epochs_stl10_100-epochs.pt",
    type=str,
    help="file name of the checkpoint",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument("--gpu-index", default=0, type=int, help="Gpu index.")

# def get_local_dataset(source_dir, model_type=ModelType.SIMCLR):
#     # assumes that source dir contains folders of images where each folder of images
#     # has images from the same class whose name is the name of the folder
#     # e.g dir/0 (label) / img_x.png
#     data = []
#     labels = []
#     print("[INFO] Preparing local images from: {}".format(source_dir))
#     for source_path in sorted(os.listdir(source_dir)):
#         labels.append(int(source_path) - 1)
#         source_example_path = sorted(os.listdir(os.path.join(source_dir, source_path)))
#         source_example_path = [os.path.join(source_dir, source_path, x) for x in source_example_path]
#         tensor_examples = [convert_tensor_fn(Image.open(x).convert('RGB')) for x in source_example_path]
#         if model_type == ModelType.SIMCLR:
#             original_tensors = torch.stack(tensor_examples)
#         elif model_type == ModelType.VIT:
#             vit_extracted = feature_extractor(tensor_examples, return_tensors = "pt")
#             original_tensors = vit_extracted['pixel_values']

#         data.append(original_tensors)
#     print("Found {} labels: {}".format(len(labels), labels))
#     return data, labels
# class stylized_dataset(torch.utils.data.dataset):


def compute_accuracies_local(
    model, device, data_loader, model_name, model_type=ModelType.SIMCLR
):
    # logit dim equals number of classes
    model = model.eval()
    model = model.to(device)
    accuracies = defaultdict(int)
    # Loop over all examples in test set
    for data_and_target in tqdm(data_loader):
        # Send the data and label to the device
        if model_type == ModelType.VIT:
            data, target = data_and_target
            data = data["pixel_values"][0]
        elif (
            model_type == ModelType.MORPHCLRSINGLE
            or model_type == ModelType.MORPHCLRDUAL
        ):
            edge_data, non_edge_data, target = data_and_target
            if len(edge_data.shape) == 3:
                edge_data = edge_data.unsqueeze(1)
            # If the image is of grayscale, repeat the dimension to create 3 channels
            if edge_data.shape[1] == 1:
                edge_data = edge_data.repeat(1, 3, 1, 1)
            data = torch.stack([edge_data, non_edge_data], dim=0)
            target = target.flatten()
        else:
            data, target = data_and_target
        data, target = data.to(device), target.to(device)

        # Forward pass the data through the model
        output = model(data)
        preds = torch.argmax(output, dim=1)

        for i in range(output.shape[0]):
            if preds[i] == target[i]:
                accuracies[int(preds[i])] += 1 / 800

    file_path = "accuracies.csv"
    if not os.path.exists(file_path):
        with open(file_path, "a") as f:
            f.write(
                "exp," + ",".join(["class_{}".format(i + 1) for i in range(10)]) + "\n"
            )

    with open(file_path, "a") as f:
        f.write(
            "{},".format(model_name)
            + ",".join([str(accuracies.get(i, 0)) for i in range(10)])
            + "\n"
        )

    print("Accuracies : {}".format(accuracies))
    return accuracies


# def get_local_stylized_dataset_tensors(source_dir):
#     data = []
#     style_labels = []
#     content_labels = []
#     print("[INFO] Preparing stylzed images from: {}".format(source_dir))
#     for source_path in sorted(os.listdir(source_dir)):
#         source_example_path = sorted(os.listdir(os.path.join(source_dir, source_path)))
#         source_example_path = [os.path.join(source_dir, source_path, x) for x in source_example_path]
#         dir_style_labels = torch.tensor([int(x.split('/')[-1].split('_')[3]) for x in source_example_path])
#         style_labels.append(dir_style_labels)
#         content_style_labels = torch.tensor([int(source_path) - 1] * dir_style_labels.shape[0])
#         content_labels.append(content_style_labels)

#         original_tensors = torch.stack([convert_tensor_fn(Image.open(x).convert('RGB')) for x in source_example_path])

#         data.append(original_tensors)

#     return data, style_labels, content_labels


class StylizedSTL10Dataset(torch.utils.data.Dataset):
    def __init__(self, source_dir, model_type=ModelType.SIMCLR):
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
                convert_tensor_fn(Image.open(x).convert("RGB"))
                for x in source_example_path
            ]

            if model_type == ModelType.VIT:
                vit_extracted = feature_extractor(tensor_images, return_tensors="pt")
                tensor_images = vit_extracted["pixel_values"]
            else:
                tensor_images = torch.stack(tensor_images)

            data.append(tensor_images)
        data = torch.cat(data, dim=0)
        style_labels = torch.cat(style_labels, dim=0).reshape(-1, 1)
        content_labels = torch.cat(content_labels, dim=0).reshape(-1, 1)
        target = torch.cat((style_labels, content_labels), dim=-1)

        self.data = data
        self.target = target

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def compute_accuracy_and_ratio(model, device, data_loader, model_name, model_type):
    # logit dim equals number of classes
    print("[INFO] Computing accuracy and style-based decision ratio.")

    model = model.eval()
    model = model.to(device)

    accuracies = defaultdict(int)
    style_decisions = defaultdict(int)
    content_decisions = defaultdict(int)
    for data_and_target in tqdm(data_loader):
        if (
            model_type == ModelType.MORPHCLRSINGLE
            or model_type == ModelType.MORPHCLRDUAL
        ):
            edge_data, non_edge_data, target = data_and_target
            if len(edge_data.shape) == 3:
                edge_data = edge_data.unsqueeze(1)
            # If the image is of grayscale, repeat the dimension to create 3 channels
            if edge_data.shape[1] == 1:
                edge_data = edge_data.repeat(1, 3, 1, 1)
            data = torch.stack([edge_data, non_edge_data], dim=0)
        else:
            data, target = data_and_target

        y_style, y_content = target[:, 0], target[:, 1]
        y_style = y_style.to(device)
        y_content = y_content.to(device)
        preds = model(data.to(device))
        preds = torch.argmax(preds, dim=1)
        for i in range(preds.shape[0]):
            if preds[i] == y_content[i]:
                accuracies[int(preds[i])] += 1 / 800
                content_decisions[int(preds[i])] += 1

            if preds[i] == y_style[i]:
                style_decisions[int(preds[i])] += 1

    print("Accuracy: {}".format(accuracies))
    print("# of style-based decision per class: {}".format(style_decisions))
    print("# of texture-based decision per class: {}".format(content_decisions))

    file_path = "stylized_accuracies.csv"
    if not os.path.exists(file_path):
        with open(file_path, "a") as f:
            f.write(
                "exp,"
                + ",".join(["class_{}_acc".format(i + 1) for i in range(10)])
                + ",".join(["class_{}_style".format(i + 1) for i in range(10)])
                + ",".join(["class_{}_content".format(i + 1) for i in range(10)])
                + "\n"
            )

    with open(file_path, "a") as f:
        f.write(
            "{},".format(model_name)
            + ",".join([str(accuracies.get(i, 0)) for i in range(10)])
            + ",".join([str(style_decisions.get(i, 0)) for i in range(10)])
            + ",".join([str(content_decisions.get(i, 0)) for i in range(10)])
            + "\n"
        )

    return accuracies, style_decisions, content_decisions


def get_stl10_data_loader(
    data_root,
    download,
    shuffle=False,
    model_type=ModelType.SIMCLR,
    batch_size=256,
    stylization=False,
    stylized_folder_path=None,
):
    if not stylization or not stylized_folder_path:
        print("[INFO] Preparing STL10 data loader.")

        if model_type == ModelType.VIT:
            stl10_transform = transforms.Compose(
                [transforms.ToTensor(), feature_extractor]
            )
        elif model_type == ModelType.SIMCLR:
            stl10_transform = transforms.ToTensor()

        test_dataset = datasets.STL10(
            data_root, split="test", download=download, transform=stl10_transform
        )
    else:
        print("[INFO] Preparing stylized STL10 data loader.")

        test_dataset = StylizedSTL10Dataset(stylized_folder_path)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=10,
        drop_last=False,
        shuffle=shuffle,
    )
    return test_loader


def get_stl10_canny_dual_data_loader(
    data_root,
    download,
    shuffle=False,
    batch_size=256,
    model_type=ModelType.SIMCLR,
    stylization=False,
    stylized_folder_path=None,
    **kwarg
):
    if not stylization or not stylized_folder_path:
        print("[INFO] Preparing STL10 canny data loader.")

        test_dataset = DualDataset(
            CannyDataset(root=data_root, split="test", transform=transforms.ToTensor()),
            STL10(root=data_root, split="test", transform=transforms.ToTensor()),
        )
    else:
        print("[INFO] Preparing stylized STL10 canny data loader.")

        test_dataset = DualDataset(
            CannyStylizedDataset(
                source_dir=stylized_folder_path,
                model_type=model_type,
            ),
            StylizedSTL10Dataset(
                source_dir=stylized_folder_path,
                model_type=model_type,
            ),
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=10,
        drop_last=False,
        shuffle=shuffle,
    )

    return test_loader


def get_stl10_dexined_dual_data_loader(
    data_root,
    download,
    shuffle=False,
    batch_size=256,
    model_type=ModelType.SIMCLR,
    stylization=False,
    stylized_folder_path=None,
    **kwarg
):
    if not stylization or not stylized_folder_path:
        print("[INFO] Preparing STL10 DexiNed data loader.")

        test_dataset = DualDataset(
            DexiNedTestDataset(
                csv_file="./Edge_images/Dexi/test/labels.csv",
                root_dir="./Edge_images/Dexi/test",
                transform=transforms.ToTensor(),
            ),
            STL10(root=data_root, split="test", transform=transforms.ToTensor()),
        )
    else:
        print("[INFO] Preparing stylized STL10 DexiNed data loader.")

        test_dataset = DualDataset(
            DexiNedStylizedTestDataset(
                csv_file="./Edge_images/Dexi_stylized/labels.csv",
                root_dir="./Edge_images/Dexi_stylized",
                transform=transforms.ToTensor(),
            ),
            StylizedSTL10Dataset(
                source_dir=stylized_folder_path,
                model_type=model_type,
            ),
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=10,
        drop_last=False,
        shuffle=shuffle,
    )

    return test_loader


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad, has_canny=False):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    if has_canny:
        perturbed_image = torch.stack(
            [
                torch.clamp(perturbed_image[0], 0, 255),
                torch.clamp(perturbed_image[1], 0, 1),
            ]
        )
    else:
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test_adversarial(
    model, criterion, device, test_loader, epsilon, model_type=ModelType.SIMCLR
):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Important: Set model to eval(), otherwise the dropout might cause the result inaccurate
    model.eval()

    # Loop over all examples in test set
    for data_and_target in tqdm(test_loader):
        # Send the data and label to the device
        if model_type == ModelType.VIT:
            data, target = data_and_target
            data = data["pixel_values"][0]
        elif (
            model_type == ModelType.MORPHCLRSINGLE
            or model_type == ModelType.MORPHCLRDUAL
        ):
            edge_data, non_edge_data, target = data_and_target
            if len(edge_data.shape) == 3:
                edge_data = edge_data.unsqueeze(1)
            # If the image is of grayscale, repeat the dimension to create 3 channels
            if edge_data.shape[1] == 1:
                edge_data = edge_data.repeat(1, 3, 1, 1)
            data = torch.stack([edge_data, non_edge_data], dim=0)
            target = target.flatten()
        else:
            data, target = data_and_target

        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = criterion(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        has_canny = (
            type(test_loader.dataset) == DualDataset
            and type(test_loader.dataset.dataset1) == CannyDataset
        )
        perturbed_data = fgsm_attack(data, epsilon, data_grad, has_canny)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print(
        "Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
            epsilon, correct, len(test_loader), final_acc
        )
    )

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def compute_adversarial_accuracies(
    adv_epsilons,
    model,
    device,
    test_loader,
    model_name,
    model_type=ModelType.SIMCLR,
):
    print("[INFO] Computing adversarial accuracies and epsilons.")
    result_root = "./adv_results"

    if not os.path.exists(result_root):
        os.makedirs(result_root)

    adv_accuracies = []
    adv_examples = []

    criterion = torch.nn.CrossEntropyLoss().to(device)

    with open(os.path.join(result_root, "{}.csv".format(model_name)), "a") as f:
        f.write("eps,acc\n")

    # Run test for each epsilon
    for eps in adv_epsilons:
        acc, ex = test_adversarial(
            model, criterion, device, test_loader, eps, model_type
        )
        adv_accuracies.append(acc)
        adv_examples.append(ex)
        with open(os.path.join(result_root, "{}.csv".format(model_name)), "a") as f:
            f.write("{},{}\n".format(eps, acc))

    plt.figure(figsize=(5, 5))
    plt.plot(adv_epsilons, adv_accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 0.06, step=0.01))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.savefig(
        os.path.join(result_root, "{}.png".format(model_name)), bbox_inches="tight"
    )

    return adv_accuracies, adv_examples


def main():
    args = parser.parse_args()
    # Check if GPU is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda:" + str(args.gpu_index))
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device("cpu")
    # Set model type
    if args.model_type == "simclr":
        model_type = ModelType.SIMCLR
    elif args.model_type == "morphclr_single":
        model_type = ModelType.MORPHCLRSINGLE
    elif args.model_type == "morphclr_dual":
        model_type = ModelType.MORPHCLRDUAL
    else:
        model_type = ModelType.VIT

    # print("[INFO] Downloading Stylized STL10")

    # file_id = "1aTVhLVG1pbsFWmoV-KoYB00YSPCSqyEv"
    # gdrive_url = "https://drive.google.com/uc?id={}".format(file_id)
    stylized_folder_name = "inter_class_stylized_dataset"
    stylized_folder_path = "stylization/{}".format(stylized_folder_name)
    # zip_name = stylized_folder_name + ".zip"
    # gdown.download(gdrive_url, zip_name, quiet=False)
    # shutil.unpack_archive(zip_name, stylized_folder_path)
    # os.remove(zip_name)

    print("[INFO] Starting evaluation...")

    model_path = os.path.join("./checkpoints/finetune/", args.checkpoint)

    print("[INFO] CUDA Device: {}".format(args.device))
    print("[INFO] Using fine-tuned checkpoint: {}".format(model_path))

    if model_type == ModelType.SIMCLR:
        model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(
            args.device
        )
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        get_dataloader_fn = get_stl10_data_loader
    elif model_type == ModelType.MORPHCLRSINGLE:
        model = MorphCLRSingleEval(base_model="resnet18")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        if args.dataset_name.endswith("canny"):
            get_dataloader_fn = get_stl10_canny_dual_data_loader
        elif args.dataset_name.endswith("dexined"):
            get_dataloader_fn = get_stl10_dexined_dual_data_loader
        else:
            raise Exception("Cannot use standard STL10 dataset for MorphCLR models.")
    elif model_type == ModelType.MORPHCLRDUAL:
        model = MorphCLRDualEval(base_model="resnet18")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        if args.dataset_name.endswith("canny"):
            get_dataloader_fn = get_stl10_canny_dual_data_loader
        elif args.dataset_name.endswith("dexined"):
            get_dataloader_fn = get_stl10_dexined_dual_data_loader
        else:
            raise Exception("Cannot use standard STL10 dataset for MorphCLR models.")

    model = model.to(args.device)
    model = model.eval()

    print("[INFO] Model checkpoint loaded.")

    # Evaluate for standard STL10 accuracies
    test = get_dataloader_fn(
        data_root=args.data, download=True, batch_size=args.batch_size
    )
    stl10_accuracies = compute_accuracies_local(
        model=model,
        device=args.device,
        data_loader=test,
        model_name=args.checkpoint.split(".")[0],
        model_type=model_type,
    )

    # Evaluate for stylized STL10 accuracies
    stylized_loader = get_dataloader_fn(
        data_root=args.data,
        download=False,
        stylization=True,
        stylized_folder_path=stylized_folder_path,
        batch_size=args.batch_size,
        model_type=model_type,
    )
    stl10_stylized_metrics = compute_accuracy_and_ratio(
        model=model,
        device=args.device,
        data_loader=stylized_loader,
        model_name=args.checkpoint.split(".")[0],
        model_type=model_type,
    )

    # Evaluate for adversarial accuracies
    test = get_dataloader_fn(data_root=args.data, download=False, batch_size=1)
    compute_adversarial_accuracies(
        adv_epsilons=[0, 0.01, 0.02, 0.03, 0.04, 0.05],
        model=model,
        device=args.device,
        test_loader=test,
        model_name=args.checkpoint.split(".")[0],
        model_type=model_type,
    )

    # model = VIT_pretrained("VIT_checkpoints/VIT_5_epochs.pt", device=device)

    # test = get_dataloader_fn(data_root=args.data, download=True, batch_size=32, model_type=ModelType.VIT)
    # # stl10_accuracies = compute_accuracies_local(model, device, test, model_type=ModelType.VIT)

    # stl10_stylized = StylizedSTL10Dataset(stylized_folder_path, model_type=ModelType.VIT)
    # stylized_loader = DataLoader(
    #     stl10_stylized,
    #     batch_size=32,
    #     num_workers=10,
    #     drop_last=False,
    #     shuffle=False,
    # )
    # stl10_stylized_metrics = compute_accuracy_and_ratio(model, device, stylized_loader)

    # # Evaluate for adversarial accuracies
    # test = get_dataloader_fn(data_root=args.data, download=False, model_type=ModelType.VIT)
    # compute_adversarial_accuracies([0, 0.01, 0.02, 0.03, 0.04, 0.05], model, device, test, model_type=ModelType.VIT)


if __name__ == "__main__":
    main()
