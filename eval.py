import torch
from torch import cuda
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
from PIL import Image

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

from transformers import ViTFeatureExtractor

from vit import VIT_pretrained

from tqdm import tqdm
import gdown
import shutil

import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

convert_tensor = torchvision.transforms.ToTensor()

model_type = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_type)

simclr_type = "simclr"
vit_type = "vit"

from collections import defaultdict

# def get_local_dataset(source_dir, model_type = simclr_type):
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
#         tensor_examples = [convert_tensor(Image.open(x).convert('RGB')) for x in source_example_path]
#         if model_type == simclr_type:
#             original_tensors = torch.stack(tensor_examples)
#         elif model_type == vit_type:
#             vit_extracted = feature_extractor(tensor_examples, return_tensors = "pt")
#             original_tensors = vit_extracted['pixel_values']

#         data.append(original_tensors)
#     print("Found {} labels: {}".format(len(labels), labels))
#     return data, labels
# class stylized_dataset(torch.utils.data.dataset):


def compute_accuracies_local(model, device, data_loader, model_type=simclr_type):
    # logit dim equals number of classes
    model = model.eval()
    model = model.to(device)
    accuracies = defaultdict(int)
    for data, target in tqdm(data_loader):
        # Send the data and label to the device
        if model_type == vit_type:
            data = data["pixel_values"][0]
        data, target = data.to(device), target.to(device)

        # Forward pass the data through the model
        output = model(data)
        preds = torch.argmax(output, dim = 1)

        for i in range(output.shape[0]):
            if preds[i] == target[i]:
                accuracies[int(preds[i])] += 1/800
    print(f'Accuracies : {accuracies}')
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

#         original_tensors = torch.stack([convert_tensor(Image.open(x).convert('RGB')) for x in source_example_path])

#         data.append(original_tensors)
    
#     return data, style_labels, content_labels

class stylized_stl10_dataset(torch.utils.data.Dataset):
    def __init__(self, source_dir, model_type = simclr_type):
        data = []
        style_labels = []
        content_labels = []
        print("[INFO] Preparing stylzed images from: {}".format(source_dir))
        for source_path in sorted(os.listdir(source_dir)):
            source_example_path = sorted(os.listdir(os.path.join(source_dir, source_path)))
            source_example_path = [os.path.join(source_dir, source_path, x) for x in source_example_path]
            dir_style_labels = torch.tensor([int(x.split('/')[-1].split('_')[3]) for x in source_example_path])
            style_labels.append(dir_style_labels)
            content_style_labels = torch.tensor([int(source_path) - 1] * dir_style_labels.shape[0])
            content_labels.append(content_style_labels)
            tensor_images = [convert_tensor(Image.open(x).convert('RGB')) for x in source_example_path]

            if model_type == vit_type:
                vit_extracted = feature_extractor(tensor_images, return_tensors = "pt")
                tensor_images = vit_extracted['pixel_values']
            elif model_type == simclr_type:
                tensor_images = torch.stack(tensor_images)

            data.append(tensor_images)
        data = torch.cat(data, dim=0)
        style_labels = torch.cat(style_labels, dim=0).reshape(-1, 1)
        content_labels = torch.cat(content_labels, dim=0).reshape(-1, 1)
        target = torch.cat((style_labels, content_labels), dim = -1)

        self.data = data
        self.target = target

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

        


def compute_accuracy_and_ratio(model, device, data_loader,):
    # logit dim equals number of classes
    print("[INFO] Computing accuracy and style-based decision ratio.")
    
    model = model.eval()
    model = model.to(device)
    
    accuracies = defaultdict(int)
    style_decisions = defaultdict(int)
    content_decisions = defaultdict(int)
    for data, target in tqdm(data_loader):
        y_style, y_content = target[:, 0], target[:, 1]
        y_style = y_style.to(device)
        y_content = y_content.to(device)
        preds = model(data.to(device))
        preds = torch.argmax(preds, dim = 1)
        for i in range(preds.shape[0]):
            if preds[i] == y_content[i]:
                accuracies[int(preds[i])] += 1/800
                content_decisions[int(preds[i])] += 1
            
            if preds[i] == y_style[i]:
                style_decisions[int(preds[i])] += 1

    print("Accuracy: {}".format(accuracies))
    print("# of style-based decision per class: {}".format(style_decisions))
    print("# of texture-based decision per class: {}".format(content_decisions))

    return accuracies, style_decisions, content_decisions

def get_stl10_data_loaders(download, shuffle=False, model_type = simclr_type, batch_size=256, test_loader_same_batch = False):
    print("[INFO] Preparing STL10 adversarial data loaders.")

    if model_type == vit_type:
        stl10_transform = transforms.Compose([transforms.ToTensor(), feature_extractor])
    elif model_type == simclr_type:
        stl10_transform = transforms.ToTensor()
    train_dataset = datasets.STL10(
        "./datasets", split="train", download=download, transform=stl10_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=False,
        shuffle=shuffle,
    )

    test_dataset = datasets.STL10(
        "./datasets", split="test", download=download, transform=stl10_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1 if not test_loader_same_batch else batch_size,  # batch_size = 1 instead of 2 * batch_size
        num_workers=10,
        drop_last=False,
        shuffle=shuffle,
    )
    return train_loader, test_loader

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test_adversarial(model, criterion, device, test_loader, epsilon, model_type = simclr_type):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Important: Set model to eval(), otherwise the dropout might cause the result inaccurate
    model.eval()

    # Loop over all examples in test set
    for data, target in tqdm(test_loader):
        # Send the data and label to the device
        if model_type == vit_type:
            data = data['pixel_values'][0]

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
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

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

def compute_adversarial_accuracies(adv_epsilons, model, device, test_loader, model_type = simclr_type):
    print("[INFO] Computing adversarial accuracies and epsilons.")

    adv_accuracies = []
    adv_examples = []

    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Run test for each epsilon
    for eps in adv_epsilons:
        acc, ex = test_adversarial(model, criterion, device, test_loader, eps, model_type)
        adv_accuracies.append(acc)
        adv_examples.append(ex)
    return adv_accuracies, adv_examples

def main():
    print("[INFO] Downloading Stylized STL10")
    
    file_id = "1aTVhLVG1pbsFWmoV-KoYB00YSPCSqyEv"
    gdrive_url = "https://drive.google.com/uc?id={}".format(file_id)
    stylized_folder_name = "inter_class_stylized_dataset"
    stylized_folder_path = "stylization/{}".format(stylized_folder_name)
    zip_name = stylized_folder_name + ".zip"
    gdown.download(gdrive_url, zip_name, quiet=False)
    shutil.unpack_archive(zip_name, stylized_folder_path)
    os.remove(zip_name)

    print("[INFO] Starting evaluation...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./checkpoints/finetune/simclr_resnet50_50-epochs_stl10_100-epochs.pt"

    print("[INFO] CUDA Device: {}".format(device))
    print("[INFO] Using fine-tuned checkpoint: {}".format(model_path))

    model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval()

    print("[INFO] Model checkpoint loaded.")

    train, test = get_stl10_data_loaders(download=True, batch_size=32, test_loader_same_batch=True)
    stl10_accuracies = compute_accuracies_local(model, device, test, )

    stl10_stylized = stylized_stl10_dataset(stylized_folder_path)
    stylized_loader = DataLoader(
        stl10_stylized,
        batch_size=32,
        num_workers=10,
        drop_last=False,
        shuffle=True,
    )
    stl10_stylized_metrics = compute_accuracy_and_ratio(model, device, stylized_loader)

    # Evaluate for adversarial accuracies
    _, test = get_stl10_data_loaders(download=False)
    compute_adversarial_accuracies([0, 0.01, 0.02, 0.03, 0.04, 0.05], model, device, test,)

    # model = VIT_pretrained("VIT_checkpoints/VIT_5_epochs.pt", device=device)

    # train, test = get_stl10_data_loaders(download=True, batch_size=32, test_loader_same_batch=True, model_type=vit_type)
    # # stl10_accuracies = compute_accuracies_local(model, device, test, model_type=vit_type )

    # stl10_stylized = stylized_stl10_dataset(stylized_folder_path, model_type=vit_type)
    # stylized_loader = DataLoader(
    #     stl10_stylized,
    #     batch_size=32,
    #     num_workers=10,
    #     drop_last=False,
    #     shuffle=True,
    # )
    # stl10_stylized_metrics = compute_accuracy_and_ratio(model, device, stylized_loader)

    # # Evaluate for adversarial accuracies
    # _, test = get_stl10_data_loaders(download=False, model_type=vit_type)
    # compute_adversarial_accuracies([0, 0.01, 0.02, 0.03, 0.04, 0.05], model, device, test, model_type=vit_type)



if __name__ == "__main__":
    main()

    
