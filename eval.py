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

from tqdm import tqdm
import gdown
import shutil

import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

convert_tensor = torchvision.transforms.ToTensor()

def get_local_dataset(source_dir):
    # assumes that source dir contains folders of images where each folder of images
    # has images from the same class whose name is the name of the folder
    # e.g dir/0 (label) / img_x.png
    data = []
    labels = []
    print("[INFO] Preparing local images from: {}".format(source_dir))
    for source_path in sorted(os.listdir(source_dir)):
        labels.append(int(source_path) - 1)
        source_example_path = sorted(os.listdir(os.path.join(source_dir, source_path)))
        source_example_path = [os.path.join(source_dir, source_path, x) for x in source_example_path]

        original_tensors = torch.stack([convert_tensor(Image.open(x).convert('RGB')) for x in source_example_path])

        data.append(original_tensors)
    print("Found {} labels: {}".format(len(labels), labels))
    return data, labels


def compute_accuracies_local(model, device, data, labels):
    # logit dim equals number of classes
    assert model(data[0]).shape[-1] == len(labels)
    model = model.eval()
    model = model.to(device)
    accuracies = {}
    for x, y in zip(data, labels):
        preds = model(x.to(device))
        accuracy = torch.sum(torch.argmax(preds, dim = 1) == y)/preds.shape[0]
        accuracies[y] = float(accuracy.cpu())
    return accuracies

def get_local_stylized_dataset(source_dir):
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

        original_tensors = torch.stack([convert_tensor(Image.open(x).convert('RGB')) for x in source_example_path])

        data.append(original_tensors)
    
    return data, style_labels, content_labels

def compute_accuracy_and_ratio(model, device, data, style_labels, content_labels):
    # logit dim equals number of classes
    print("[INFO] Computing accuracy and style-based decision ratio.")
    
    model = model.eval()
    model = model.to(device)

    assert model(data[0].to(device)).shape[-1] == len(style_labels)
    
    accuracies = {}
    style_decisions = {}
    content_decisions = {}
    for x, y_style, y_content in zip(data, style_labels, content_labels):
        y_style = y_style.to(device)
        y_content = y_content.to(device)
        preds = model(x.to(device))
        preds = torch.argmax(preds, dim = 1)
        content_count = int(torch.sum(preds == y_content))
        style_count = int(torch.sum(preds == y_style))
        label = int(y_content[0])
        style_decisions[label] = style_count
        content_decisions[label] = content_count
        accuracies[label] = float(content_count/y_content.shape[0])

    print("Accuracy: {}".format(accuracies))
    print("# of style-based decision per class: {}".format(style_decisions))
    print("# of texture-based decision per class: {}".format(content_decisions))

    return accuracies, style_decisions, content_decisions

def get_stl10_data_loaders(download, shuffle=False, batch_size=256):
    print("[INFO] Preparing STL10 adversarial data loaders.")
    train_dataset = datasets.STL10(
        "./datasets", split="train", download=download, transform=transforms.ToTensor()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=False,
        shuffle=shuffle,
    )

    test_dataset = datasets.STL10(
        "./datasets", split="test", download=download, transform=transforms.ToTensor()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # batch_size = 1 instead of 2 * batch_size
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

def test_adversarial(model, criterion, device, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Important: Set model to eval(), otherwise the dropout might cause the result inaccurate
    model.eval()

    # Loop over all examples in test set
    for data, target in tqdm(test_loader):
        # Send the data and label to the device
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

def compute_adversarial_accuracies(adv_epsilons, model, device, test_loader):
    print("[INFO] Computing adversarial accuracies and epsilons.")

    adv_accuracies = []
    adv_examples = []

    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Run test for each epsilon
    for eps in adv_epsilons:
        acc, ex = test_adversarial(model, criterion, device, test_loader, eps)
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

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_path = "./checkpoints/finetune/simclr_resnet50_50-epochs_stl10_100-epochs.pt"

    # print("[INFO] CUDA Device: {}".format(device))
    # print("[INFO] Using fine-tuned checkpoint: {}".format(model_path))

    # model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model = model.eval()

    # print("[INFO] Model checkpoint loaded.")

    # # Evaluate on normal STL10 and stylized STL10
    # x, style, content = get_local_stylized_dataset(stylized_folder_name)
    # compute_accuracy_and_ratio(model, device, x, style, content)

    # # Evaluate for adversarial accuracies
    # _, test_loader = get_stl10_data_loaders(download=True)
    # compute_adversarial_accuracies([0, 0.01, 0.02, 0.03, 0.04, 0.05], model, device, test_loader)

if __name__ == "__main__":
    main()

    
