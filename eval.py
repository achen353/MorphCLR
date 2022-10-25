import torch
from torch import cuda
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
from PIL import Image


convert_tensor = torchvision.transforms.ToTensor()

def get_local_dataset(source_dir):
    # assumes that source dir contains folders of images where each folder of images
    # has images from the same class whose name is the name of the folder
    # e.g dir/0 (label) / img_x.png
    data = []
    labels = []
    print("Collecting local images from", source_dir)
    for source_path in sorted(os.listdir(source_dir)):
        labels.append(int(source_path) - 1)
        source_example_path = sorted(os.listdir(os.path.join(source_dir, source_path)))
        source_example_path = [os.path.join(source_dir, source_path, x) for x in source_example_path]

        original_tensors = torch.stack([convert_tensor(Image.open(x).convert('RGB')) for x in source_example_path])

        data.append(original_tensors)
    print(f"Found {len(labels)} labels: {labels}")
    return data, labels


def compute_accuracies_local(model, device, data, labels):
    # logit dim equals number of classes
    assert model(data[0]).shape[-1] == len(labels)
    model = model.eval()
    accuracies = {}
    for x, y in zip(data, labels):
        preds = model(x.to(device))
        accuracy = torch.sum(torch.argmax(preds, dim = 1) == y)/preds.shape[0]
        accuracies[y] = float(accuracy.cpu())
    return accuracies

def main():
    device = "cpu"

    model_path = "stylization/linear_eval_checkpoint.pt"

    model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval()

    data, labels = get_local_dataset("stylization/stylized_output")

    print(compute_accuracies_local(model, device, data, labels ))


if __name__ == "__main__":
    main()

    
