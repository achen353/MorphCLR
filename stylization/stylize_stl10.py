#!/usr/bin/env python
import argparse
from function import adaptive_instance_normalization
import net
from pathlib import Path
from PIL import Image
import random
import torch
import torch.nn as nn
import torchvision.transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description='This script applies the AdaIN style transfer method to arbitrary datasets.')
parser.add_argument('--content-dir', type=str,
                    help='Directory path to a batch of content images')
# parser.add_argument('--style-dir', type=str,
#                     help='Directory path to a batch of style images')
parser.add_argument('--output-dir', type=str, default='output',
                    help='Directory to save the output images')
parser.add_argument('--num-styles', type=int, default=1, help='Number of styles to \
                        create for each image (default: 1)')
parser.add_argument('--alpha', type=float, default=1.,
                    help='The weight that controls the degree of \
                          stylization. Should be between 0 and 1')

# Advanced options
parser.add_argument('--content-size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style-size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', type=int, default=0,
                    help='If set to anything else than 0, center crop of this size will be applied to the content image \
                    after resizing in order to create a squared image (default: 0)')

random.seed(0)

def input_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(torchvision.transforms.Resize(size))
    if crop != 0:
        transform_list.append(torchvision.transforms.CenterCrop(crop))
    transform_list.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def main():
    args = parser.parse_args()

    source_dir = args.content_dir

    output_dir = args.output_dir

    data_paths = []
    stylization_paths = []
    labels = []
    for source_path in sorted(os.listdir(source_dir)):

        labels.append(int(source_path) - 1)

        source_example_path = sorted(os.listdir(os.path.join(source_dir, source_path)))

        source_example_path = [os.path.join(source_dir, source_path, x) for x in source_example_path]
        data_paths.append(source_example_path)

        style_paths = []
        for style_path in sorted(os.listdir(source_dir)):
            if int(style_path) -1 != int(source_path) - 1:
                style_example_path = sorted(os.listdir(os.path.join(source_dir, style_path)))
                style_example_path = [os.path.join(source_dir, style_path, x) for x in style_example_path]
                style_paths += style_example_path
        stylization_paths.append(style_paths)



    decoder = net.decoder
    vgg = net.vgg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('models/decoder.pth'))
    vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    content_tf = input_transform(args.content_size, args.crop)
    style_tf = input_transform(args.style_size, 0)


    # disable decompression bomb errors
    Image.MAX_IMAGE_PIXELS = None
    skipped_imgs = []

    # actual style transfer as in AdaIN
    for content_paths, styles, label in zip(data_paths, stylization_paths, labels):
        with tqdm(total=len(content_paths)) as pbar:
            for content_path in content_paths:
                content_img = Image.open(content_path).convert('RGB')
                for style_path in random.sample(styles, args.num_styles):
                    style_img = Image.open(style_path).convert('RGB')

                    content = content_tf(content_img)
                    style = style_tf(style_img)
                    style = style.to(device).unsqueeze(0)
                    content = content.to(device).unsqueeze(0)
                    with torch.no_grad():
                        output = style_transfer(vgg, decoder, content, style,
                                                args.alpha)
                    output = output.cpu()

                    save_dir = os.path.join(output_dir, str(label + 1))
                    try:
                        os.mkdir(save_dir)
                    except FileExistsError:
                        pass
                    source_name = content_path.split('/')[-1].split('.')[0]
                    style_category = style_path.split('/')[1]

                    save_name = f'source_{source_name}_style_{style_category}_category_{label}.png'
                    output_path = os.path.join(save_dir, save_name)

                    save_image(output, output_path, padding=0) #default image padding is 2.
                    style_img.close()
                content_img.close()
                pbar.update(1)

if __name__ == '__main__':
    main()
