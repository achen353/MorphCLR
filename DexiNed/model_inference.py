"""
python file to help with inference on DexiNed model
"""
import numpy as np
import torch
import os
import cv2
from DexiNed.utils.image import image_normalization

from DexiNed.config.definitions import ROOT_DIR
from DexiNed.model import DexiNed

model = None


def get_instance(device):
    global model
    if model == None:
        checkpoint_path = os.path.join(
            ROOT_DIR, "checkpoints", "BIPED", "10", "10_model.pth"
        )
        model = DexiNed().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
    return model


def model_process(image, device):
    """
    Simple function to automatically preprocess, run model, and postprocess a numpy array image
    """
    model = get_instance(device)
    image, image_shape = pre_process(image)
    image = image[np.newaxis, :]
    image = image.to(device)
    image = model(image)  # needs to have batch dimension.
    image = post_process_single(image, image_shape)
    return image


def pre_process(image):
    """
    Takes an image and prepares it to be processed by model

    Parameters:
    ----------
    image: ndarray
        array with dimensions width, height, channel
        loaded in BGR from cv2.imread(img_file, cv2.IMREAD_COLOR)
    Returns:
    --------
    Tuple[ndarray, Tuple[int, int]]
        image and orignial width, height, which will be needed to restore image to original dimensions.
    """
    label_path = None
    # goal dimensions
    image_shape = [image.shape[0], image.shape[1]]
    img_height = 512
    img_width = 512

    print(f"actual size: {image.shape}, target size: {( img_height,img_width,)}")
    # img = cv2.resize(img, (self.img_width, self.img_height))
    image = cv2.resize(image, (img_width, img_height))

    image = np.array(image, dtype=np.float32)
    # subtract mean pixel values. values found in main.py default args.
    image -= np.array([103.939, 116.779, 123.68, 137.86])[:3]
    image = image.transpose(
        (2, 0, 1)
    )  # torch does channel, w, h, whereas cv2 loads the channel last.
    image = torch.from_numpy(image.copy()).float()
    gt = None
    gt = np.zeros((image.shape[:2]))
    gt = torch.from_numpy(np.array([gt])).float()

    label = None
    img_dict = dict(image=image)
    return (image, image_shape)


def post_process_single(image, image_shape):
    fuse_name = "fused"
    av_name = "avg"
    tmp_img2 = None
    fuse = None
    edge_maps = []
    for i in image:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
    image = np.array(edge_maps)

    # image_shape = [x.cpu().detach().numpy() for x in image_shape]
    # # (H, W) -> (W, H)
    if not isinstance(image_shape[0], int):
        image_shape = [[y, x] for x, y in zip(image_shape[0], image_shape[1])]
    else:
        image_shape = [image_shape[::-1]]

    for idx, i_shape in enumerate(image_shape):
        tmp = image[:, idx, ...]
        tmp = np.squeeze(tmp)

        # Iterate our all 7 NN outputs for a particular image
        preds = []
        for i in range(tmp.shape[0]):
            tmp_img = tmp[i]
            tmp_img = np.uint8(image_normalization(tmp_img))
            tmp_img = cv2.bitwise_not(tmp_img)
            # tmp_img[tmp_img < 0.0] = 0.0
            # tmp_img = 255.0 * (1.0 - tmp_img)

            # Resize prediction to match input image size
            if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
                tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))
            preds.append(tmp_img)
            # plt.figure()
            # plt.imshow(tmp_img)
            # plt.show()
            if i == 6:
                fuse = tmp_img
                fuse = fuse.astype(np.uint8)
        # Get the mean prediction of all the 7 outputs
        average = np.array(preds, dtype=np.float32)
        average = np.uint8(np.mean(average, axis=0))
    return (fuse, average)


def post_process(tensor, image_shape):
    # TODO: be able to process images in a batch???
    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    device = torch.device("cuda")
    images_dir = os.path.join(ROOT_DIR, "data")
    root_dir, _, image_files = list(os.walk(images_dir))[0]
    print(image_files)
    images = []
    for img_file in image_files:
        image = cv2.imread(os.path.join(root_dir, img_file), cv2.IMREAD_COLOR)[:,:,::-1]
        images.append(image)
        result = model_process(image, device)[0]
        c_low = 370
        c_high = 500
        result_canny = cv2.Canny(image, c_low, c_high)
        fig =plt.figure(figsize=(12,6), dpi = 75)
        plt.subplot(1,3,1)
        plt.title(f"edge image {img_file}")
        plt.imshow(image)
        plt.subplot(1,3,2)
        plt.title("fuse")
        plt.imshow(result)
        plt.subplot(1,3,3)
        plt.title(f"canny {c_low}, {c_high}")
        plt.imshow(result_canny)
        plt.show()
    image = images[-1]
    # fig = plt.figure(figsize=(12, 6), dpi = 75)
    # plt.subplot(1,2,1)
    # plt.title("original")
    # plt.imshow(image[:,:,::-1])
    # plt.subplot(1,2,2)
    # plt.title("fuse")
    # plt.imshow(result)
    # plt.show()
