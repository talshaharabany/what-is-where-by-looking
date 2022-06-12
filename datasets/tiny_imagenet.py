import torch
import PIL
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision.datasets as tvdataset
from datasets.tfs import get_tiny_transform

import pdb


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=False, loader=pil_loader):
        if train:
            img_folder = os.path.join(root, "train")
        else:
            img_folder = os.path.join(root, "val", 'images')
            self.labels = os.listdir(os.path.join(root, "train"))
        self.root = img_folder
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.data_len = len(os.listdir(self.root))
        if not train:
            bounding_box = pd.read_csv(os.path.join(root, "val", 'val_annotations.txt'),
                                       sep="\t", header=None, names=['idx', 'label', 'x', 'y', 'w', 'h'])
            self.bounding_box = set_bbox(bounding_box)

    def __getitem__(self, index):
        index = index % self.data_len
        if self.train:
            target = os.listdir(self.root)[index]
            file = np.random.randint(0, 500)
            folder = os.path.join(self.root, target, 'images')
            file_name = os.listdir(folder)[file]
            img = self.loader(os.path.join(self.root, target, 'images', file_name))
            img = self.transform(img)
            return img, index
        else:
            file_name = os.listdir(self.root)[index]
            img = self.loader(os.path.join(self.root, file_name))
            target = self.labels.index(self.bounding_box[file_name][4])
            left_bottom_x = self.bounding_box[file_name][0]
            left_bottom_y = self.bounding_box[file_name][1]
            right_top_x = self.bounding_box[file_name][2]
            right_top_y = self.bounding_box[file_name][3]

            left_bottom_x = np.maximum(left_bottom_x / 64 * 72 - 4.5, 0)
            left_bottom_y = np.maximum(left_bottom_y / 64 * 72 - 4.5, 0)
            right_top_x = np.minimum(right_top_x / 64 * 72 - 4.5, 63)
            right_top_y = np.minimum(right_top_y / 64 * 72 - 4.5, 63)

            bbox = [left_bottom_x, left_bottom_y, right_top_x, right_top_y]
            img = self.transform(img)
            return img, target, bbox

    def __len__(self):
        if self.train:
            return self.data_len * 500
        else:
            return self.data_len * 1


def get_tiny_dataset(is_bbox=False, size=64, datadir=''):
    transform_train, transform_test = get_tiny_transform()
    ds_train = ImageLoader(datadir, train=True, transform=transform_train)
    ds_test = ImageLoader(datadir, train=False, transform=transform_test)
    return ds_train, ds_test

def set_bbox(bbox):
    out = {}
    for i in range(len(bbox['idx'])):
        name = str(bbox['idx'][i])
        label = str(bbox['label'][i])
        x = bbox['x'][i]
        y = bbox['y'][i]
        w = bbox['w'][i]
        h = bbox['h'][i]
        out[name] = [x, y, w, h, label]
    return out


if __name__ == "__main__":
    from tqdm import tqdm
    import cv2
    ds_train, ds_test = get_tiny_dataset(is_bbox=True)
    ds = torch.utils.data.DataLoader(ds_test,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=False,
                                     drop_last=False)
    pbar = tqdm(ds)
    # for i, (real_imgs, labels) in enumerate(pbar):
    #     image = real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    #     image = (image - image.min()) / (image.max() - image.min())
    #     image = np.array(255 * image).copy().astype(np.uint8)

    for i, (real_imgs, labels, bbox) in enumerate(pbar):
        image = real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        image = np.array(255*image).copy().astype(np.uint8)
        gxa = int(bbox[0])
        gya = int(bbox[1])
        gxb = int(bbox[2])
        gyb = int(bbox[3])
        image = cv2.rectangle(image, (gxa, gya), (gxb, gyb), (0, 0, 255), 2)
        cv2.imwrite('kaki.jpg', image)

