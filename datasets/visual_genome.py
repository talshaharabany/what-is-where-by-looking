import pickle
import json
import torch
from PIL import Image
import os
import pandas as pd
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from datasets.tfs import get_flicker_transform
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm
from utils_grounding import union
from random import shuffle


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# class ImageLoader(torch.utils.data.Dataset):
#     def __init__(self, root, transform=None, split="train", loader=pil_loader):
#
#         self.imgs_data_folder = os.path.join(root, "VG_Annotations", "imgs_data.pickle")
#         self.splits_folder = os.path.join(root, "VG_Annotations", "data_splits.pickle")
#         self.annotations_folder = os.path.join(root, "VG_Annotations", "region_descriptions.json")
#         with open(self.annotations_folder, 'rb') as f:
#             self.annotations = json.load(f, encoding='latin1')
#         with open(self.splits_folder, 'rb') as f:
#             self.splits = pickle.load(f, encoding='latin1')
#         with open(self.imgs_data_folder, 'rb') as f:
#             self.imgs_data = pickle.load(f, encoding='latin1')
#         self.img_folder = os.path.join(root, "VG_Images")
#
#         self.transform = transform
#         self.loader = loader
#         self.files = list(self.splits[split])
#         self.split = split
#         self.annotations = sync_data(self.files, self.annotations, self.imgs_data, split)
#         self.files = list(self.annotations.keys())
#         print('num of data:{}'.format(len(self.files)))
#
#     def __getitem__(self, index):
#         item = str(self.files[index])
#         img_path = os.path.join(self.img_folder, item + '.jpg')
#         img = pil_loader(img_path)
#         image_sizes = (img.height, img.width)
#         img = self.transform(img)
#         ann = self.annotations[int(item)]
#         if self.split == 'train':
#             region_id = np.random.randint(0, len(ann))
#             return img, ann[region_id]['phrase']
#         return img, ann, image_sizes, img_path
#
#     def __len__(self):
#         return len(self.files) * 1

class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, split="train", loader=pil_loader):

        self.imgs_data_folder = os.path.join(root, "VG_Annotations", "imgs_data.pickle")
        self.splits_folder = os.path.join(root, "VG_Annotations", "data_splits.pickle")
        self.annotations_folder = os.path.join(root, "VG_Annotations", "region_descriptions.json")
        with open(self.annotations_folder, 'rb') as f:
            self.annotations = json.load(f, encoding='latin1')
        with open(self.splits_folder, 'rb') as f:
            self.splits = pickle.load(f, encoding='latin1')
        with open(self.imgs_data_folder, 'rb') as f:
            self.imgs_data = pickle.load(f, encoding='latin1')
        self.img_folder = os.path.join(root, "VG_Images")

        self.transform = transform
        self.loader = loader
        self.files = list(self.splits[split])
        self.split = split
        self.annotations = sync_data(self.files, self.annotations, self.imgs_data, split)
        self.files = list(self.annotations.keys())
        print('num of data:{}'.format(len(self.files)))

    def __getitem__(self, index):
        item = str(self.files[index])
        img_path = os.path.join(self.img_folder, item + '.jpg')
        img = pil_loader(img_path)
        image_sizes = (img.height, img.width)
        img = self.transform(img)
        ann = self.annotations[int(item)]
        if self.split == 'train':
            region_id = np.random.randint(0, len(ann))
            return img, 'image of ' + ann[region_id]['phrase'].lower()
        out = {}
        for i in range(0, len(ann)):
            tmp = {}
            tmp['sentences'] = 'image of ' + ann[i]['phrase'].lower()
            bbox = [[int(ann[i]['x']), int(ann[i]['y']),\
                   int(ann[i]['x']) + int(ann[i]['width']), int(ann[i]['y']) + int(ann[i]['height'])]]
            tmp['bbox'] = bbox
            if (bbox[0][3] - bbox[0][1]) * (bbox[0][2] - bbox[0][0]) > 0.05 * image_sizes[0] * image_sizes[1]:
                out[str(i)] = tmp
        return img, out, image_sizes, img_path

    def __len__(self):
        return len(self.files) * 1


def get_VG_dataset(args):
    datadir = args['data_path']
    transform_train, transform_test = get_flicker_transform(args)
    ds_train = ImageLoader(datadir, split='train', transform=transform_train)
    return ds_train


def get_VGtest_dataset(args):
    datadir = args['val_path']
    transform_train, transform_test = get_flicker_transform(args)
    ds_test = ImageLoader(datadir, split='test', transform=transform_test)
    return ds_test


def sync_data(files, annotations, imgs_data, split='train'):
    out = {}
    for ann in tqdm(annotations):
        if ann['id'] in files:
            tmp = []
            for item in ann['regions']:
                if len(item['phrase'].split(' ')) < 80:
                    tmp.append(item)
            out[ann['id']] = tmp
    return out


if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-Isize', '--Isize', default=224, help='image size', required=False)
    args = vars(parser.parse_args())
    ds = get_VG_dataset(args=args)
    ds = torch.utils.data.DataLoader(ds,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=False,
                                     drop_last=False)
    pbar = tqdm(ds)
    for i, (real_imgs, text) in enumerate(pbar):
        pass
    # for i, (real_imgs, meta, size, img_path) in enumerate(pbar):
    #     size = [int(size[1]), int(size[0])]
    #     image = real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    #     image = (image - image.min()) / (image.max() - image.min())
    #     image = np.array(255*image).copy().astype(np.uint8)
    #     image = cv2.resize(image, size)
    #     for sen in meta:
    #         item = sen['phrase']
    #         bbox = int(sen['x']), int(sen['y']), int(sen['x']) + int(sen['width']), int(sen['y']) + int(sen['height'])
    #         (gxa, gya, gxb, gyb) = bbox
    #         image = cv2.rectangle(image, (gxa, gya), (gxb, gyb), (0, 0, 255), 2)
    #         cv2.imwrite('kaki.jpg', image)
