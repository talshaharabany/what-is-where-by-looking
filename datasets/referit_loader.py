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
import os, sys, re, pickle, cv2, lmdb, json


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, split="train", loader=pil_loader):
        annt_path = os.path.join(root, 'annotations', split + '.pickle')
        with open(annt_path, 'rb') as f:
            self.annotations = pickle.load(f, encoding='latin1')
        self.files = list(self.annotations.keys())
        print('num of data:{}'.format(len(self.files)))
        self.transform = transform
        self.loader = loader
        self.split = split
        self.img_folder = os.path.join(root, 'ReferIt_Images')

    def __getitem__(self, index):
        item = str(self.files[index])
        folder = (2 - len(str(int(item) // 1000))) * '0' + str(int(item) // 1000)
        img_path = os.path.join(self.img_folder, folder, 'images', item + '.jpg')
        img = pil_loader(img_path)
        image_sizes = (img.height, img.width)
        img = self.transform(img)
        ann = self.annotations[item]['annotations']
        if self.split == 'train':
            region_id = np.random.randint(0, len(ann))
            return img, ann[region_id]['query']
        out = {}
        for i in range(0, len(ann)):
            tmp = {}
            bbox = ann[i]['bbox']
            if (bbox[0][3]-bbox[0][1]) * (bbox[0][2]-bbox[0][0]) > 0.05 * image_sizes[0] * image_sizes[1]:
                tmp['sentences'] = ann[i]['query']
                tmp['bbox'] = bbox
                out[str(i)] = tmp
        return img, out, image_sizes, img_path

    def __len__(self):
        return len(self.files) * 1


def get_refit_test_dataset(args):
    datadir = r'/path_to_data/coco/RefIt'
    transform_train, transform_test = get_flicker_transform(args)
    ds_test = ImageLoader(datadir, split='test', transform=transform_test)
    return ds_test


if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-Isize', '--Isize', default=224, help='image size', required=False)
    args = vars(parser.parse_args())
    ds = get_refit_test_dataset(args=args)
    ds = torch.utils.data.DataLoader(ds,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=False,
                                     drop_last=False)
    pbar = tqdm(ds)
    # for i, (real_imgs, text) in enumerate(pbar):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (real_imgs, meta, size) in enumerate(pbar):
        size = [int(size[1]), int(size[0])]
        for sen in meta.keys():
            image = real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            image = (image - image.min()) / (image.max() - image.min())
            image = np.array(255 * image).copy().astype(np.uint8)
            image = cv2.resize(image, size)
            item = meta[sen]
            text, bbox = item['sentences'], item['bbox']
            bbox = torch.tensor(bbox)

            x1, x2, y1, y2 = [bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]]
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            bbox_norm = [x1 / width, y1 / height, x2 / width, y2 / height]
            if max(bbox_norm) > 1:
                continue
        #     (gxa, gya, gxb, gyb) = list(bbox.squeeze())
        #     image = cv2.rectangle(image, (int(gxa), int(gya)), (int(gxb), int(gyb)), (0, 0, 255), 2)
        #     cv2.putText(image, text[0], (int(gxa)+10, int(gya)), font, fontScale=0.3,
        #                 color=(0, 0, 0),
        #                 thickness=1)
        #     cv2.imwrite('kaki.jpg', image)
        #     pass
        # pass

