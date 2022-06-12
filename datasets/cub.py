import torch
import PIL
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision.datasets as tvdataset
from datasets.tfs import get_cub_transform

import pdb


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=False, loader=pil_loader, is_bbox=False, is_siam=False):
        img_folder = os.path.join(root, "images")
        img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,  names=['idx', 'label'])
        train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,  names=['idx', 'train_flag'])
        bounding_box = pd.read_csv(os.path.join(root, "bounding_boxes.txt"), sep=" ", header=None,  names=['idx', 'x', 'y', 'w', 'h'])
        data = pd.concat([img_paths, img_labels, train_test_split, bounding_box], axis=1)
        data['label'] = data['label'] - 1
        alldata = data.copy()
        data = data[data['train_flag'] == train]
        imgs = data.reset_index(drop=True)

        if len(imgs) == 0:
            raise(RuntimeError("no csv file"))
        self.root = img_folder
        self.is_bbox = is_bbox
        self.is_siam = is_siam
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        print('num of data:{}'.format(len(imgs)))
        if self.is_bbox:
            self.bounding_box = bounding_box
            # self.bbox = load_bbox_size(dataset_path=root)
            # print('kaki')

        self.text_class = ['bird', 'sky', 'lake', 'sea', 'tree', 'leaves', 'axes', 'person']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        index = index % len(self.imgs)
        item = self.imgs.iloc[index]
        file_path = item['path']
        target = item['label']
        if self.is_siam:
            file_path_pos, _ = self.get_sample(target, 1)
            file_path_neg, _ = self.get_sample(target, 0)
            img = self.loader(os.path.join(self.root, file_path))
            pos = self.loader(os.path.join(self.root, file_path_pos))
            neg = self.loader(os.path.join(self.root, file_path_neg))
            img = self.transform(img)
            pos = self.transform(pos)
            neg = self.transform(neg)
            return img, pos, neg
        elif self.is_bbox:
            img = self.loader(os.path.join(self.root, file_path))
            image_sizes = (img.width, img.height)
            img = self.transform(img)
            if img.shape[1] == 448:
                resize = 512
                shift = 32
                max_value = 447
            else:
                resize = 256
                shift = 16
                max_value = 223
            x = item['x']
            y = item['y']
            bbox_width = item['w']
            bbox_height = item['h']
            image_width, image_height = image_sizes
            left_bottom_x = np.maximum(x / image_width * resize - shift, 0)
            left_bottom_y = np.maximum(y / image_height * resize - shift, 0)

            right_top_x = np.minimum((x + bbox_width) / image_width * resize - shift, max_value)
            right_top_y = np.minimum((y + bbox_height) / image_height * resize - shift, max_value)
            resized_bbox = [left_bottom_x, left_bottom_y, right_top_x, right_top_y]
            return img, target, resized_bbox
        else:
            img = self.loader(os.path.join(self.root, file_path))
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.train:
            return len(self.imgs) * 1
        else:
            return len(self.imgs) * 1
        # return len(self.imgs) * 1

    def get_sample(self, target, is_pos=True):
        tmp = True
        while tmp:
            index = np.random.randint(0, len(self.imgs))
            item = self.imgs.iloc[index]
            file_path = item['path']
            target_sample = item['label']
            if (is_pos and target_sample == target) or (not is_pos and target_sample != target):
                tmp = False
        return file_path, target_sample


def get_dataset(is_bbox=False, size=224, datadir='', is_siam=False):
    transform_train, transform_test = get_cub_transform(size)
    ds_train = ImageLoader(datadir, train=True, transform=transform_train, is_bbox=is_bbox, is_siam=is_siam)
    ds_test = ImageLoader(datadir, train=False, transform=transform_test, is_bbox=is_bbox, is_siam=is_siam)
    return ds_train, ds_test


def load_bbox_size(dataset_path='datalist', resize_size=512, crop_size=448):
    origin_bbox = {}
    image_sizes = {}
    resized_bbox = {}
    with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
        for each_line in f:
            file_info = each_line.strip().split()
            image_id = int(file_info[0])

            x, y, bbox_width, bbox_height = map(float, file_info[1:])

            origin_bbox[image_id] = [x, y, bbox_width, bbox_height]

    with open(os.path.join(dataset_path, 'sizes.txt')) as f:
        for each_line in f:
            file_info = each_line.strip().split()
            image_id = int(file_info[0])
            image_width, image_height = map(float, file_info[1:])

            image_sizes[image_id] = [image_width, image_height]

    resize_size = float(resize_size-1)
    shift_size = (resize_size - crop_size) // 2
    for i in origin_bbox.keys():
        x, y, bbox_width, bbox_height = origin_bbox[i]
        image_width, image_height = image_sizes[i]
        left_bottom_x = x / image_width * resize_size - shift_size
        left_bottom_y = y / image_height * resize_size - shift_size

        right_top_x = (x+bbox_width) / image_width * resize_size - shift_size
        right_top_y = (y+bbox_height) / image_height * resize_size - shift_size
        resized_bbox[i] = [left_bottom_x, left_bottom_y, right_top_x, right_top_y]
    return resized_bbox


def create_image_sizes_file(dataset_path):
    '''
    save 'sizes.txt' in forms of
    [image_id] [width] [height]
    '''
    import cv2

    image_paths = load_image_path(dataset_path)
    image_sizes = []
    for image_id, image_path in image_paths.items():
        image = cv2.imread(os.path.join(dataset_path, 'images', image_path))
        image_sizes.append([image_id, image.shape[1], image.shape[0]])
    with open(os.path.join(dataset_path, 'sizes.txt'), 'w') as f:
        for image_id, w, h in image_sizes:
            f.write("%s %d %d\n" % (str(image_id), w, h))


def load_image_path(dataset_path):
    '''
    return dict{image_id : image_path}
    '''
    image_paths = {}
    with open(os.path.join(dataset_path, 'images.txt')) as f:
        for line in f:
            image_id, image_path = line.strip().split()
            image_paths[image_id] = image_path
    return image_paths


if __name__ == "__main__":
    from tqdm import tqdm
    import cv2
    ds_train, ds_test = get_dataset(is_bbox=True)
    ds = torch.utils.data.DataLoader(ds_test,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=False,
                                     drop_last=False)
    pbar = tqdm(ds)
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

