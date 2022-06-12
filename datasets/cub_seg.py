import torch
import PIL
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision.datasets as tvdataset
from datasets.tfs import get_cub_seg_transform

import pdb
import cv2


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=False, loader=pil_loader):
        img_folder = os.path.join(root, "images")
        self.mask_folder = os.path.join(root, "masks")
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
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        print('num of data:{}'.format(len(imgs)))

        self.ref_path = r'/path_to_data/weakly_mask/benchmarks/Attributional-Robustness/WSOL_CUB/' \
                        r'train_log/resnet_beta50_eps2/results_best'

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
        img = self.loader(os.path.join(self.root, file_path))
        img = self.transform(img)
        if not self.train:
            ref_path = os.path.join(self.ref_path, 'mask_HEAT_TEST_1_' + str(index) + '.jpg')
            mask_ref = cv2.imread(ref_path, 0)

            file_path = file_path.split('.jpg')[0] + '.png'
            mask_path = os.path.join(self.mask_folder, file_path)
            mask = cv2.imread(mask_path, 0)
            mask[mask <= 128] = 0
            mask[mask > 128] = 255
            mask = self.target_transform(mask)
            return img, target, mask, mask_ref
        else:
            return img, target

    def __len__(self):
        if self.train:
            return len(self.imgs) * 1
        else:
            return len(self.imgs) * 1


def get_cub_seg_dataset(size=224):
    datadir = '/path_to_data/CUB/CUB_200_2011'
    transform_train, transform_test, transform_mask = get_cub_seg_transform(size)
    ds_train = ImageLoader(datadir, train=True, transform=transform_train)
    ds_test = ImageLoader(datadir, train=False, transform=transform_test, target_transform=transform_mask)
    return ds_train, ds_test


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
    import torch

    ds_train, ds_test = get_dataset()
    ds = torch.utils.data.DataLoader(ds_test,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=False,
                                     drop_last=False)
    pbar = tqdm(ds)
    # for i, (real_imgs, labels) in enumerate(pbar):
    #     image = real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    #     image = (image - image.min()) / (image.max() - image.min())
    #     image = np.array(255*image).copy().astype(np.uint8)

    ref_path = r'/path_to_data/weakly_mask/benchmarks/Attributional-Robustness/WSOL_CUB/' \
               r'train_log/resnet_beta50_eps2/results_best'
    a = os.listdir(ref_path)
    a.sort()
    for i, (real_imgs, labels, mask, mask_ref) in enumerate(pbar):
        image = real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        mask = mask.squeeze().detach().cpu().numpy()
        mask_ref = mask_ref.squeeze().detach().cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        image = np.array(255*image).copy().astype(np.uint8)
        mask = np.array(255*mask).copy().astype(np.uint8)

        cv2.imwrite('kaki.jpg', image)
        cv2.imwrite('kaki_mask.jpg', mask)
        cv2.imwrite('ref_mask.jpg', mask_ref)

