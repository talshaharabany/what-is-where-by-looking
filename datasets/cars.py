import torch
import PIL
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from scipy.io import loadmat
from datasets.tfs import get_car_transform
import pdb


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def get_mat_frame(path, img_folder):
    results = {}
    tmp_mat = loadmat(path)
    anno = tmp_mat['annotations'][0]
    results['path'] = [os.path.join(img_folder, anno[i][-1][0]) for i in range(anno.shape[0])]
    results['label'] = [anno[i][-2][0, 0] for i in range(anno.shape[0])]
    results['x1'] = [anno[i][0][0, 0] for i in range(anno.shape[0])]
    results['y1'] = [anno[i][1][0, 0] for i in range(anno.shape[0])]
    results['x2'] = [anno[i][2][0, 0] for i in range(anno.shape[0])]
    results['y2'] = [anno[i][3][0, 0] for i in range(anno.shape[0])]
    return results


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self,
                 root='Stanford_Cars',
                 transform=None,
                 target_transform=None,
                 train=False,
                 loader=pil_loader,
                 is_bbox=False):
        img_folder = root
        pd_train = pd.DataFrame.from_dict(get_mat_frame(os.path.join(root, 'devkit', 'cars_train_annos.mat'), 'cars_train'))
        pd_test = pd.DataFrame.from_dict(get_mat_frame(os.path.join(root, 'devkit', 'cars_test_annos_withlabels.mat'), 'cars_test'))
        data = pd.concat([pd_train, pd_test])
        data['train_flag'] = pd.Series(data.path.isin(pd_train['path']))
        data = data[data['train_flag'] == train]
        data['label'] = data['label'] - 1
        imgs = data.reset_index(drop=True)
        if len(imgs) == 0:
            raise(RuntimeError("no csv file"))
        self.root = img_folder
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.is_bbox = is_bbox
        self.text_class = ['car', 'sky', 'garden', 'road', 'tree', 'highway', 'signs', 'persons']

    def __getitem__(self, index):
        item = self.imgs.iloc[index]
        file_path = item['path']
        target = item['label']
        img = self.loader(os.path.join(self.root, file_path))
        image_sizes = (img.width, img.height)
        img = self.transform(img)
        if self.is_bbox:
            x1 = item['x1']
            x2 = item['x2']
            y1 = item['y1']
            y2 = item['y2']
            resize = img.shape[1] * 8 / 7
            shift = img.shape[1] / 14
            max_value = img.shape[1] - 1
            image_width, image_height = image_sizes
            left_bottom_x = np.maximum(x1 / image_width * resize - shift, 0)
            left_bottom_y = np.maximum(y1 / image_height * resize - shift, 0)
            right_top_x = np.minimum(x2 / image_width * resize - shift, max_value)
            right_top_y = np.minimum(y2 / image_height * resize - shift, max_value)
            resized_bbox = [left_bottom_x, left_bottom_y, right_top_x, right_top_y]
            return img, target, resized_bbox
        else:
            return img, target

    def __len__(self):
        return len(self.imgs)


def get_cars_dataset(size=448, is_bbox=False, datadir=''):
    transform_train, transform_test = get_car_transform(size=size)
    ds_train = ImageLoader(datadir, train=True, transform=transform_train)
    ds_test = ImageLoader(datadir, train=False, transform=transform_test, is_bbox=is_bbox)
    return ds_train, ds_test


if __name__ == "__main__":
    from tqdm import tqdm
    import cv2
    ds_train, ds_test = get_cars_dataset(is_bbox=True)
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
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite('kaki.jpg', image)