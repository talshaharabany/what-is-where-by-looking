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
from pycocotools.coco import COCO


def get_dict_coco(coco_obj, image_objects, caption_objects, names, supercats):
    dictionary = {}
    for n, img_obj in enumerate(image_objects):
        img_id, height, width, filename = (img_obj['id'], img_obj['height'], img_obj['width'], img_obj['file_name'])
        annotations = []
        bbox_annIds = coco_obj.getAnnIds(imgIds=img_id)
        bbox_anns = coco_obj.loadAnns(bbox_annIds)
        for bbox_ann in bbox_anns:
            bbox = bbox_ann['bbox']
            x_min, x_max, y_min, y_max = [bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]]
            bbox = [x_min, y_min, x_max, y_max]
            bbox_norm = [x_min / width, y_min / height, x_max / width, y_max / height]
            category_id = bbox_ann['category_id']
            supercategory = supercats[category_id]
            category_name = names[category_id]
            object_id = bbox_ann['id']
            segmentation = bbox_ann['segmentation']
            bbox_entity = {'image_id': img_id,
                           'bbox': bbox,
                           'bbox_norm': bbox_norm,
                           'supercategory': supercategory,
                           'category': category_name,
                           'category_id': category_id,
                           'obj_id': object_id,
                           'segmentation': segmentation}
            annotations.append(bbox_entity)
        capsAnnIds = caption_objects.getAnnIds(imgIds=img_id);
        caps = caption_objects.loadAnns(capsAnnIds)
        sentences = [cap['caption'] for cap in caps]
        dictionary[str(img_id)] = {'size': (height, width, 3),
                                   'queries': sentences,
                                   'captions': sentences,
                                   'annotations': annotations}
    return dictionary

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, split="train", loader=pil_loader):
        self.captions_file = os.path.join(root, 'annotations', "captions_" + split + "2014.json")
        self.instances_file = os.path.join(root, 'annotations', "instances_" + split + "2014.json")
        self.img_folder = os.path.join(root, 'images', split + "2014")

        coco_obj = COCO(self.instances_file)
        caption_objects = COCO(self.captions_file)
        categories = coco_obj.loadCats(coco_obj.getCatIds())
        names = {}
        supercats = {}
        for cat in categories:
            names[cat['id']] = cat['name']
            supercats[cat['id']] = cat['supercategory']
        imageIds = coco_obj.getImgIds();
        image_objects = coco_obj.loadImgs(imageIds)

        self.annotations = get_dict_coco(coco_obj, image_objects, caption_objects, names, supercats)
        self.files = list(self.annotations.keys())

        self.transform = transform
        self.loader = loader
        self.split = split
        print('num of data:{}'.format(len(self.files)))

    def __getitem__(self, index):
        item = str(self.files[index])
        img_path = os.path.join(self.img_folder, 'COCO_train2014_' + '0'*(12 - len(item)) + item + '.jpg')
        img = pil_loader(img_path)
        img = self.transform(img)
        queries = self.annotations[item]['queries']
        region_id = np.random.randint(0, len(queries))
        a = len(queries[region_id].split(' '))
        mask = np.random.randint(0, 5, a) > 0
        text_list = np.array(queries[region_id].split(' '))[mask].tolist()
        text = ' '.join(text_list)
        return img, 'image of ' + text

    def __len__(self):
        return len(self.files) * 1


def get_coco_dataset(args):
    datadir = args['data_path']
    transform_train, transform_test = get_flicker_transform(args)
    ds_train = ImageLoader(datadir, split='train', transform=transform_train)
    return ds_train


def get_coco_test_dataset(args):
    datadir = args['data_path']
    transform_train, transform_test = get_flicker_transform(args)
    ds_test = ImageLoader(datadir, split='val', transform=transform_test)
    return ds_test


if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-Isize', '--Isize', default=224, help='image size', required=False)
    args = vars(parser.parse_args())
    ds = get_coco_dataset(args=args)
    ds = torch.utils.data.DataLoader(ds,
                                     batch_size=4,
                                     num_workers=0,
                                     shuffle=False,
                                     drop_last=False)
    pbar = tqdm(ds)
    for i, (real_imgs, text) in enumerate(pbar):
        pass
    # for i, (real_imgs, meta, size) in enumerate(pbar):
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
