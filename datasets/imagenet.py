import torch
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def load_test_bbox(root, test_gt_path, crop_size, resize_size):
    test_gt = []
    test_txt = []
    shift_size = (resize_size - crop_size) // 2
    with open(test_gt_path, 'r') as f:
        for line in tqdm(f):
            temp_gt = []
            part_1, part_2 = line.strip('\n').split(';')
            img_path, w, h, _ = part_1.split(' ')
            part_2 = part_2[1:]
            bbox = part_2.split(' ')
            bbox = np.array(bbox, dtype=np.float32)
            box_num = len(bbox) // 4
            w, h = np.float32(w), np.float32(h)
            for i in range(box_num):
                bbox[4 * i], bbox[4 * i + 1], bbox[4 * i + 2], bbox[4 * i + 3] = bbox[
                                                                                     4 * i] / w * resize_size - shift_size, \
                                                                                 bbox[
                                                                                     4 * i + 1] / h * resize_size - shift_size, \
                                                                                 bbox[
                                                                                     4 * i + 2] / w * resize_size - shift_size, \
                                                                                 bbox[
                                                                                     4 * i + 3] / h * resize_size - shift_size
                if bbox[4 * i] < 0:
                    bbox[4 * i] = 0
                if bbox[4 * i + 1] < 0:
                    bbox[4 * i + 1] = 0
                if bbox[4 * i + 2] > crop_size:
                    bbox[4 * i + 2] = crop_size
                if bbox[4 * i + 3] > crop_size:
                    bbox[4 * i + 3] = crop_size
                temp_gt.append([bbox[4 * i], bbox[4 * i + 1], bbox[4 * i + 2], bbox[4 * i + 3]])
            test_gt.append(temp_gt)
            # img_path = img_path.replace("\\\\", "\\")
            img_path = img_path.split('\\\\')[1]
            test_txt.append(img_path)
    final_dict = {}
    for k, v in zip(test_txt, test_gt):
        k = os.path.join(root, 'val', k)
        k = k.replace('/', '\\')
        final_dict[k] = v
    return final_dict


class ImageDataset(data.Dataset):
    def __init__(self, root, phase='train'):
        self.root = root
        self.test_txt_path = 'datasets/val_list.txt'
        self.test_gt_path = 'datasets/val_gt.txt'
        self.crop_size = 224
        self.resize_size = 256
        self.phase = phase
        self.num_classes = 1000
        self.text_class = get_cat()
        if self.phase == 'train':
            self.img_dataset = ImageFolder(os.path.join(self.root, 'train'))
            self.transform = transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
            self.label_classes = []
            for k, v in self.img_dataset.class_to_idx.items():
                self.label_classes.append(k)
            self.img_dataset = self.img_dataset.imgs
            print('num of data:{}'.format(len(self.img_dataset)))

        elif self.phase == 'test':
            # self.img_dataset = ImageFolder(os.path.join(self.root, 'val'))
            img_folder = os.path.join(root, "val")
            self.img_dataset = prepare(file=self.test_txt_path, rootdir=img_folder)
            self.transform = transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
            self.test_bbox = load_test_bbox(self.root, self.test_gt_path, self.crop_size, self.resize_size)
            print('done!')


    def __getitem__(self, index):
        path, img_class = self.img_dataset[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        if self.phase == 'train':
            return img, int(img_class)
        else:
            path = path.replace('/', '\\')
            bbox = self.test_bbox[path]
            return img, int(img_class), bbox

    def __len__(self):
        return len(self.img_dataset)


def get_imagenet_dataset():
    datadir = '/path_to_data/imagenet'
    ds_train = ImageDataset(datadir, phase='train')
    ds_test = ImageDataset(datadir, phase='test')
    return ds_train, ds_test


def prepare(file, rootdir):
    main = []
    f = open(file, 'r')
    buffer = f.readlines()
    for file in buffer:
        filename, label = file.split('\n')[0].split('/')[1].split(';')
        filename = os.path.join(rootdir, filename)
        main.append((filename, label))
    return main


def get_cat():
    f = open('datasets/imagenet_labels.txt', 'r')
    buffer = f.readlines()
    main = []
    for line in buffer:
        main.append(line.split('\t')[1].split(',')[0].strip('\n'))
    return main


if __name__ == "__main__":
    # create_image_sizes_file(dataset_path='/path_to_data/CUB/CUB_200_2011')
    from tqdm import tqdm
    import cv2

    train, test = get_imagenet_dataset()
    ds_train = torch.utils.data.DataLoader(train,
                                           batch_size=16,
                                           num_workers=8,
                                           shuffle=False,
                                           drop_last=False)
    ds_test = torch.utils.data.DataLoader(test,
                                          batch_size=1,
                                          num_workers=0,
                                          shuffle=False,
                                          drop_last=False)
    # pbar = tqdm(ds_train)
    # for i, (real_imgs, labels) in enumerate(pbar):
    #     pass
        # image = real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        # image = (image - image.min()) / (image.max() - image.min())
        # image = np.array(255 * image).copy().astype(np.uint8)

    pbar = tqdm(ds_test)
    for i, (real_imgs, labels, bbox_list) in enumerate(pbar):
        image = real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        image = np.array(255*image).copy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for bbox in bbox_list:
            gxa = int(bbox[0])
            gya = int(bbox[1])
            gxb = int(bbox[2])
            gyb = int(bbox[3])
            image = cv2.rectangle(image, (gxa, gya), (gxb, gyb), (0, 0, 255), 2)
        cv2.imwrite(test.categories[int(labels)] + '.jpg', image)
