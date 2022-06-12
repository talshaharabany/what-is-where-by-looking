#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/10 10:28
# @File    : load_data.py
# @Author  : NUS_LuoKe

import os

from skimage import io
from skimage.transform import resize
from tqdm import tqdm
import numpy as np
import cv2


def seg2mask(segmentation_dir, mask_save_dir):
    '''
    segmentation given by the data set is flower instead of mask of the flower.
    generate mask from their corresponding segmentation.
    '''
    if not os.path.isdir(mask_save_dir):
        os.mkdir(mask_save_dir)
    a = os.listdir(segmentation_dir)
    for seg in tqdm(a):
        seg_path = os.path.join(segmentation_dir, seg)
        count = os.path.basename(seg_path).split("_")[1]
        seg_array = io.imread(seg_path, as_gray=True)

        # the pixel value of the background in the gray image is 0.07181725490196078
        # Can use skimage.color.rgb2gray()
        seg_array[seg_array > 0.15] = 1
        seg_array[seg_array != 1] = 0
        seg_array = (seg_array*255).astype(np.uint8)
        cv2.imwrite(os.path.join(mask_save_dir, "mask_{}".format(count)), seg_array)
        # io.imsave(fname=os.path.join(mask_save_dir, "mask_{}".format(count)), arr=seg_array)


def resize_image(image_dir, resized_image_save_dir, prefix, output_shape):
    if not os.path.isdir(resized_image_save_dir):
        os.mkdir(resized_image_save_dir)
    for image in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image)
        count = os.path.basename(image_path).split("_")[1]

        img_arr = io.imread(image_path)
        re_img_arr = resize(img_arr, output_shape)
        io.imsave(fname=os.path.join(resized_image_save_dir, prefix + "_{}".format(count)), arr=re_img_arr)
