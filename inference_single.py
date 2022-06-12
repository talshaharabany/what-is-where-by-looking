import torch.utils.data
import cv2
from tqdm import tqdm
import os
import numpy as np

from model import *
from datasets.cub import get_dataset
from datasets.cars import get_cars_dataset
from datasets.tiny_imagenet import get_tiny_dataset
from datasets.flowers import get_flowers_dataset
from datasets.dogs import get_dogs_dataset

from utils import generate_bbox, calculate_IOU, intensity_to_rgb, MaskEvaluator, get_clip_maps
import matplotlib.cm as cm
import CLIP.clip as clip


def norm_img(img):
    img = img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img = img * 0.5 + 0.5
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def norm_mask(mask):
    mask = mask.squeeze().detach().cpu().numpy()
    return mask


def inference_bbox(ds, model, args):
    pbar = tqdm(ds)
    hit_local = 0
    cnt = 0
    vis = True
    for i, (real_imgs, labels, bbox) in enumerate(pbar):
        if vis and i > 100:
            break
        real_imgs = real_imgs.cuda()
        mask_tensor = model(real_imgs)
        mask = mask_tensor.squeeze().detach().cpu().numpy().copy()
        image = real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy()

        if args['task'] == 'imagenet' or args['task'] == 'dog':
            max_iou = -1
            estimated_bbox, blend_bbox, mask = generate_bbox(image, mask, gt_bbox=None, thr_val=float(args['th']))
            for b in bbox:
                gt_bbox = []
                gt_bbox.append(b[0].item())
                gt_bbox.append(b[1].item())
                gt_bbox.append(b[2].item())
                gt_bbox.append(b[3].item())
                IOU = calculate_IOU(estimated_bbox, gt_bbox)
                if IOU > max_iou:
                    max_iou = IOU
            IOU = max_iou
        else:
            gt_bbox = []
            gt_bbox.append(bbox[0].item())
            gt_bbox.append(bbox[1].item())
            gt_bbox.append(bbox[2].item())
            gt_bbox.append(bbox[3].item())
            estimated_bbox, blend_bbox, mask = generate_bbox(image, mask, gt_bbox, thr_val=float(args['th']))
            IOU = calculate_IOU(estimated_bbox, gt_bbox)
        if IOU >= 0.5:
            hit_local += 1
        cnt += 1
        pbar.set_description(
            '(Inference | {task}) Resnet{order} :: localization Acc {acc_local:.4f}'.format(
                task=args['task'],
                order=args['order_ae'],
                acc_local=100*float(hit_local/cnt),
            ))
        if vis:
            x1, y1, x2, y2 = estimated_bbox
            tmp = np.zeros(mask.shape)
            tmp[y1:y2, x1:x2] = 1
            mask = mask * tmp
            heatmap = intensity_to_rgb(mask, normalize=True).astype('uint8')
            gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
            thr_val = 0.1 * np.max(gray_heatmap)
            _, mask = cv2.threshold(gray_heatmap, int(thr_val), 255, cv2.THRESH_TOZERO)

            jet = cm.get_cmap("jet")
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[mask]
            superimposed_img = jet_heatmap * 0.4 + cv2.cvtColor(image * 0.5 + 0.5
                                                                , cv2.COLOR_BGR2RGB)
            cv2.imwrite('out/localization/' + args['task'] + '/' + str(i) + '_image.png',
                        cv2.cvtColor(128 * image + 128, cv2.COLOR_BGR2RGB))
            cv2.imwrite('out/localization/' + args['task'] + '/' + str(i) + '_blend.png',
                        cv2.cvtColor(blend_bbox, cv2.COLOR_BGR2RGB))
            cv2.imwrite('out/localization/' + args['task'] + '/' + str(i) + '_map.png',
                        255*superimposed_img)
            # cv2.imwrite(
            #     'out/localization/' + args['task'] + '/' + str(i) + '_background.png',
            #     cv2.cvtColor(1 - mask, cv2.COLOR_GRAY2RGB) * cv2.cvtColor(128 * image + 128, cv2.COLOR_BGR2RGB))


def inference_mask(ds, model, args):
    pbar = tqdm(ds)
    mask_eval = MaskEvaluator()
    for i, (real_imgs, labels, mask) in enumerate(pbar):
        real_imgs = real_imgs.cuda()
        gt_mask = mask.squeeze().detach().cpu().numpy()
        _, mask_tensor = model(real_imgs)
        scoremap = mask_tensor.squeeze().detach().cpu().numpy().copy()
        mask_eval.accumulate(scoremap, gt_mask)
        pbar.set_description(
            '(Inference | {task}) :: segmentation Acc {acc_local:.4f} '.format(
                task=args['task'],
                acc_local=mask_eval.compute(),
            ))
    return mask_eval.compute()


def inference_clip(ds, clip_model, text_class, args):
    pbar = tqdm(ds)
    hit_local = 0
    cnt = 0
    vis = False
    for i, (real_imgs, labels, bbox) in enumerate(pbar):
        if vis and i > 100:
            break
        real_imgs = real_imgs.cuda()
        mask_tensor = get_clip_maps(real_imgs, text_class, clip_model, 'cuda').detach().clone()
        mask = mask_tensor.squeeze().detach().cpu().numpy().copy()
        image = real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy()

        if args['task'] == 'imagenet' or args['task'] == 'dogs':
            max_iou = -1
            estimated_bbox, _, _ = generate_bbox(image, mask, gt_bbox=None, thr_val=float(args['th']))
            for b in bbox:
                gt_bbox = []
                gt_bbox.append(b[0].item())
                gt_bbox.append(b[1].item())
                gt_bbox.append(b[2].item())
                gt_bbox.append(b[3].item())
                IOU = calculate_IOU(estimated_bbox, gt_bbox)
                if IOU > max_iou:
                    max_iou = IOU
            IOU = max_iou
        else:
            gt_bbox = []
            gt_bbox.append(bbox[0].item())
            gt_bbox.append(bbox[1].item())
            gt_bbox.append(bbox[2].item())
            gt_bbox.append(bbox[3].item())
            estimated_bbox, blend_bbox, num_of_objects = generate_bbox(image, mask, gt_bbox, thr_val=float(args['th']))
            IOU = calculate_IOU(estimated_bbox, gt_bbox)
        if IOU >= 0.5:
            hit_local += 1
        cnt += 1
        pbar.set_description(
            '(Inference | {task}) Resnet{order} :: localization Acc {acc_local:.4f}'.format(
                task=args['task'],
                order=args['order_ae'],
                acc_local=100*float(hit_local/cnt),
            ))
        if vis:
            x1, y1, x2, y2 = estimated_bbox
            tmp = np.zeros(mask.shape)
            tmp[y1:y2, x1:x2] = 1
            mask = mask * tmp
            heatmap = intensity_to_rgb(mask, normalize=True).astype('uint8')
            gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
            thr_val = 0.1 * np.max(gray_heatmap)
            _, mask = cv2.threshold(gray_heatmap, int(thr_val), 255, cv2.THRESH_TOZERO)

            jet = cm.get_cmap("jet")
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[mask]
            superimposed_img = jet_heatmap * 0.4 + cv2.cvtColor(image * 0.5 + 0.5
                                                                , cv2.COLOR_BGR2RGB)
            cv2.imwrite('out/localization/' + args['task'] + '/' + str(i) + '_image_clip.png',
                        cv2.cvtColor(128 * image + 128, cv2.COLOR_BGR2RGB))
            cv2.imwrite('out/localization/' + args['task'] + '/' + str(i) + '_blend_clip.png',
                        cv2.cvtColor(blend_bbox, cv2.COLOR_BGR2RGB))
            cv2.imwrite('out/localization/' + args['task'] + '/' + str(i) + '_map_clip.png',
                        255*superimposed_img)
            # cv2.imwrite(
            #     'out/localization/' + args['task'] + '/' + str(i) + '_background_clip.png',
            #     cv2.cvtColor(1 - mask, cv2.COLOR_GRAY2RGB) * cv2.cvtColor(128 * image + 128, cv2.COLOR_BGR2RGB))

def vis(ds, model, clip_model, text_class, args):
    pbar = tqdm(ds)
    for i, (real_imgs, labels, bbox) in enumerate(pbar):
        if i > 100:
            break
        if args['task'] == "dogs":
            bbox = bbox[0]
        if bbox is not None:
            _gt_bbox = list()
            _gt_bbox.append(max(int(bbox[0]), 0))
            _gt_bbox.append(max(int(bbox[1]), 0))
            _gt_bbox.append(min(int(bbox[2]), 224 - 1))
            _gt_bbox.append(min(int(bbox[3]), 224))
        real_imgs = real_imgs.cuda()
        image = real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        mask_tensor = get_clip_maps(real_imgs, text_class, clip_model, 'cuda').detach().clone()
        mask_clip = mask_tensor.squeeze().detach().cpu().numpy().copy()
        estimated_clip_bbox, _, _ = generate_bbox(image, mask_clip, gt_bbox=None, thr_val=float(args['th']))
        mask_tensor = model(real_imgs)
        mask = mask_tensor.squeeze().detach().cpu().numpy().copy()
        estimated_bbox, _, _ = generate_bbox(image, mask, gt_bbox=None, thr_val=float(args['th']))

        image = (128 * real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy() + 128).astype(np.uint8).copy()
        x1, y1, x2, y2 = _gt_bbox
        cv2.rectangle(image,
                      (x1, y1),
                      (x2, y2),
                      (0, 255, 0), 2)
        x1, y1, x2, y2 = estimated_clip_bbox
        cv2.rectangle(image,
                      (x1, y1),
                      (x2, y2),
                      (255, 0, 0), 2)
        x1, y1, x2, y2 = estimated_bbox
        cv2.rectangle(image,
                      (x1, y1),
                      (x2, y2),
                      (0, 0, 255), 2)
        cv2.imwrite('out/localization/' + args['task'] + '/' + str(i) + '_vis.png',
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def main(args=None):
    gpu_num = torch.cuda.device_count()

    model = torch.nn.DataParallel(Model(args=args), list(range(gpu_num))).cuda().eval()
    model1 = torch.load(args['path_best'])
    model.load_state_dict(model1.state_dict())

    if args['task'] == 'cub':
        trainset, testset = get_dataset(is_bbox=True, size=int(args['Isize']), datadir=args['data_path'])
    elif args['task'] == 'car':
        trainset, testset = get_cars_dataset(is_bbox=True, size=int(args['Isize']), datadir=args['data_path'])
    elif args['task'] == 'tiny':
        trainset, testset = get_tiny_dataset(is_bbox=True, size=int(args['Isize']), datadir=args['data_path'])
    elif args['task'] == 'flowers':
        trainset, testset = get_flowers_dataset(datadir=args['data_path'])
    elif args['task'] == 'dogs':
        trainset, testset = get_dogs_dataset()

    ds = torch.utils.data.DataLoader(testset,
                                     batch_size=1,
                                     num_workers=1,
                                     shuffle=False,
                                     drop_last=False)
    # with torch.no_grad():
    #     inference_bbox(ds, model, args)
        # if args['task'] == 'flowers':
        #     inference_mask(ds, model, args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    class_text = clip.tokenize(trainset.text_class).to(device)
    inference_clip(ds, clip_model, class_text, args)

    # vis(ds, model, clip_model, class_text, args)


if __name__ == '__main__':
    '''
        datadir = '/path_to_data/CUB/CUB_200_2011'
        datadir = '/path_to_data/cars'
        datadir = '/path_to_data/tiny_imagenet/tiny-imagenet-200'
        datadir = '/path_to_data/flowers'
    '''
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-nW', '--nW', default=1, help='number of workers', required=False)
    parser.add_argument('-backbone', '--backbone', default='resnet', help='number of workers', required=False)
    parser.add_argument('-order_ae', '--order_ae', default=50, help='order of the backbone - ae', required=False)
    parser.add_argument('-task', '--task', default='cub', help='dataset task', required=False)
    parser.add_argument('-th', '--th', default=0.1, help='evaluation th', required=False)
    parser.add_argument('-nC', '--nC', default=200, help='number of classes', required=False)
    parser.add_argument('-Isize', '--Isize', default=224, help='image size', required=False)
    parser.add_argument('-pretrained', '--pretrained', default=False, help='pretrined models', required=False)
    parser.add_argument('-path_ae', '--path_ae', default=4, help='ae folder path', required=False)
    parser.add_argument('-data_path', '--data_path',
                        default='/path_to_data/CUB/CUB_200_2011', help='data set path', required=False)
    args = vars(parser.parse_args())

    args['path_best'] = os.path.join('results', 'gpu' + str(args['path_ae']),
                                     'net_best.pth')
    args['path_best'] = r'/path_to_data/weakly_two_images/results/car/gpu3/net_best.pth'
    main(args=args)
