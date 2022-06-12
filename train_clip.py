import torch.optim as optim
import torch.utils.data
import torch.nn as nn
# from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import numpy as np

from model import *

from datasets.cub import get_dataset
from datasets.cars import get_cars_dataset
from datasets.flowers import get_flowers_dataset
from datasets.imagenet import get_imagenet_dataset
from datasets.dogs import get_dogs_dataset

from utils import generate_bbox, calculate_IOU, MaskEvaluator, interpret_batch
import CLIP.clip as clip


def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))


def gen_step(optimizer_G, clip_model, real_imgs, labels, model, criterion, text_class, args):
    bs = real_imgs.shape[0]
    optimizer_G.zero_grad()
    clip_model.to('cuda:' + str(real_imgs.get_device()))
    model.to('cuda:' + str(real_imgs.get_device()))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cam = interpret_batch(real_imgs, text_class, clip_model, device).detach().clone().float()

    M = model(real_imgs)
    clip_cam_loss = F.mse_loss(M, cam)

    logits_fr, _ = clip_model(M * real_imgs, text_class)
    logits_bg, _ = clip_model((1 - M) * real_imgs, text_class)

    regularization = M.mean()

    if args['task'] == 'imagenet':
        gt = torch.cumsum(torch.ones(bs), dim=0).long().to('cuda:' + str(real_imgs.get_device())) - 1
        bg_loss = (logits_bg.diag() / (logits_fr.detach().diag() + 1e-8)).mean()
    else:
        logits, _ = clip_model(real_imgs, text_class)
        gt = torch.zeros(bs).long().to('cuda:' + str(real_imgs.get_device()))
        bg_loss = logits_bg[:, 0] / (logits[:, 0] + 1e-8)

    weakly_loss = criterion(logits_fr, gt)

    loss = float(args['w3']) * weakly_loss + \
           float(args['w0']) * regularization +\
           float(args['w1']) * clip_cam_loss +\
           float(args['w2']) * bg_loss.mean()
    loss.backward()
    optimizer_G.step()
    return loss.item(), regularization.item()


def logger(writer, loss_list, tplt_loss, step):
    writer.add_scalar('Loss', loss_list, global_step=step)
    writer.add_scalar('tplt_loss', tplt_loss, global_step=step)


def train(ds, model, clip_model, optimizer_G, writer, steps, text_class, args):
    loss_list = []
    pbar = tqdm(ds)
    criterion = nn.CrossEntropyLoss()
    for i, inputs in enumerate(pbar):
        real_imgs = inputs[0].cuda()
        labels = inputs[1].cuda()
        loss, tplt_loss = gen_step(optimizer_G, clip_model, real_imgs, labels, model, criterion, text_class, args)
        loss_list.append(loss)
        # logger(writer, loss, tplt_loss, steps)
        steps = steps + 1
        pbar.set_description(
            '(train | {}) steps {steps} ::'
            ' loss {loss:.4f}'.format(
                'Weakly mask',
                steps=steps,
                loss=np.mean(loss_list)
            ))
    return steps, np.mean(loss_list)


def inference_bbox(ds, model, epoch, args):
    pbar = tqdm(ds)
    model.eval()
    hit = 0
    cnt = 0
    for i, inputs in enumerate(pbar):
        real_imgs = inputs[0].cuda()
        bbox = inputs[2]
        mask = model(real_imgs)
        mask = mask.squeeze().detach().cpu().numpy().copy()
        image = real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        if args['task'] == 'imagenet' or args['task'] == 'dog':
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
            estimated_bbox, _, _ = generate_bbox(image, mask, gt_bbox, thr_val=float(args['th']))
            IOU = calculate_IOU(estimated_bbox, gt_bbox)
        if IOU >= 0.5:
            hit += 1
        cnt += 1
        pbar.set_description(
            '(Inference | {task}) Epoch {epoch} :: Accuracy {loss:.4f}'.format(
                task=args['task'],
                epoch=epoch,
                loss=100 * float(hit / cnt)))
    model.train()
    return 100 * float(hit / cnt)


def inference_mask(ds, model, epoch, args):
    pbar = tqdm(ds)
    model.eval()
    mask_eval = MaskEvaluator()
    for i, inputs in enumerate(pbar):
        real_imgs = inputs[0]
        labels = inputs[1]
        gt_mask = inputs[2].squeeze().detach().cpu().numpy()
        real_imgs = real_imgs.cuda()
        mask = model(real_imgs)
        scoremap = mask.squeeze().detach().cpu().numpy().copy()
        mask_eval.accumulate(scoremap, gt_mask)
        pbar.set_description(
            '(Inference | {task}) Epoch {epoch} :: Accuracy {loss:.4f}'.format(
                task=args['task'],
                epoch=epoch,
                loss=mask_eval.compute()))
    model.train()
    return mask_eval.compute()


def main(args=None, writer=None):
    gpu_num = torch.cuda.device_count()
    if not args['backbone'] == 'mobile':
        model = Model(args=args)
        model = torch.nn.DataParallel(model, list(range(gpu_num))).cuda()
        model1 = torch.load(args['path_init'])
        model.load_state_dict(model1)
    else:
        model = Model(args=args)
        model = torch.nn.DataParallel(model, list(range(gpu_num))).cuda()
        model1 = torch.load(args['path_init'])
        model.module.E.load_state_dict(model1)


    optimizer_G = optim.SGD(model.parameters(),
                            lr=float(args['learning_rate']),
                            weight_decay=float(args['WD']),
                            momentum=float(args['M']))

    if args['task'] == 'imagenet':
        trainset, testset = get_imagenet_dataset()
    elif args['task'] == 'cub':
        trainset, testset = get_dataset(size=int(args['Isize']), is_bbox=True, datadir=args['data_path'])
    elif args['task'] == 'car':
        trainset, testset = get_cars_dataset(size=int(args['Isize']), is_bbox=True, datadir=args['data_path'])
    elif args['task'] == 'flowers':
        trainset, testset = get_flowers_dataset(datadir=args['data_path'])
    elif args['task'] == 'dog':
        trainset, testset = get_dogs_dataset()


    ds = torch.utils.data.DataLoader(trainset,
                                     batch_size=int(args['Batch_size']),
                                     num_workers=int(args['nW']),
                                     shuffle=True,
                                     drop_last=True)
    ds_test = torch.utils.data.DataLoader(testset,
                                          batch_size=1,
                                          num_workers=1,
                                          shuffle=False,
                                          drop_last=False)
    model.train()
    steps = 0
    results_path = os.path.join('results',
                                'gpu' + args['folder'],
                                'results.csv')
    best_path = os.path.join('results',
                             'gpu' + args['folder'],
                             'best.csv')
    f_all = open(results_path, 'w')
    f_best = open(best_path, 'w')
    f_all.write('epoches,label,acc\n')
    f_best.write('epoches,acc\n')
    best = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    class_text = clip.tokenize(trainset.text_class).to(device)

    for epoch in range(int(args['epoches'])):
        steps, _ = train(ds, model.train(), clip_model.eval(), optimizer_G, writer, steps, class_text, args)
        if epoch >= 0:
            with torch.no_grad():
                if args['task'] == 'flowers':
                    acc = inference_mask(ds_test, model, epoch, args)
                else:
                    acc = inference_bbox(ds_test, model, epoch, args)
            f_all.write(str(epoch) + ',' + str('test') + ',' + str(acc) + '\n')
            f_all.flush()
            if acc > best:
                torch.save(model, args['path_best'])
                best = acc
                f_best.write(str(epoch) + ',' + str(acc) + '\n')
                f_best.flush()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr', '--learning_rate', default=0.0012, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=16, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', default=5000, help='number of epoches', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='number of workers', required=False)
    parser.add_argument('-WD', '--WD', default=1e-4, help='weight decay', required=False)
    parser.add_argument('-order_ae', '--order_ae', default=16, help='order of the backbone - ae', required=False)
    parser.add_argument('-backbone', '--backbone', default='vgg', help='order of the backbone - ae', required=False)
    parser.add_argument('-task', '--task', default='cub', help='dataset task', required=False)
    parser.add_argument('-Isize', '--Isize', default=224, help='image size', required=False)
    parser.add_argument('-nC', '--nC', default=200, help='number of classes', required=False)
    parser.add_argument('-th', '--th', default=0.1, help='evaluation th', required=False)
    parser.add_argument('-temp', '--temp', default=1, help='pretrined models', required=False)
    parser.add_argument('-w0', '--w0', default=1, help='pretrined models', required=False)
    parser.add_argument('-w1', '--w1', default=4, help='pretrined models', required=False)
    parser.add_argument('-w2', '--w2', default=2, help='pretrined models', required=False)
    parser.add_argument('-w3', '--w3', default=1, help='pretrined models', required=False)
    parser.add_argument('-M', '--M', default=0.96, help='pretrined models', required=False)
    parser.add_argument('-step_size', '--step_size', default=20, help='pretrined models', required=False)
    parser.add_argument('-gamma', '--gamma', default=1, help='pretrined models', required=False)
    parser.add_argument('-pretrained', '--pretrained', default=False, help='pretrined models', required=False)
    parser.add_argument('-data_path', '--data_path',
                        default='/path_to_data/CUB/CUB_200_2011', help='data set path', required=False)
    args = vars(parser.parse_args())

    folder = open_folder('results')
    Isize = str(args['Isize'])
    args['folder'] = folder
    args['path_best'] = os.path.join('results', 'gpu' + folder, 'net_best.pth')
    args['path_save_init'] = os.path.join('results', 'gpu' + folder, 'net_init.pth')
    args['path_init'] = os.path.join('results', 'init', 'cub',
                                     str(args['backbone']) + str(args['order_ae']), 'net_init.pth')
    main(args=args, writer=None)

