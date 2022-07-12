import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np

from model import *

from datasets.flicker import get_flicker1K_dataset, get_flicker_dataset
from datasets.visual_genome import get_VG_dataset
from datasets.coco import get_coco_dataset

from utils import interpret_batch, interpret_new
import CLIP.clip as clip
from inference_grounding import inference_bbox


def norm_z(z):
    return z / z.norm(dim=1).unsqueeze(dim=1)


def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))


def get_logits(clip_model, real_imgs, text_pos, text_neg):
    logits_pos, _ = clip_model(real_imgs, text_pos)
    logits_neg, _ = clip_model(real_imgs, text_neg)
    logits_fr = torch.cat((logits_pos.diag().unsqueeze(-1),
                           logits_neg.diag().unsqueeze(-1)),
                          dim=1)
    return logits_fr


def gen_step(optimizer_G, clip_model, real_imgs, text, model, criterion, args):
    bs = real_imgs.shape[0]
    gt = torch.zeros(bs).long().to('cuda:' + str(real_imgs.get_device()))
    optimizer_G.zero_grad()
    clip_model.to('cuda:' + str(real_imgs.get_device()))
    model.to('cuda:' + str(real_imgs.get_device()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_pos = text[:, :, 0]
    z_t = norm_z(clip_model.encode_text(text_pos))
    real_imgs_224 = F.interpolate(real_imgs, size=(224, 224), mode="bilinear", align_corners=True)
    cam = interpret_new(real_imgs_224.detach(), text_pos.detach(), clip_model, device).detach().clone().float()
    cam = F.interpolate(cam, size=(int(args['Isize']), int(args['Isize'])), mode="bilinear", align_corners=True)
    M = model(real_imgs, z_t)
    clip_cam_loss = F.mse_loss(M, cam)
    M = F.interpolate(M, size=(224, 224), mode="bilinear", align_corners=True)
    z_fr = norm_z(clip_model.encode_image(real_imgs_224 * M))
    z_bg = norm_z(clip_model.encode_image(real_imgs_224 * (1 - M)))
    regularization = M.mean()
    fr_loss = (1 - (z_fr @ z_t.T)).mean()
    bg_loss = torch.abs(z_bg @ z_t.T).mean()
    loss = float(args['w3']) * fr_loss + \
           float(args['w0']) * regularization +\
           float(args['w1']) * clip_cam_loss +\
           float(args['w2']) * bg_loss.mean()
    loss.backward()
    optimizer_G.step()
    return loss.item(), 0


def logger(writer, loss_list, tplt_loss, step):
    writer.add_scalar('Loss', loss_list, global_step=step)
    writer.add_scalar('tplt_loss', tplt_loss, global_step=step)


def train(ds, model, clip_model, optimizer_G, args):
    loss_list = []
    pbar = tqdm(ds)
    criterion = nn.CrossEntropyLoss()
    for i, inputs in enumerate(pbar):
        real_imgs = inputs[0].cuda()
        text_pos = inputs[1]
        text_pos_token = clip.tokenize(text_pos).to('cuda').unsqueeze(dim=2)
        loss, tplt_loss = gen_step(optimizer_G, clip_model, real_imgs, text_pos_token, model, criterion, args)
        loss_list.append(loss)
        pbar.set_description('(train) :: loss {loss:.4f}'.format(loss=np.mean(loss_list)))
    return np.mean(loss_list)


def main(args=None):
    args['is_blip'] = False
    gpu_num = torch.cuda.device_count()
    model = MultiModel(args=args)
    model = torch.nn.DataParallel(model, list(range(gpu_num))).cuda()
    if bool(int(args['resume'])):
        model1 = torch.load(args['resume_folder'])
        model.load_state_dict(model1.state_dict())
    optimizer_G = optim.SGD(model.parameters(),
                            lr=float(args['learning_rate']),
                            weight_decay=float(args['WD']),
                            momentum=float(args['M']))

    if args['task'] == 'flicker':
        trainset, testset = get_flicker_dataset(args=args)
    elif args['task'] == 'vg_train':
        trainset = get_VG_dataset(args=args)
        testset = get_flicker1K_dataset(args=args)
    elif args['task'] == 'coco':
        trainset = get_coco_dataset(args=args)
        testset = get_flicker1K_dataset(args=args)

    ds = torch.utils.data.DataLoader(trainset,
                                     batch_size=int(args['Batch_size']),
                                     num_workers=int(args['nW']),
                                     shuffle=True,
                                     drop_last=True)
    ds_test = torch.utils.data.DataLoader(testset,
                                          batch_size=1,
                                          num_workers=int(args['nW_eval']),
                                          shuffle=False,
                                          drop_last=False)
    model.train()
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
    for epoch in range(int(args['epoches'])):
        train(ds, model.train(), clip_model.eval(), optimizer_G, args)
        acc = inference_bbox(ds_test, model.eval(), clip_model.eval(), epoch, args)
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
    parser.add_argument('-bs', '--Batch_size', default=32, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', default=5000, help='number of epoches', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='number of workers', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='number of workers', required=False)
    parser.add_argument('-WD', '--WD', default=1e-4, help='weight decay', required=False)
    parser.add_argument('-order_ae', '--order_ae', default=16, help='order of the backbone - ae', required=False)
    parser.add_argument('-backbone', '--backbone', default='vgg', help='order of the backbone - ae', required=False)
    parser.add_argument('-task', '--task', default='vg_train', help='dataset task', required=False)
    parser.add_argument('-Isize', '--Isize', default=304, help='image size', required=False)
    parser.add_argument('-nC', '--nC', default=200, help='number of classes', required=False)
    parser.add_argument('-th', '--th', default=0.1, help='evaluation th', required=False)
    parser.add_argument('-temp', '--temp', default=1, help='pretrined models', required=False)
    parser.add_argument('-w0', '--w0', default=1, help='pretrined models', required=False)
    parser.add_argument('-w1', '--w1', default=4, help='pretrined models', required=False)
    parser.add_argument('-w2', '--w2', default=1, help='pretrined models', required=False)
    parser.add_argument('-w3', '--w3', default=1, help='pretrined models', required=False)
    parser.add_argument('-M', '--M', default=0.9, help='pretrined models', required=False)
    parser.add_argument('-prob', '--prob', default=10, help='pretrined models', required=False)
    parser.add_argument('-step_size', '--step_size', default=20, help='pretrined models', required=False)
    parser.add_argument('-gamma', '--gamma', default=1, help='pretrined models', required=False)
    parser.add_argument('-resume', '--resume', default=False, help='pretrined models', required=False)
    parser.add_argument('-resume_folder', '--resume_folder', default='41', help='pretrined models', required=False)
    parser.add_argument('-pretrained', '--pretrained', default=False, help='pretrined models', required=False)
    parser.add_argument('-img_path', '--img_path', default=True, help='pretrined models', required=False)
    parser.add_argument('-data_path', '--data_path',
                        default='/media/media1/talshah/coco/VG', help='data set path', required=False)
    parser.add_argument('-val_path', '--val_path',
                        default=r'/media/media1/talshah/coco/flicker', help='data set path', required=False)
    args = vars(parser.parse_args())
    folder = open_folder('results')
    Isize = str(args['Isize'])
    args['folder'] = folder
    args['path_best'] = os.path.join('results', 'gpu' + folder, 'net_best.pth')
    args['resume_folder'] = os.path.join('results', 'gpu' + args['resume_folder'], 'net_best.pth')
    args['path_save_init'] = os.path.join('results', 'gpu' + folder, 'net_init.pth')
    args['path_init'] = os.path.join('results', 'init', 'cub',
                                     str(args['backbone']) + str(args['order_ae']), 'net_init.pth')
    main(args=args)

