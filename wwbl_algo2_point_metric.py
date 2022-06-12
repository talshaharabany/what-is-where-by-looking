import torch.utils.data
from tqdm import tqdm
import os

from model import *
from datasets.flicker import get_flicker1K_dataset
from datasets.referit_loader import get_refit_test_dataset
from datasets.visual_genome import get_VGtest_dataset

from utils_grounding import *
from utils import interpret_batch, calculate_IOU
import CLIP.clip as clip
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from BLIP.models.blip import blip_decoder
from torchvision.transforms.functional import InterpolationMode


def norm_z(z):
    return z / z.norm(dim=1).unsqueeze(dim=1)


def norm_img(img):
    img = img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img = img * 0.5 + 0.5
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def norm_mask(mask):
    mask = mask.squeeze().detach().cpu().numpy()
    return mask


def load_blip_img(raw_image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 384

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def save_image_bbox(img_path, bboxes, org_caption):
    pil_img = Image.open(img_path)
    image = np.array(pil_img.resize((224, 224)))
    image = cv2.copyMakeBorder(image, 0, 20, 0, 256, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gap = 15
    font = cv2.FONT_HERSHEY_SIMPLEX
    for ix, bbox in enumerate(bboxes):
        gxa = int(bbox[0])
        gya = int(bbox[1])
        gxb = int(bbox[2])
        gyb = int(bbox[3])
        conf = float(bbox[4])
        text = bbox[5][0]
        cv2.putText(image, str(ix) + ': ' + text + ': ' + str(conf)[1:6], (224, 10 + gap * ix), font, fontScale=0.3, color=(0, 0, 0),
                    thickness=1)
        cv2.putText(image, str(ix), (gxa, gya + 10), font, fontScale=0.32, color=(0, 0, 0), thickness=1)
        cv2.putText(image, org_caption, (10, 240), font, fontScale=0.3, color=(0, 0, 0), thickness=1)
        image = cv2.rectangle(image, (gxa, gya), (gxb, gyb), (np.random.randint(0, 256),
                                                              np.random.randint(0, 256),
                                                              np.random.randint(0, 256)), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (448, 448))
    return image

def draw_bboxes(img_path, bboxes):
    COLORS = [(255,0,0), (0,0,255), (255,0,255), (0,128,0), (128,0,128), (255,165,0), (128,0,0), (245,245,220)]

    pil_img = Image.open(img_path)
    image = np.array(pil_img.resize((224, 224)))
    # image = cv2.copyMakeBorder(image, 0, 20, 0, 256, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    for ix, bbox in enumerate(bboxes):
        gxa = int(bbox[0])
        gya = int(bbox[1])
        gxb = int(bbox[2])
        gyb = int(bbox[3])

        image = cv2.rectangle(image, (gxa, gya), (gxb, gyb), COLORS[ix], 1)

    return image

def bbox_overlap(pred_bbox, gt_bbox):
    xA = max(pred_bbox[0], gt_bbox[0])
    yA = max(pred_bbox[1], gt_bbox[1])
    xB = min(pred_bbox[2], gt_bbox[2])
    yB = min(pred_bbox[3], gt_bbox[3])

    # compute the area of intersection rectangle
    interArea = max(xB - xA + 1, 0) * max(yB - yA + 1, 0)

    # compute the area of both the prediction and ground-truth
    # rectangles
    pred_bbox_area = (pred_bbox[2] - pred_bbox[0] + 1) * (pred_bbox[3] - pred_bbox[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    overlap = interArea / float(pred_bbox_area)

    # return the intersection over union value
    return overlap


def calc_pointing_metric(dataloader, model, clip_model, text_generator, stop_threshold, verbal=False,
                         start=0, end=100000, clip_eval=False):
    cnt_overall = 0
    cnt_correct_hit = 0

    pbar = tqdm(range(start, min(end, len(dataloader))))
    for idx in pbar:
        (flicker_img, meta, size, img_path) = dataloader.dataset[idx]
        flicker_img = flicker_img.cuda()
        # img_path = os.path.join(dataloader.dataset.img_folder, str(dataloader.dataset.files[idx]).strip('\n') + '.jpg')
        pil_img = Image.open(img_path)
        np_img = np.array(pil_img.resize((224, 224)))

        if len(np_img.shape) == 2:
            np_img = np.expand_dims(np_img, -1).repeat(3, -1)

        bboxes = []

        blip_img = load_blip_img(Image.fromarray(np_img))
        org_caption = text_generator.generate(blip_img, sample=False, num_beams=3, max_length=30, min_length=5)
        org_text = clip.tokenize(org_caption).to('cuda')
        z_org_text = norm_z(clip_model.encode_text(org_text))
        for i in range(15):
            blip_img = load_blip_img(Image.fromarray(np_img))

            caption = text_generator.generate(blip_img, sample=False, num_beams=3, max_length=30, min_length=5)
            text = clip.tokenize(caption).to('cuda')
            z_text = norm_z(clip_model.encode_text(text))
            curr_image = flicker_img.repeat(text.shape[0], 1, 1, 1)

            if not clip_eval:
                heatmap = model(curr_image, z_text)
            else:
                heatmap = interpret_batch(curr_image, text, clip_model, 'cuda', index=None)
            mask = heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy()

            for bbox in bboxes:
                mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0

            curr_bbox = generate_bbox(mask, threshold=0.4)

            crop_img = Image.fromarray(np_img[curr_bbox[1]:curr_bbox[3], curr_bbox[0]:curr_bbox[2], :])
            blip_crop_img = load_blip_img(crop_img)

            crop_caption = text_generator.generate(blip_crop_img, sample=False, num_beams=3, max_length=30,
                                                   min_length=5)
            crop_caption_tokenize = clip.tokenize(crop_caption).to('cuda')
            z_crop_text = norm_z(clip_model.encode_text(crop_caption_tokenize))
            conf = z_crop_text @ z_text.T

            max_overlap = 0
            for bbox in bboxes:
                overlap = bbox_overlap(bbox, curr_bbox)
                max_overlap = max(max_overlap, overlap)

            if i < 2 or (conf.item() > stop_threshold and max_overlap < 0.5 and mask[curr_bbox[1]:curr_bbox[3], curr_bbox[0]:curr_bbox[2]].mean() > 0.1):
                curr_bbox[4] = conf.item()
                curr_bbox.append(crop_caption)
                curr_bbox.append(mask)
                curr_bbox.append(z_crop_text)
                bboxes.append(curr_bbox)
                np_img[curr_bbox[1]:curr_bbox[3], curr_bbox[0]:curr_bbox[2], :] = 0
                flicker_img[:, curr_bbox[1]:curr_bbox[3], curr_bbox[0]:curr_bbox[2]] = 0
            else:
                break

        bboxes_text_vectors = torch.cat([x[7] for x in bboxes])
        for sen in meta.keys():
            item = meta[sen]
            size = [int(size[0]), int(size[1])]
            title, bbox = item['sentences'], item['bbox']
            gt_title_clip = clip.tokenize(title).to('cuda')
            gt_title_vector = norm_z(clip_model.encode_text(gt_title_clip)).mean(dim=0)

            closest_bbox_idx = torch.argmax(bboxes_text_vectors @ gt_title_vector).item()
            closest_bbox = bboxes[closest_bbox_idx]
            heatmap = closest_bbox[6]

            hit_c = calc_correctness(bbox, heatmap.astype(np.float), size)
            cnt_correct_hit += hit_c
            cnt_overall += 1

        var = 100. * cnt_correct_hit / cnt_overall
        prnt = 'Pointing Accucary:{:.2f}'.format(var)
        pbar.set_description(prnt)

    print(f'Final acc: {var}')
    print('Correct hit:', cnt_correct_hit)
    print('Overall:', cnt_overall)


def inference_BLIP(dataloader, model, clip_model, text_generator, stop_threshold, verbal=False):

    idx_list = [96]
    for idx in idx_list:
        curr_data = dataloader.dataset[idx]
        (flicker_img, meta, size) = curr_data
        flicker_img = flicker_img.cuda()
        img_path = os.path.join(dataloader.dataset.img_folder, str(dataloader.dataset.files[idx]).strip('\n') + '.jpg')
        pil_img = Image.open(img_path)
        np_img = np.array(pil_img.resize((224, 224)))
        bboxes = []

        blip_img = load_blip_img(Image.fromarray(np_img))
        org_caption = text_generator.generate(blip_img, sample=False, num_beams=3, max_length=30, min_length=5)
        org_text = clip.tokenize(org_caption).to('cuda')
        z_org_text = norm_z(clip_model.encode_text(org_text))
        for i in range(15):
            blip_img = load_blip_img(Image.fromarray(np_img))

            if verbal:
                plt.imshow(Image.fromarray(np_img))
                plt.show()

            caption = text_generator.generate(blip_img, sample=False, num_beams=3, max_length=30, min_length=5)
            text = clip.tokenize(caption).to('cuda')
            z_text = norm_z(clip_model.encode_text(text))
            curr_image = flicker_img.repeat(text.shape[0], 1, 1, 1)

            heatmap = model(curr_image, z_text)
            mask = heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy()

            for bbox in bboxes:
                mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0

            if verbal:
                plt.imshow(mask)
                plt.show()

            curr_bbox = generate_bbox(mask, threshold=0.4)

            crop_img = Image.fromarray(np_img[curr_bbox[1]:curr_bbox[3], curr_bbox[0]:curr_bbox[2], :])
            blip_crop_img = load_blip_img(crop_img)

            if verbal:
                plt.imshow(crop_img)
                plt.show()

            crop_caption = text_generator.generate(blip_crop_img, sample=False, num_beams=3, max_length=30, min_length=5)
            crop_caption_tokenize = clip.tokenize(crop_caption).to('cuda')
            z_crop_text = norm_z(clip_model.encode_text(crop_caption_tokenize))
            conf = z_crop_text @ z_text.T

            print(i, mask[curr_bbox[1]:curr_bbox[3], curr_bbox[0]:curr_bbox[2]].mean(), crop_caption, conf)

            max_overlap = 0
            for bbox in bboxes:
                overlap = bbox_overlap(bbox, curr_bbox)
                max_overlap = max(max_overlap, overlap)

            if i < 2 or (conf.item() > stop_threshold and max_overlap < 0.5 and mask[curr_bbox[1]:curr_bbox[3], curr_bbox[0]:curr_bbox[2]].mean() > 0.1):
                curr_bbox[4] = conf.item()
                curr_bbox.append(crop_caption)
                bboxes.append(curr_bbox)
                np_img[curr_bbox[1]:curr_bbox[3], curr_bbox[0]:curr_bbox[2], :] = 0
                flicker_img[:, curr_bbox[1]:curr_bbox[3], curr_bbox[0]:curr_bbox[2]] = 0
            else:
                break

        image = draw_bboxes(img_path, bboxes)
        plt.imshow(image[:,:,::-1])
        plt.show()
        cv2.imwrite('Apps/' + str(idx) + '.jpg', image)

def main(args=None):
    gpu_num = torch.cuda.device_count()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.DataParallel(MultiModel(args=args), list(range(gpu_num))).cuda().eval()
    model1 = torch.load(args['path_best'])
    model.load_state_dict(model1.state_dict())

    if args['dataset'] == 'flicker':
        testset = get_flicker1K_dataset(args=args)
    elif args['dataset'] == 'vg':
        testset = get_VGtest_dataset(args=args)
    elif args['dataset'] == 'refit':
        testset = get_refit_test_dataset(args=args)

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    image_size = 384
    text_generator = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    text_generator.eval()
    text_generator = text_generator.to(device)

    ds = torch.utils.data.DataLoader(testset,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=False,
                                     drop_last=False)
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)

    if args['task'] == 'metric':
        calc_pointing_metric(ds, model, clip_model, text_generator,
                             args['stop_th'], start=args['start'], end=args['end'],
                             clip_eval=bool(int(args['clip_eval']))
                             )
    else:
        inference_BLIP(ds, model, clip_model, text_generator, args['stop_th'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-nW', '--nW', default=1, help='number of workers', required=False)
    parser.add_argument('-backbone', '--backbone', default='vgg', help='number of workers', required=False)
    parser.add_argument('-order_ae', '--order_ae', default=16, help='order of the backbone - ae', required=False)
    parser.add_argument('-task', '--task', default='metric', help='dataset task', required=False) # metric/visualize
    parser.add_argument('-stop_th', '--stop_th', default=0.6, help='stopping rule th', required=False)
    parser.add_argument('-nC', '--nC', default=200, help='number of classes', required=False)
    parser.add_argument('-Isize', '--Isize', default=224, help='image size', required=False)
    parser.add_argument('-pretrained', '--pretrained', default=False, help='pretrined models', required=False)
    parser.add_argument('-path_ae', '--path_ae', default=41, help='ae folder path', required=False)
    parser.add_argument('-clip_eval', '--clip_eval', default=0, help='ae folder path', required=False)
    parser.add_argument('-data_path', '--data_path',
                        default='/path_to_data/cars', help='data set path', required=False)
    parser.add_argument('--dataset',
                        default='flicker', help='flicker/vg/refit', required=False)

    parser.add_argument('--start', type=int, default=0, help='ae folder path', required=False)
    parser.add_argument('--end', type=int, default=100000, help='ae folder path', required=False)

    args = vars(parser.parse_args())

    args['path_best'] = os.path.join('results', 'gpu' + str(args['path_ae']),
                                     'net_best.pth')
    args['clip_eval'] = bool(int(args['clip_eval']))
    main(args=args)