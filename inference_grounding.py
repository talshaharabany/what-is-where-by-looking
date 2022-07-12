import torch.utils.data
import os

from model import *
from datasets.flicker import get_flicker1K_dataset
from datasets.referit_loader import get_refit_test_dataset
from datasets.visual_genome import get_VGtest_dataset

from utils_grounding import *
from utils import interpret, interpret_batch, interpret_new
import CLIP.clip as clip
import clip as clip_org
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from sentence_transformers import util
import pickle
from BLIP.models.blip_itm import blip_itm


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
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def save_image_bbox(img_path, proposals, org_caption):
    pil_img = Image.open(img_path)
    image = np.array(pil_img.resize((224, 224)))
    image = cv2.copyMakeBorder(image, 0, 20, 0, 256, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gap = 15
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, org_caption, (10, 240), font, fontScale=0.3, color=(0, 0, 0), thickness=1)
    for ix, bboxs in enumerate(proposals):
        curr_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        text = bboxs[-1]
        cv2.putText(image, str(ix) + ': ' + text, (224, 10 + gap * ix), font, fontScale=0.3, color=(0, 0, 0),
                    thickness=1)
        for bbox in bboxs[:-1]:
            gxa = int(bbox[0])
            gya = int(bbox[1])
            gxb = int(bbox[2])
            gyb = int(bbox[3])
            cv2.putText(image, str(ix), (gxa, gya + 10), font, fontScale=0.32, color=(0, 0, 0), thickness=1)
            image = cv2.rectangle(image, (gxa, gya), (gxb, gyb), curr_color, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def inference_wwbl(ds, model, clip_model, text_generator, idx, ss, args, predictions = {}):
    (flicker_img, meta, size, img_path) = ds[idx]
    flicker_img = flicker_img.cuda()
    pil_img = Image.open(img_path)
    np_img = np.array(pil_img.resize((224, 224)))
    curr_image = flicker_img.repeat(1, 1, 1, 1)
    ss.setBaseImage(curr_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
    ss.switchToSelectiveSearchFast()
    proposals = ss.process()
    proposals = list(proposals)
    for bbox_size in [1/2, 1/3, 1/4]:
        wh = int(224*bbox_size) + 1
        for tl_x in list(range(0, 224, wh)):
            for tl_y in list(range(0, 224, wh)):
                proposals.append([tl_x, tl_y, wh, wh])
    blip_crop_imgs = []
    proposals = [x for x in proposals if x[2]*x[3] > 0.1*224*224]
    for p in proposals[:]:
        x, y, w, h = p
        crop = np_img[y:y + h, x:x + w, :]
        blip_crop_img = load_blip_img(Image.fromarray(crop))
        blip_crop_imgs.append(blip_crop_img)
    crops_captions = []
    chunk_size = 120
    with torch.no_grad():
        for crops_batch in [blip_crop_imgs[i:i + chunk_size] for i in range(0, len(blip_crop_imgs), chunk_size)]:
            crops_captions.extend(text_generator.generate(torch.cat(crops_batch), sample=False, num_beams=3,
                                                          max_length=30, min_length=5))
    del crops_batch, blip_crop_imgs
    crop_caption_tokenize = clip_org.tokenize(crops_captions).to('cuda').detach()
    z_crop_texts = norm_z(clip_model.encode_text(crop_caption_tokenize)).detach().float().cpu()
    clusters = util.community_detection(z_crop_texts,
                                        min_community_size=args['cluster_min_size'],
                                        threshold=args['cluster_threshold'])
    del crop_caption_tokenize
    for cluster_idx, cluster in enumerate(clusters):
        cluster_txts = [crops_captions[x] for x in cluster]
        cluster_vectors = torch.stack([z_crop_texts[x] for x in cluster])

        mean_cluster_vec = torch.mean(cluster_vectors, dim=0)
        mean_cluster_vec = mean_cluster_vec / mean_cluster_vec.norm()

        closest_vector_idx = torch.argmax(cluster_vectors @ mean_cluster_vec).item()
        closest_txt = cluster_txts[closest_vector_idx]

        if bool(int(args['clip_eval'])):
            text = clip.tokenize(closest_txt).to('cuda').detach()
            heatmap = interpret(curr_image.detach(), text, clip_model, 'cuda')
        else:
            text = clip.tokenize(closest_txt).to('cuda')
            z_text = norm_z(clip_model.encode_text(text))
            heatmap = model(curr_image, z_text)
        mask = heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy()
        torch.cuda.empty_cache()
        if idx not in predictions:
            predictions[idx] = []

        predictions[idx].append((closest_txt, mask, []))
    return predictions


def inference_bbox(ds, model, clip_model, epoch, args):
    pbar = tqdm(ds)
    cnt_overall = 0
    cnt_correct = 0
    cnt_correct_hit = 0
    att_correct = 0
    for i, inputs in enumerate(pbar):
        real_imgs, meta, size, _ = inputs
        if len(list(meta.keys())) == 0:
            continue
        real_imgs = real_imgs.cuda()
        size = [int(size[0]), int(size[1])]
        if args['dataset'] == "flicker" or args['task'] == "vg_train" or args['task'] == "coco":
            for sen in meta.keys():
                item = meta[sen]
                title, bbox = no_tuple(item['sentences']), item['bbox']
                text = clip.tokenize(title).to('cuda')
                z_text = norm_z(clip_model.encode_text(text))
                curr_image = real_imgs.repeat(z_text.shape[0], 1, 1, 1)
                heatmap = model(curr_image, z_text)
                heatmap = heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy()
                bbox_c, hit_c, att_c = calc_correctness(bbox, heatmap.astype(np.float), size)
                cnt_correct += bbox_c
                cnt_correct_hit += hit_c
                att_correct += att_c
                cnt_overall += 1
        else:
            text = []
            bboxes = []
            for item in list(meta.values()):
                text.append('image of ' + item['sentences'][0])
                bboxes.append(item['bbox'])
            text = clip.tokenize(text).to('cuda')
            z_text = norm_z(clip_model.encode_text(text))
            curr_image = real_imgs.repeat(z_text.shape[0], 1, 1, 1)
            with torch.no_grad():
                heatmaps = model(curr_image, z_text)
                for k, heatmap in enumerate(heatmaps):
                    heatmap = heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy().astype(np.float)
                    bbox_c, hit_c, att_c = calc_correctness(bboxes[k], heatmap.astype(np.float), size)
                    cnt_correct += bbox_c
                    cnt_correct_hit += hit_c
                    att_correct += att_c
                    cnt_overall += 1
        bbox_correctness = 100. * cnt_correct / cnt_overall
        hit_correctness = 100. * cnt_correct_hit / cnt_overall
        att_correctness = 100. * att_correct / cnt_overall
        prnt = 'bbox_correctness:{:.2f}; hit_correctness:{:.2f}; att_correctness:{:.2f}'.\
            format(bbox_correctness, hit_correctness, att_correctness)
        pbar.set_description(prnt)
    return hit_correctness


def inference_clip(ds, clip_model, args):
    pbar = tqdm(ds)
    cnt_overall = 0
    cnt_correct = 0
    cnt_correct_hit = 0
    att_correct = 0
    for i, (real_imgs, meta, size, _) in enumerate(pbar):
        if len(list(meta.keys())) == 0:
            continue
        real_imgs = real_imgs.cuda()
        size = [int(size[0]), int(size[1])]
        if args['dataset'] == "flicker" or args['task'] == "vg_train" or args['task'] == "coco":
            for sen in meta.keys():
                item = meta[sen]
                title, bbox = no_tuple(item['sentences']), item['bbox']
                text = clip.tokenize(title).to('cuda')
                curr_image = real_imgs.repeat(text.shape[0], 1, 1, 1)
                index = np.cumsum(np.ones(text.shape[0])).astype(np.uint8) - 1
                heatmap = interpret_batch(curr_image, text, clip_model, 'cuda', index=index, ground=True)
                heatmap = heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy()
                bbox_c, hit_c, att_c = calc_correctness(bbox, heatmap.astype(np.float), size)
                cnt_correct += bbox_c
                cnt_correct_hit += hit_c
                att_correct += att_c
                cnt_overall += 1
        else:
            text = []
            bboxes = []
            for item in list(meta.values()):
                text.append(item['sentences'][0])
                bboxes.append(item['bbox'])
            text_tokens = clip.tokenize(text).to('cuda')
            curr_image = real_imgs.repeat(text_tokens.shape[0], 1, 1, 1)
            index = np.cumsum(np.ones(text_tokens.shape[0])).astype(np.uint8) - 1
            heatmaps = interpret_new(curr_image, text_tokens, clip_model, 'cuda')
            for k, heatmap in enumerate(heatmaps):
                heatmap = heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy().astype(np.float)
                bbox_c, hit_c, att_c = calc_correctness(bboxes[k], heatmap.astype(np.float), size)
                cnt_correct += bbox_c
                cnt_correct_hit += hit_c
                att_correct += att_c
                cnt_overall += 1
        bbox_correctness = 100. * cnt_correct / cnt_overall
        hit_correctness = 100. * cnt_correct_hit / cnt_overall
        att_correctness = 100. * att_correct / cnt_overall
        prnt = 'bbox_correctness:{:.2f}; hit_correctness:{:.2f}; att_correctness:{:.2f}'. \
            format(bbox_correctness, hit_correctness, att_correctness)
        pbar.set_description(prnt)


def main(args=None):
    gpu_num = torch.cuda.device_count()
    model = torch.nn.DataParallel(MultiModel(args=args), list(range(gpu_num))).cuda().eval()
    model1 = torch.load(args['path_best'])
    model.load_state_dict(model1.state_dict())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args['dataset'] == 'flicker':
        testset = get_flicker1K_dataset(args=args)
    elif args['dataset'] == 'refit':
        testset = get_refit_test_dataset(args=args)
    elif args['dataset'] == 'vg':
        testset = get_VGtest_dataset(args=args)
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    ds = torch.utils.data.DataLoader(testset,
                                     batch_size=1,
                                     num_workers=int(args['nW']),
                                     shuffle=False,
                                     drop_last=False)
    if args['task'] == 'grounding':
        if args['dataset'] == 'flicker' or args['dataset'] == 'refit':
            if args['clip_eval']:
                inference_clip(ds, clip_model, args)
            else:
                inference_bbox(ds, model.eval(), clip_model, 0, args)
        elif args['dataset'] == 'vg':
            if args['clip_eval']:
                inference_clip(ds, clip_model, args)
            else:
                inference_bbox(ds, model.eval(), clip_model, 0, args)
    elif args['task'] == 'app':
        from BLIP.models.blip import blip_decoder
        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
        image_size = 384
        text_generator = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
        text_generator.eval()
        text_generator = text_generator.to(device)
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        predictions = {}
        out_dir_name = f'predictions_data-{args["dataset"]}-_th-{args["cluster_threshold"]}_min-{args["cluster_min_size"]}_model-{args["path_ae"]}'
        os.makedirs(out_dir_name, exist_ok=True)
        if not bool(int(args['clip_eval'])):
            del clip_model
            clip_model, _ = clip_org.load("ViT-B/32", device=device)
            with torch.no_grad():
                for idx in tqdm(range(args['start'], args['end'])):
                    predictions = inference_wwbl(testset, model, clip_model.eval(),
                                                 text_generator, idx, ss, args, predictions=predictions)
                    torch.cuda.empty_cache()
        else:
            del model
            for idx in tqdm(range(args['start'], args['end'])):
                predictions = inference_wwbl(testset, None, clip_model.eval(),
                                             text_generator, idx, ss, args, predictions=predictions)
                torch.cuda.empty_cache()
        if args['save_prediction'] == 1:
            with open(f'{out_dir_name}/predictions_{args["start"]}-{args["end"]}.pickle', 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-nW', '--nW', default=0, help='number of workers', required=False)
    parser.add_argument('-task', '--task', default='grounding', help='dataset task', required=False)
    parser.add_argument('-th', '--th', default=0.1, help='evaluation th', required=False)
    parser.add_argument('-nC', '--nC', default=200, help='number of classes', required=False)
    parser.add_argument('-Isize', '--Isize', default=304, help='image size', required=False)
    parser.add_argument('-pretrained', '--pretrained', default=False, help='pretrined models', required=False)
    parser.add_argument('-path_ae', '--path_ae', default=22, help='ae folder path', required=False)
    parser.add_argument('-clip_eval', '--clip_eval', default=0, help='ae folder path', required=False)
    parser.add_argument('-data_path', '--data_path',
                        default='/path_to_data/cars', help='data set path', required=False)
    parser.add_argument('-val_path', '--val_path', default='', help='data set path', required=False)
    parser.add_argument('--start', type=int, default=0, help='ae folder path', required=False)
    parser.add_argument('--end', type=int, default=1000, help='ae folder path', required=False)
    parser.add_argument('--cluster_threshold', type=float, default=0.85, help='ae folder path', required=False)
    parser.add_argument('--cluster_min_size', type=int, default=2, help='ae folder path', required=False)
    parser.add_argument('--save_prediction', type=int, default=1, help='ae folder path', required=False)
    parser.add_argument('-dataset', '--dataset', default='flicker', help='dataset task', required=False)
    parser.add_argument('-img_path', '--img_path', default=1, help='dataset task', required=False)
    args = vars(parser.parse_args())

    args['path_best'] = os.path.join('results', 'gpu' + str(args['path_ae']),
                                     'net_best.pth')
    args['clip_eval'] = bool(int(args['clip_eval']))
    main(args=args)
