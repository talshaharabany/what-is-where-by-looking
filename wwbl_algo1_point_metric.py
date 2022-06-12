import torch.utils.data
import os

from datasets.flicker import get_flicker1K_dataset
from datasets.referit_loader import get_refit_test_dataset
from datasets.visual_genome import get_VGtest_dataset

from utils_grounding import *
import clip
from tqdm import tqdm
import pickle
from PIL import Image
from sentence_transformers import util
import torch.nn.functional as F
from datasets.tfs import get_flicker_transform
import matplotlib.cm as cm


def save_image_bbox(img_path, proposals):
    pil_img = Image.open(img_path)
    image = np.array(pil_img.resize((224, 224)))
    image = cv2.copyMakeBorder(image, 0, 20, 0, 256, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gap = 15
    font = cv2.FONT_HERSHEY_SIMPLEX
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


def norm_z(z):
    return z / z.norm(dim=1).unsqueeze(dim=1)


def calc_pointing_metric(dataloader, clip_model, predictions):
    cnt_overall = 0
    cnt_correct = 0
    cnt_correct_hit = 0
    att_correct = 0

    # pbar = tqdm(range(0, len(dataloader)))
    pbar = tqdm(list(predictions.keys()))
    for idx in pbar:
        if idx not in predictions:
            continue

        (flicker_img, meta, size, img_path) = dataloader.dataset[idx]

        curr_predictions = predictions[idx]
        pred_heatmaps = [x[1] for x in curr_predictions]
        pred_texts = [x[0] for x in curr_predictions]
        pred_text_vectors = [norm_z(clip_model.encode_text(clip.tokenize(x).to('cuda'))) for x in pred_texts]

        for sen in meta.keys():
            item = meta[sen]
            size = [int(size[0]), int(size[1])]
            title, bbox = item['sentences'], item['bbox']
            gt_title_clip = clip.tokenize(title).to('cuda')
            gt_title_vector = norm_z(clip_model.encode_text(gt_title_clip)).mean(dim=0)

            closest_pred_idx = torch.argmax(torch.cat(pred_text_vectors) @ gt_title_vector).item()
            heatmap = pred_heatmaps[closest_pred_idx]

            # # # Always center
            # heatmap = np.zeros((224,224))
            # heatmap[112,112] = 1.0

            # hit_c = calc_correctness(bbox, heatmap.astype(np.float), size)
            bbox_c, hit_c, att_c = calc_correctness(bbox, heatmap.astype(np.float), size)
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

    var = 100. * cnt_correct_hit / cnt_overall
    print(var)


def vis(dataloader, clip_model, predictions, args):
    pbar = tqdm(list(predictions.keys()))
    for idx in pbar:
        if idx not in predictions:
            continue
        (flicker_img, meta, size, img_path) = dataloader.dataset[idx]
        curr_predictions = predictions[idx]
        pred_heatmaps = [x[1] for x in curr_predictions]
        pred_texts = [x[0] for x in curr_predictions]
        proposals = []
        for text, hetamap in zip(pred_texts, pred_heatmaps):
            estimated_bbox = generate_bbox(hetamap, threshold=0.5, nms_threshold=0.05, max_drop_th=0.5)
            estimated_bbox.append(text)
            proposals.append(estimated_bbox)

        crop_caption_tokenize = clip.tokenize(pred_texts).to('cuda').detach()
        z_crop_texts = norm_z(clip_model.encode_text(crop_caption_tokenize)).detach().float().cpu()
        clusters = util.community_detection(z_crop_texts,
                                            min_community_size=1,
                                            threshold=0.75)
        out = []
        for cluster in clusters:
            curr_proposals = np.array(proposals)[cluster].tolist()
            curr_texts = np.array(pred_texts)[cluster].tolist()
            curr_tokenizes = crop_caption_tokenize[cluster]
            curr_bboxs = []
            curr_conf = []
            curr_cation = []
            for curr_proposal, curr_tokenize, curr_text in zip(curr_proposals, curr_tokenizes, curr_texts):
                for p in curr_proposal[:-1]:
                    p = [int(i) for i in p]
                    x, y, w, h, score = p
                    crop = F.interpolate(flicker_img[:, y:y + h, x:x + w].unsqueeze(dim=0),
                                         size=(224, 224),
                                         mode="bilinear",
                                         align_corners=True)
                    score = clip_model(crop.cuda(), curr_tokenize.unsqueeze(dim=0))[0].item() / clip_model.logit_scale.exp()
                    curr_bboxs.append((x, y, w, h))
                    curr_conf.append(score)
                    curr_cation.append(curr_text + ': ' + str(float(score))[1:4])
            inx = torchvision.ops.nms(torch.tensor(curr_bboxs).float(),
                                      torch.tensor(curr_conf).float(),
                                      0.3)
            for i in inx:
                out.append([curr_bboxs[i], curr_cation[i]])
        image = save_image_bbox(img_path, out)
        cv2.imwrite(os.path.join('WWbL', str(idx) + '.jpg'), image)


def teaser(dataloader, predictions):
    pbar = tqdm(list(predictions.keys()))
    for idx in pbar:
        if idx > 240:
            break
        if idx not in predictions:
            continue
        (_, _, _, img_path) = dataloader.dataset[idx]
        pil_img = Image.open(img_path)
        np_img = np.array(pil_img.resize((224, 224)))
        curr_predictions = predictions[idx]
        pred_heatmaps = [x[1] for x in curr_predictions]
        pred_texts = [x[0] for x in curr_predictions]
        for text, heatmap in zip(pred_texts, pred_heatmaps):
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmap = (heatmap * 255).astype(np.uint8)
            thr_val = 0.5 * np.max(heatmap)
            _, thr_gray_heatmap = cv2.threshold(heatmap, int(thr_val), 255, cv2.THRESH_TOZERO)
            jet = cm.get_cmap("jet")
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[thr_gray_heatmap]
            superimposed_img = jet_heatmap * 0.5 + 0.5 * cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB) / 255
            # cv2.imwrite(os.path.join('teaser', str(idx) + '_' + text + '_heatmap.jpg'),
            #             cv2.cvtColor((255 * jet_colors[heatmap]).astype(np.uint8), cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join('teaser', str(idx) + '_' + text + '_thr.jpg'),
                        cv2.cvtColor((255 * superimposed_img).astype(np.uint8), cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join('teaser', str(idx) + '_' + text + '_image.jpg'),
                    cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args['dataset'] == 'flicker':
        testset = get_flicker1K_dataset(args=args)
    elif args['dataset'] == 'vg':
        testset = get_VGtest_dataset(args=args)
    elif args['dataset'] == 'refit':
        testset = get_refit_test_dataset(args=args)

    ds = torch.utils.data.DataLoader(testset,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=False,
                                     drop_last=False)
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)

    predictions_path = args['predictions_path']
    files = os.listdir(predictions_path)

    predictions = {}
    for file in files:
        curr_path = os.path.join(predictions_path, file)
        with open(curr_path, 'rb') as handle:
            curr_pred = pickle.load(handle)
            predictions.update(curr_pred)

    if args['vis']:
        teaser(ds, predictions)
    else:
        return calc_pointing_metric(ds, clip_model, predictions)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-nW', '--nW', default=1, help='number of workers', required=False)
    parser.add_argument('-backbone', '--backbone', default='vgg', help='number of workers', required=False)
    parser.add_argument('-order_ae', '--order_ae', default=16, help='order of the backbone - ae', required=False)
    parser.add_argument('-task', '--task', default='visualize', help='dataset task', required=False)  # metric/visualize
    parser.add_argument('-stop_th', '--stop_th', default=0.6, help='stopping rule th', required=False)
    parser.add_argument('-nC', '--nC', default=200, help='number of classes', required=False)
    parser.add_argument('-Isize', '--Isize', default=224, help='image size', required=False)
    parser.add_argument('-pretrained', '--pretrained', default=False, help='pretrined models', required=False)
    parser.add_argument('-path_ae', '--path_ae', default=41, help='ae folder path', required=False)
    parser.add_argument('-clip_eval', '--clip_eval', default=0, help='ae folder path', required=False)
    parser.add_argument('-img_path', '--img_path', default=1, help='ae folder path', required=False)
    parser.add_argument('-predictions_path', '--predictions_path',
                        default='predictions_data-flicker-_th-0.85_min-2_model-207',
                        help='ae folder path', required=False)
    parser.add_argument('-data_path', '--data_path',
                        default='/path_to_data/cars', help='data set path', required=False)
    parser.add_argument('--dataset',
                        default='flicker', help='flicker/vg/refit', required=False)
    parser.add_argument('--start', type=int, default=0, help='ae folder path', required=False)
    parser.add_argument('--end', type=int, default=100000, help='ae folder path', required=False)
    parser.add_argument('--vis', type=bool, default=True, help='ae folder path', required=False)

    args = vars(parser.parse_args())

    args['path_best'] = os.path.join('results', 'gpu' + str(args['path_ae']),
                                     'net_best.pth')
    args['clip_eval'] = bool(int(args['clip_eval']))
    main(args=args)

