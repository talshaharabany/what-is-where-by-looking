import torch.utils.data
import os

from datasets.flicker import get_flicker1K_dataset
from datasets.referit_loader import get_refit_test_dataset
from datasets.visual_genome import get_VGtest_dataset

from utils_grounding import *
import clip
from tqdm import tqdm
import pickle


def norm_z(z):
    return z / z.norm(dim=1).unsqueeze(dim=1)


def calc_pointing_metric(dataloader, clip_model, predictions):
    cnt_overall = 0
    cnt_correct = 0
    cnt_correct_hit = 0
    att_correct = 0
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
    return calc_pointing_metric(ds, clip_model, predictions)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-nW', '--nW', default=1, help='number of workers', required=False)
    parser.add_argument('-Isize', '--Isize', default=224, help='image size', required=False)
    parser.add_argument('-pretrained', '--pretrained', default=False, help='pretrined models', required=False)
    parser.add_argument('-img_path', '--img_path', default=1, help='ae folder path', required=False)
    parser.add_argument('-predictions_path', '--predictions_path',
                        default='predictions_data-flicker-_th-0.85_min-2_model-22',
                        help='ae folder path', required=False)
    parser.add_argument('-val_path', '--val_path', default='', help='data set path', required=False)
    parser.add_argument('--dataset', default='flicker', help='flicker/vg/refit', required=False)
    args = vars(parser.parse_args())
    main(args=args)

