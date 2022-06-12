from skimage.feature import peak_local_max
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import sys
import torch
# from support.layer.nms import nms
import torchvision

from skimage import filters
from skimage.measure import regionprops


rel_peak_thr = .3
rel_rel_thr = .3
ioa_thr = .6
topk_boxes = 3


def heat2bbox(heat_map, original_image_shape):
    h, w = heat_map.shape

    bounding_boxes = []

    heat_map = heat_map - np.min(heat_map)
    heat_map = heat_map / np.max(heat_map)

    bboxes = []
    box_scores = []

    peak_coords = peak_local_max(heat_map, exclude_border=False,
                                 threshold_rel=rel_peak_thr)  # find local peaks of heat map

    heat_resized = cv2.resize(heat_map, (
    original_image_shape[1], original_image_shape[0]))  ## resize heat map to original image shape
    peak_coords_resized = ((peak_coords + 0.5) *
                           np.asarray([original_image_shape]) /
                           np.asarray([[h, w]])
                           ).astype('int32')

    for pk_coord in peak_coords_resized:
        pk_value = heat_resized[tuple(pk_coord)]
        mask = heat_resized > pk_value * rel_rel_thr
        labeled, n = ndi.label(mask)
        l = labeled[tuple(pk_coord)]
        yy, xx = np.where(labeled == l)
        min_x = np.min(xx)
        min_y = np.min(yy)
        max_x = np.max(xx)
        max_y = np.max(yy)
        bboxes.append((min_x, min_y, max_x, max_y))
        box_scores.append(pk_value)  # you can change to pk_value * probability of sentence matching image or etc.

    ## Merging boxes that overlap too much
    box_idx = np.argsort(-np.asarray(box_scores))
    box_idx = box_idx[:min(topk_boxes, len(box_scores))]
    bboxes = [bboxes[i] for i in box_idx]
    box_scores = [box_scores[i] for i in box_idx]

    to_remove = []
    for iii in range(len(bboxes)):
        for iiii in range(iii):
            if iiii in to_remove:
                continue
            b1 = bboxes[iii]
            b2 = bboxes[iiii]
            isec = max(min(b1[2], b2[2]) - max(b1[0], b2[0]), 0) * max(min(b1[3], b2[3]) - max(b1[1], b2[1]), 0)
            ioa1 = isec / ((b1[2] - b1[0]) * (b1[3] - b1[1]))
            ioa2 = isec / ((b2[2] - b2[0]) * (b2[3] - b2[1]))
            if ioa1 > ioa_thr and ioa1 == ioa2:
                to_remove.append(iii)
            elif ioa1 > ioa_thr and ioa1 >= ioa2:
                to_remove.append(iii)
            elif ioa2 > ioa_thr and ioa2 >= ioa1:
                to_remove.append(iiii)

    for i in range(len(bboxes)):
        if i not in to_remove:
            bounding_boxes.append({
                'score': box_scores[i],
                'bbox': bboxes[i],
                'bbox_normalized': np.asarray([
                    bboxes[i][0] / heat_resized.shape[1],
                    bboxes[i][1] / heat_resized.shape[0],
                    bboxes[i][2] / heat_resized.shape[1],
                    bboxes[i][3] / heat_resized.shape[0],
                ]),
            })

    return bounding_boxes


def img_heat_bbox_disp(image, heat_map, title='', en_name='', alpha=0.6, cmap='viridis', cbar='False', dot_max=False,
                       bboxes=[], order=None, show=True):
    thr_hit = 1  # a bbox is acceptable if hit point is in middle 85% of bbox area
    thr_fit = .60  # the biggest acceptable bbox should not exceed 60% of the image
    H, W = image.shape[0:2]
    # resize heat map
    heat_map_resized = cv2.resize(heat_map, (H, W))

    # display
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(title, size=15)
    ax = plt.subplot(1, 3, 1)
    plt.imshow(image)
    if dot_max:
        max_loc = np.unravel_index(np.argmax(heat_map_resized, axis=None), heat_map_resized.shape)
        plt.scatter(x=max_loc[1], y=max_loc[0], edgecolor='w', linewidth=3)

    if len(bboxes) > 0:  # it gets normalized bbox
        if order == None:
            order = 'xxyy'

        for i in range(len(bboxes)):
            bbox_norm = bboxes[i]
            if order == 'xxyy':
                x_min, x_max, y_min, y_max = int(bbox_norm[0] * W), int(bbox_norm[1] * W), int(bbox_norm[2] * H), int(
                    bbox_norm[3] * H)
            elif order == 'xyxy':
                x_min, x_max, y_min, y_max = int(bbox_norm[0] * W), int(bbox_norm[2] * W), int(bbox_norm[1] * H), int(
                    bbox_norm[3] * H)
            x_length, y_length = x_max - x_min, y_max - y_min
            box = plt.Rectangle((x_min, y_min), x_length, y_length, edgecolor='w', linewidth=3, fill=False)
            plt.gca().add_patch(box)
            if en_name != '':
                ax.text(x_min + .5 * x_length, y_min + 10, en_name,
                        verticalalignment='center', horizontalalignment='center',
                        # transform=ax.transAxes,
                        color='white', fontsize=15)
                # an = ax.annotate(en_name, xy=(x_min,y_min), xycoords="data", va="center", ha="center", bbox=dict(boxstyle="round", fc="w"))
                # plt.gca().add_patch(an)

    plt.imshow(heat_map_resized, alpha=alpha, cmap=cmap)

    # plt.figure(2, figsize=(6, 6))
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    # plt.figure(3, figsize=(6, 6))
    plt.subplot(1, 3, 3)
    plt.imshow(heat_map_resized)
    fig.tight_layout()
    fig.subplots_adjust(top=.85)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def filter_bbox(bbox_dict, order=None):
    thr_fit = .90  # the biggest acceptable bbox should not exceed 80% of the image
    if order == None:
        order = 'xxyy'

    filtered_bbox = []
    filtered_bbox_norm = []
    filtered_score = []
    if len(bbox_dict) > 0:  # it gets normalized bbox
        for i in range(len(bbox_dict)):
            bbox = bbox_dict[i]['bbox']
            bbox_norm = bbox_dict[i]['bbox_normalized']
            bbox_score = bbox_dict[i]['score']
            if order == 'xxyy':
                x_min, x_max, y_min, y_max = bbox_norm[0], bbox_norm[1], bbox_norm[2], bbox_norm[3]
            elif order == 'xyxy':
                x_min, x_max, y_min, y_max = bbox_norm[0], bbox_norm[2], bbox_norm[1], bbox_norm[3]
            if bbox_score > 0:
                x_length, y_length = x_max - x_min, y_max - y_min
                if x_length * y_length < thr_fit:
                    filtered_score.append(bbox_score)
                    filtered_bbox.append(bbox)
                    filtered_bbox_norm.append(bbox_norm)
    return filtered_bbox, filtered_bbox_norm, filtered_score



def crop_resize_im(image, bbox, size, order='xxyy'):
    H, W, _ = image.shape
    if order == 'xxyy':
        roi = image[int(bbox[2] * H):int(bbox[3] * H), int(bbox[0] * W):int(bbox[1] * W), :]
    elif order == 'xyxy':
        roi = image[int(bbox[1] * H):int(bbox[3] * H), int(bbox[0] * W):int(bbox[2] * W), :]
    roi = cv2.resize(roi, size)
    return roi


def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


def IoU(boxA, boxB):
    # order = xyxy
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou




def img_heat_bbox_disp(image, heat_map, size, title='', en_name='', alpha=0.6, cmap='viridis', cbar='False', dot_max=True,
                       gt=[], bbox=[], order=None, show=False, filename=''):

    heat_map = (256 * heat_map).astype(np.uint8)
    thr_hit = 1  # a bbox is acceptable if hit point is in middle 85% of bbox area
    thr_fit = .60  # the biggest acceptable bbox should not exceed 60% of the image
    H, W = size
    # resize heat map
    heat_map_resized = cv2.resize(heat_map, (W, H))
    image = cv2.resize(image, (W, H))
    # heat_map_resized = heat_map
    # display
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(title, size=15)
    ax = plt.subplot(1, 3, 1)
    plt.imshow(image)
    if dot_max:
        max_loc = np.unravel_index(np.argmax(heat_map_resized, axis=None), heat_map_resized.shape)
        plt.scatter(x=max_loc[1], y=max_loc[0], edgecolor='w', linewidth=3)

    # if len(bboxes) > 0:  # it gets normalized bbox
    #     if order == None:
    #         order = 'xxyy'
    #
    #     for i in range(len(bboxes)):
    #         bbox_norm = bboxes[i]
    #         if order == 'xxyy':
    #             x_min, x_max, y_min, y_max = int(bbox_norm[0] * W), int(bbox_norm[1] * W), int(bbox_norm[2] * H), int(
    #                 bbox_norm[3] * H)
    #         elif order == 'xyxy':
    #             x_min, x_max, y_min, y_max = int(bbox_norm[0] * W), int(bbox_norm[2] * W), int(bbox_norm[1] * H), int(
    #                 bbox_norm[3] * H)
    # x_length, y_length = x_max - x_min, y_max - y_min

    # if en_name != '':
    #     ax.text(x_min + .5 * x_length, y_min + 10, en_name,
    #             verticalalignment='center', horizontalalignment='center',
    #             # transform=ax.transAxes,
    #             color='white', fontsize=15)
        # an = ax.annotate(en_name, xy=(x_min,y_min), xycoords="data", va="center", ha="center", bbox=dict(boxstyle="round", fc="w"))
        # plt.gca().add_patch(an)

    plt.imshow(heat_map_resized, alpha=alpha, cmap=cmap)

    # plt.figure(2, figsize=(6, 6))
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    # box = plt.Rectangle((gt[0, 0], gt[0, 1]), gt[0, 2] - gt[0, 0], gt[0, 3] - gt[0, 1],
    #                     edgecolor='w', linewidth=3, fill=False, color='green')
    # plt.gca().add_patch(box)
    # if len(bbox) > 0:
    #     box = plt.Rectangle((bbox[0][0], bbox[0][1]), bbox[0][2] - bbox[0][0], bbox[0][3] - bbox[0][1],
    #                         edgecolor='w', linewidth=3, fill=False, color='red')
    #     plt.gca().add_patch(box)
    # plt.figure(3, figsize=(6, 6))
    plt.subplot(1, 3, 3)
    plt.imshow(heat_map_resized, cmap='jet')
    fig.tight_layout()
    fig.subplots_adjust(top=.85)

    if show:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
    plt.close()


def isCorrect(bbox_annot, bbox_pred, iou_thr=.5, size_h=224):
    for bbox_p in bbox_pred:
        bbox_p = (np.array(bbox_p) / size_h).tolist()
        for bbox_a in bbox_annot:
            if IoU(bbox_p, bbox_a) >= iou_thr:
                return 1
    return 0


def isCorrectHit(bbox_annot, heatmap, orig_img_shape):
    H, W = orig_img_shape
    heatmap_resized = cv2.resize(heatmap, (W, H))
    max_loc = np.unravel_index(np.argmax(heatmap_resized, axis=None), heatmap_resized.shape)

    # try:
    #     threshold_value = filters.threshold_minimum(heatmap_resized)
    #     labeled_foreground = (heatmap_resized > threshold_value).astype(int)
    #     properties = regionprops(labeled_foreground, heatmap_resized)
    #     center_of_mass = properties[0].centroid
    #     weighted_center_of_mass = properties[0].weighted_centroid
    #     max_loc = weighted_center_of_mass
    # except:
    #     max_loc = np.unravel_index(np.argmax(heatmap_resized, axis=None), heatmap_resized.shape)

    for bbox in bbox_annot:
        if bbox[0] <= max_loc[1] <= bbox[2] and bbox[1] <= max_loc[0] <= bbox[3]:
            return 1
    return 0


def check_percent(bboxes):
    for bbox in bboxes:
        x_length = bbox[2] - bbox[0]
        y_length = bbox[3] - bbox[1]
        if x_length * y_length < .05:
            return False
    return True


def union(bbox):
    if len(bbox) == 0:
        return []
    if type(bbox[0]) == type(0.0) or type(bbox[0]) == type(0):
        bbox = [bbox]
    maxes = np.max(bbox, axis=0)
    mins = np.min(bbox, axis=0)
    return [[mins[0], mins[1], maxes[2], maxes[3]]]


def attCorrectness(bbox_annot, heatmap, orig_img_shape):
    H, W = orig_img_shape
    heatmap_resized = cv2.resize(heatmap, (W, H))
    h_s = np.sum(heatmap_resized)
    if h_s == 0:
        return 0
    else:
        heatmap_resized /= h_s
    att_correctness = 0
    for bbox in bbox_annot:
        x0, y0, x1, y1 = bbox
        att_correctness += np.sum(heatmap_resized[y0:y1, x0:x1])
    return att_correctness


def calc_correctness(annot, heatmap, orig_img_shape):
    # bbox_dict = heat2bbox(heatmap, orig_img_shape)
    size_h = heatmap.shape[-1]
    bbox_dict = generate_bbox(heatmap)
    # bbox, bbox_norm, bbox_score = filter_bbox(bbox_dict=bbox_dict, order='xyxy')
    annot = process_gt_bbox(annot, orig_img_shape)
    bbox_norm_annot = union(annot['bbox_norm'])
    bbox_annot = annot['bbox']
    bbox_dict = union(np.array(bbox_dict)[:, :4])
    bbox_correctness = isCorrect(bbox_norm_annot, bbox_dict, iou_thr=.5, size_h=size_h)
    hit_correctness = isCorrectHit(bbox_annot, heatmap, orig_img_shape)
    # att_correctness = attCorrectness(bbox_annot, heatmap, orig_img_shape)
    # return bbox_correctness, hit_correctness, att_correctness, bbox
    # return bbox_correctness, hit_correctness, att_correctness
    return bbox_correctness, hit_correctness, 0


def process_gt_bbox(annot, orig_img_shape):
    out = {}
    h, w = orig_img_shape
    bbox = torch.tensor(annot).numpy()
    out['bbox'] = bbox.copy()
    bbox = bbox.astype(np.float)
    bbox[:, 0] = bbox[:, 0] / w
    bbox[:, 1] = bbox[:, 1] / h
    bbox[:, 2] = bbox[:, 2] / w
    bbox[:, 3] = bbox[:, 3] / h
    out['bbox_norm'] = bbox.copy()
    return out


def no_tuple(a):
    out = []
    for item in a:
        out.append(item[0])
    return out


def cluster_gt(bbox, heatmap):
    out = {}
    bbox = torch.tensor(bbox).numpy().squeeze()
    for i in range(len(bbox)):
        curr_heatmap = heatmap[i:i+1].clone()
        curr_bbox = bbox[i].copy()
        if len(out.keys()) == 0:
            out[str(len(out.keys()))] = (curr_bbox, curr_heatmap)
            continue
        flag = False
        for j, key in enumerate(out.keys()):
            old_bbox, old_heatmap = out[key]
            if IoU(curr_bbox, old_bbox) == 1:
                out[key] = (curr_bbox, torch.cat((old_heatmap, curr_heatmap), dim=0))
                flag = True
                break
        if not flag:
            out[str(len(out.keys()))] = (curr_bbox, curr_heatmap)

    for ix, key in enumerate(out.keys()):
        curr_bbox, old_heatmap = out[key]
        old_heatmap = old_heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy()
        out[key] = (np.expand_dims(curr_bbox, axis=0),
                    old_heatmap)
    return out


def intensity_to_rgb(intensity, cmap='cubehelix', normalize=False):
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype('float32') * 255.0


def generate_bbox(cam, threshold=0.5, nms_threshold=0.05, max_drop_th=0.5):
    heatmap = intensity_to_rgb(cam, normalize=True).astype('uint8')
    gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)

    thr_val = threshold * np.max(gray_heatmap)

    _, thr_gray_heatmap = cv2.threshold(gray_heatmap,
                                        int(thr_val), 255,
                                        cv2.THRESH_TOZERO)
    try:
        _, contours, _ = cv2.findContours(thr_gray_heatmap,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        proposals = [cv2.boundingRect(c) for c in contours]
        # proposals = [(x, y, w, h) for (x, y, w, h) in proposals if h * w > 0.05 * 224 * 224]
        if len(proposals) > 0:
            proposals_with_conf = [thr_gray_heatmap[y:y + h, x:x + w].mean()/255 for (x, y, w, h) in proposals]
            inx = torchvision.ops.nms(torch.tensor(proposals).float(),
                                      torch.tensor(proposals_with_conf).float(),
                                      nms_threshold)
            estimated_bbox = torch.cat((torch.tensor(proposals).float()[inx],
                                        torch.tensor(proposals_with_conf)[inx].unsqueeze(dim=1)),
                                       dim=1).tolist()
            estimated_bbox = [(x, y, x+w, y+h, conf) for (x, y, w, h, conf) in estimated_bbox
                              if conf > max_drop_th * np.max(proposals_with_conf)]
        else:
            estimated_bbox = [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]]
    else:
        estimated_bbox = [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]]
    return estimated_bbox


def generate_proposals(cam, threshold=0.5):
    heatmap = intensity_to_rgb(cam, normalize=True).astype('uint8')
    gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    thr_val = threshold * np.max(gray_heatmap)
    _, gray_heatmap = cv2.threshold(gray_heatmap,
                                    int(thr_val), 255,
                                    cv2.THRESH_TOZERO)
    try:
        _, contours, _ = cv2.findContours(gray_heatmap,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    proposals = []
    for c in contours:
        if c.shape[0]<5:
            continue
        x, y, w, h = cv2.boundingRect(c)
        energy = gray_heatmap[y:y + h, x:x + w].sum() / gray_heatmap.sum()
        if energy < 1:
            proposals.append([x, y, x + w, y + h, energy])
    return proposals, gray_heatmap


def get_scores(proposals, mask):
    scores = []
    bboxes = []
    for p in proposals:
        x, y, w, h = p
        if w*h < 0.05*224*224:
            continue
        energy = (mask[y:y + h, x:x + w, :]/255/3).sum()/(w*h)
        scores.append(energy)
        bboxes.append([x, y, x+w, h+y])
    return bboxes, scores

