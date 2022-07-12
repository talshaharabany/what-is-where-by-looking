import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def rand_bbox(size, lam, center=False, attcen=None):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    elif len(size) == 2:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    if attcen is None:
        # uniform
        cx = 0
        cy = 0
        if W>0 and H>0:
            cx = np.random.randint(W)
            cy = np.random.randint(H)
        if center:
            cx = int(W/2)
            cy = int(H/2)
    else:
        cx = attcen[0]
        cy = attcen[1]

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_bbox(imgsize=(224,224),beta=1.0):

    r = np.random.rand(1)
    lam = np.random.beta(beta, beta)
    bbx1, bby1, bbx2, bby2 = rand_bbox(imgsize, lam)

    return [bbx1,bby1,bbx2,bby2]


def generate_bbox(image, cam, gt_bbox, thr_val=0.1):
    '''
    image: single image with shape (h, w, 3)
    cam: single image with shape (h, w, 1)
    gt_bbox: [x, y, x + w, y + h]
    thr_val: float value (0~1)
    return estimated bounding box, blend image with boxes
    '''
    # cam = gaussian_filter(cam, 3)

    image_height, image_width, _ = image.shape

    if gt_bbox is not None:
        _gt_bbox = list()
        _gt_bbox.append(max(int(gt_bbox[0]), 0))
        _gt_bbox.append(max(int(gt_bbox[1]), 0))
        _gt_bbox.append(min(int(gt_bbox[2]), image_height-1))
        _gt_bbox.append(min(int(gt_bbox[3]), image_width))
    # cam = cv2.resize(cam, (image_height, image_width),
    #                  interpolation=cv2.INTER_CUBIC)
    heatmap = intensity_to_rgb(cam, normalize=True).astype('uint8')
    # heatmap_BGR = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    blend = image * 128 + 128
    gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)

    thr_val = thr_val * np.max(gray_heatmap)

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

    _img_bbox = (image.copy()).astype('uint8')

    blend_bbox = blend.copy()
    if gt_bbox is not None:
        cv2.rectangle(blend_bbox,
                      (_gt_bbox[0], _gt_bbox[1]),
                      (_gt_bbox[2], _gt_bbox[3]),
                      (0, 255, 0), 2)

    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
        cv2.rectangle(blend_bbox,
                      (x, y),
                      (x + w, y + h),
                      (255, 0, 0), 2)
        energy = thr_gray_heatmap[y:y+h, x:x+w].sum()/thr_gray_heatmap.sum()
    else:
        estimated_bbox = [0, 0, 1, 1]
        energy = 0

    return estimated_bbox, blend_bbox, thr_gray_heatmap


def intensity_to_rgb(intensity, cmap='cubehelix', normalize=False):
    """
    Convert a 1-channel matrix of intensities to an RGB image employing a colormap.
    This function requires matplotlib. See `matplotlib colormaps
    <http://matplotlib.org/examples/color/colormaps_reference.html>`_ for a
    list of available colormap.
    Args:
        intensity (np.ndarray): array of intensities such as saliency.
        cmap (str): name of the colormap to use.
        normalize (bool): if True, will normalize the intensity so that it has
            minimum 0 and maximum 1.
    Returns:
        np.ndarray: an RGB float32 image in range [0, 255], a colored heatmap.
    """
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype('float32') * 255.0


def calculate_IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0,(xB - xA + 1)) * max(0,(yB - yA + 1))

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_spm(input, target, model):
    imgsize = (224, 224)
    bs = input.size(0)
    with torch.no_grad():
        output, fms = model(input, flag=True)
        clsw = model.fc1
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0), weight.size(1), 1, 1)
        fms = F.relu(fms)
        poolfea = F.adaptive_avg_pool2d(fms, (1, 1)).squeeze()
        clslogit = F.softmax(clsw.forward(poolfea))
        logitlist = []
        for i in range(bs):
            logitlist.append(clslogit[i, target[i]])

        out = F.conv2d(fms, weight, bias=bias)

        outmaps = []
        for i in range(bs):
            evimap = out[i, target[i]]
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0),1,outmaps.size(1),outmaps.size(2))
            outmaps = F.interpolate(outmaps, imgsize, mode='bilinear', align_corners=False)

        outmaps = outmaps.squeeze()

        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()
    return outmaps


class MaskEvaluator:
    def __init__(self, **kwargs):
        super(MaskEvaluator, self).__init__()
        self.cam_threshold_list = list(np.arange(0, 1, 0.01))
        self.num_bins = len(self.cam_threshold_list) + 2
        self.threshold_list_right_edge = np.append(self.cam_threshold_list,
                                                   [1.0, 2.0, 3.0])
        self.gt_true_score_hist = np.zeros(self.num_bins, dtype=np.float)
        self.gt_false_score_hist = np.zeros(self.num_bins, dtype=np.float)

    def accumulate(self, scoremap, gt_mask):
        """
        Score histograms over the score map values at GT positive and negative
        pixels are computed.
        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """
        gt_true_scores = scoremap[gt_mask == 1]
        gt_false_scores = scoremap[gt_mask == 0]

        # histograms in ascending order
        gt_true_hist, _ = np.histogram(gt_true_scores,
                                       bins=self.threshold_list_right_edge)
        self.gt_true_score_hist += gt_true_hist.astype(np.float)

        gt_false_hist, _ = np.histogram(gt_false_scores,
                                        bins=self.threshold_list_right_edge)
        self.gt_false_score_hist += gt_false_hist.astype(np.float)

    def compute(self):
        """
        Arrays are arranged in the following convention (bin edges):
        gt_true_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        gt_false_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        tp, fn, tn, fp: >=2.0, >=1.0, ..., >=0.0
        Returns:
            auc: float. The area-under-curve of the precision-recall curve.
               Also known as average precision (AP).
        """
        num_gt_true = self.gt_true_score_hist.sum()
        tp = self.gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = self.gt_false_score_hist.sum()
        fp = self.gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            raise RuntimeError("No positive prediction in the eval set.")

        non_zero_indices = (tp + fp) != 0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        auc *= 100

        # print("Mask AUC on split {}: {}".format(self.split, auc))
        return auc


def interpret(image, text, model, device, index=None):
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    if index is None:
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()
    one_hot.backward(retain_graph=False)

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    for i, blk in enumerate(image_attn_blocks):
        if i <= 10:
          continue
        grad = blk.attn_grad
        cam = blk.attn_probs
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        R += torch.matmul(cam, R)
    R[0, 0] = 0
    image_relevance = R[0, 1:]

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.detach().reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    del image_attn_blocks, R, one_hot, grad, cam
    torch.cuda.empty_cache()
    return image_relevance


def interpret_batch(image, text, model, device, index=None, ground=False):
    bs = image.shape[0]
    logits_per_image, logits_per_text = model(image, text)
    if index is None:
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((bs, logits_per_image.size()[-1]), dtype=np.float32)
    one_hot[:, index] = 1
    if ground:
        one_hot = np.eye(bs, dtype=np.float32)
    one_hot = torch.from_numpy(one_hot).requires_grad_(True).cuda()
    one_hot = torch.sum(one_hot * logits_per_image, dim=1).mean()
    model.zero_grad()
    one_hot.backward(retain_graph=False)

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).\
        to('cuda:' + str(image.get_device())).repeat(bs, 1, 1)
    for i, blk in enumerate(image_attn_blocks):
        if i <= 10:
          continue
        grad = blk.attn_grad
        cam = blk.attn_probs
        cam = cam.reshape(bs, -1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(bs, -1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=1)
        R += torch.matmul(cam, R)
    R[:, 0, 0] = 0
    image_relevance = R[:, 0, 1:].detach().clone()

    dim = int(image_relevance.shape[1] ** 0.5)
    image_relevance = image_relevance.reshape(bs, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')

    min_value = image_relevance.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, 224, 224)
    max_value = image_relevance.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, 224, 224)
    image_relevance = (image_relevance - min_value) / (max_value - min_value)
    return image_relevance


def get_clip_maps(image, class_text, clip_model, device):
    out = torch.zeros((image.shape[0], 1, 224, 224))
    for i in range(image.shape[0]):
        tmp = interpret(image[i:i+1], class_text, clip_model, device, index=0)
        out[i, 0, :, :] = tmp
    return out.cuda()


def interpret_new(images, texts, model, device):
    bs = images.shape[0]
    batch_size = texts.shape[0]
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < 11:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=False)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:].detach().clone()

    dim = int(image_relevance.shape[1] ** 0.5)
    image_relevance = image_relevance.reshape(bs, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')

    min_value = image_relevance.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, 224, 224)
    max_value = image_relevance.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, 224, 224)
    image_relevance = (image_relevance - min_value) / (max_value - min_value)
    return image_relevance


if __name__ == "__main__":
    import CLIP.clip as clip
    from datasets.cub import get_dataset
    from tqdm import tqdm

    trainset, testset = get_dataset(size=224, is_bbox=True, datadir='/path_to_data/CUB/CUB_200_2011')
    ds = torch.utils.data.DataLoader(trainset,
                                     batch_size=16,
                                     num_workers=0,
                                     shuffle=True,
                                     drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    class_text = clip.tokenize(trainset.text_class).to(device)
    pbar = tqdm(ds)

    for i, inputs in enumerate(pbar):
        real_imgs = inputs[0].cuda()
        image_relevance1 = get_clip_maps(real_imgs, class_text, clip_model, device)
        image_relevance2 = interpret_batch(real_imgs, class_text, clip_model, device, index=None)
        cv2.imwrite('no_batch.jpg', (255 * image_relevance1[0, 0].cpu().detach().numpy()).astype(np.uint8))
        cv2.imwrite('batch.jpg', (255 * image_relevance2[0, 0].cpu().detach().numpy()).astype(np.uint8))
        print('kaki!')