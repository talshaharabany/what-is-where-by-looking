import os
from typing import List, Tuple, Dict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from io import StringIO
import sys


def evaluate(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int],
             probs: List[float]) -> Tuple[float, str]:
    self._write_results(path_to_results_dir, image_ids, bboxes, classes, probs)

    annType = 'bbox'
    path_to_coco_dir = os.path.join(self._path_to_data_dir, 'COCO')
    path_to_annotations_dir = os.path.join(path_to_coco_dir, 'annotations')
    path_to_annotation = os.path.join(path_to_annotations_dir, 'instances_val2017.json')

    cocoGt = COCO(path_to_annotation)
    cocoDt = cocoGt.loadRes(os.path.join(path_to_results_dir, 'results.json'))

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.evaluate()
    cocoEval.accumulate()

    original_stdout = sys.stdout
    string_stdout = StringIO()
    sys.stdout = string_stdout
    cocoEval.summarize()
    sys.stdout = original_stdout

    mean_ap = cocoEval.stats[0].item()  # stats[0] records AP@[0.5:0.95]
    detail = string_stdout.getvalue()

    return mean_ap, detail