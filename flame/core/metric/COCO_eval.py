import json
import numpy as np
import pycocotools.mask as mask_util

from pathlib import Path
from ignite.metrics import Metric
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from typing import List, Dict, Optional, Callable, Union, Tuple


class COCOEvaluator(Metric):
    def __init__(
        self,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        classes: Optional[Union[Dict[str, int], int]] = None,
        annotation_file: Optional[str] = None,
        label_to_coco_label: Optional[Dict[int, int]] = {None: 0},
        annotation_types: List[str] = ['segm', 'bbox'],  # segm
        detection_path: str = None,
        ground_truth_path: str = None,
        output_transform: Callable = lambda x: x
    ):
        super(COCOEvaluator, self).__init__(output_transform)
        self.detection_path = detection_path
        self.ground_truth_path = ground_truth_path

        self.image_size = image_size

        if isinstance(classes, int):
            classes = {i: i for i in range(classes)}
        self.classes = classes

        self.annotation_file = annotation_file
        self.label_to_coco_label = label_to_coco_label

        if all([annotation_type in ['segm','bbox','keypoints'] for annotation_type in annotation_types]):
            self.annotation_types = annotation_types
        else:
            print('Annotation Type is invalid.')
        self.annotation_types = annotation_types

    def reset(self):
        self.detections: List[Dict] = []  # List[{'image_id': ..., 'category_id': ..., 'score': ..., 'bbox': ...}]

        if self.annotation_file is None:
            self.annot_id = 0
            # initialize groundtruth in COCO format
            self.ground_truths: Dict = {
                'annotations': [],
                'images': [],
                'categories': [
                    {'id': class_id, 'name': class_name}
                    for class_name, class_id in self.classes.items()
                ]
            }

    def update(self, output):
        preds, targets, image_infos = output
        for pred, target, image_info in zip(preds, targets, image_infos):
            pred_boxes = pred['boxes'].cpu().numpy()  # N x 4, format x1, y1, x2, y2
            pred_labels = pred['labels'].cpu().numpy().tolist()  # N
            pred_scores = pred['scores'].cpu().numpy().tolist()  # N
            pred_masks = pred['masks'].cpu().numpy()  # N x 1 x H x W

            image_id = target['image_id'].item()
            scale = max(image_info[1]) / max(self.image_size) if self.image_size is not None else 1.  # deal with input sample is paded to square (bottom-right)

            pred_boxes[:, [2, 3]] -= pred_boxes[:, [0, 1]]  # convert x1, y1, x2, y2 to x1, y1, w, h
            pred_boxes = (pred_boxes * scale).astype(np.int32).tolist()  # scale boxes to orginal size.

            # detection
            for box, score, label, mask in zip(pred_boxes, pred_scores, pred_labels, pred_masks):
                if label == -1:
                    continue

                # generate segmentation mask for evaluation
                mask = mask > 0.5
                rle = mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")

                self.detections.append(
                    {
                        'image_id': image_id,
                        'category_id': self.label_to_coco_label.get(label, label),
                        'segmentation': rle,
                        'score': score,
                        'bbox': box
                    }
                )

            if self.annotation_file is None:
                true_labels = target['labels'].cpu().numpy().tolist()
                true_areas = target['area'].cpu().numpy().tolist()
                true_boxes = target['boxes'].cpu().numpy()  # format x1, y1, x2, y2
                true_masks = target['masks'].cpu().numpy()
                true_boxes[:, [2, 3]] -= true_boxes[:, [0, 1]]  # convert x1, y1, x2, y2 to x1, y1, w, h
                true_boxes = (true_boxes * scale).astype(np.int32).tolist()  # scale boxes to orginal size.

                # create ground truth in COCO format if has no COCO ground truth file.
                for box, label, area, mask in zip(true_boxes, true_labels, true_areas, true_masks):
                    if label == -1:
                        continue

                    # generate segmentation mask for evaluation
                    rle = mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                    rle["counts"] = rle["counts"].decode("utf-8")

                    annotation = {
                        'image_id': image_id,
                        'category_id': label,
                        'bbox': box,
                        'segmentation': rle,
                        'iscrowd': 0,
                        'area': area,
                        'id': self.annot_id,
                    }
                    self.ground_truths['annotations'].append(annotation)
                    self.annot_id += 1

                self.ground_truths['images'].append(
                    {
                        'file_name': image_info[0],
                        'height': image_info[1][1],
                        'width': image_info[1][0],
                        'id': image_id,
                    }
                )

    def compute(self):
        if not len(self.detections):
            raise Exception('the model does not provide any valid output,\
                            check model architecture and the data input')

        # Create Ground Truth COCO Format
        if self.annotation_file is not None:
            groundtruth_coco = COCO(annotation_file=self.annotation_file)
        else:
            with open(file=self.ground_truth_path, mode='w', encoding='utf-8') as f:
                json.dump(self.ground_truths, f, ensure_ascii=False, indent=4)
            # save ground truth to json file and then load to COCO class
            groundtruth_coco = COCO(annotation_file=self.ground_truth_path)

        # Create Detection COCO Format
        if Path(self.detection_path).exists():
            Path(self.detection_path).unlink()

        with open(file=self.detection_path, mode='w', encoding='utf-8') as f:
            json.dump(self.detections, f, ensure_ascii=False, indent=4)

        # using COCO object to load detections.
        detection_coco = groundtruth_coco.loadRes(self.detection_path)

        # Evaluation
        for annotation_type in self.annotation_types:
            coco_eval = COCOeval(groundtruth_coco, detection_coco, annotation_type)
            coco_eval.params.imgIds = groundtruth_coco.getImgIds()
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
