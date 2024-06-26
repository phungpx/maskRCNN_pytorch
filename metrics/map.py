import torch
import numpy as np

from torch import nn
from shapely import geometry
from collections import Counter
from prettytable import PrettyTable
from typing import List, Tuple, Dict

from handler.evaluator import MetricBase


class mAP(MetricBase):
    def __init__(
        self,
        classes: Dict[str, int],
        iou_threshold: float = 0.5,
        method: str = "every_point_interpolation",  # or '11_point_interpolation'
        print_detail_mAP: bool = False,
        print_FP_files: bool = False,
        output_transform=lambda x: x,
    ):
        super(mAP, self).__init__(output_transform)
        self.eval_fn = MeanAveragePrecision(
            classes,
            iou_threshold,
            method,
            print_detail_mAP,
            print_FP_files,
        )

    def _get_bboxes(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        image_path: str,
    ) -> Tuple[list, list]:
        """
        Args:
            pred: {
                boxes: TensorFloat [N x 4],
                labels: TensorInt64 [N],
                scores: TensorFloat [N],
            }
            target: {
                image_id: TensorInt64 [M],
                boxes: TensorFloat [M x 4],
                labels: TensorInt64 [M],
            }
            image_path: str
        Output:
            detections: List[
                [image_idx, class_prediction, prob_score, [x1, y1, x2, y2], image_path]
            ]

            ground_truths: List[
                [image_idx, class_target, 1, [x1, y1, x2, y2], image_path]
            ]
        """
        detections, ground_truths = [], []

        image_idx = target["image_id"].item()
        target_boxes = target["boxes"].detach().cpu().numpy().tolist()
        target_labels = target["labels"].detach().cpu().numpy().tolist()

        pred_boxes = pred["boxes"].detach().cpu().numpy().tolist()
        pred_labels = pred["labels"].detach().cpu().numpy().tolist()
        pred_scores = pred["scores"].detach().cpu().numpy().tolist()

        for class_id, bbox in zip(target_labels, target_boxes):
            # [image_idx, class_target, 1, [x1, y1, x2, y2], image_path]
            ground_truth = [image_idx, class_id, 1, bbox, image_path]
            ground_truths.append(ground_truth)

        for class_id, score, bbox in zip(pred_labels, pred_scores, pred_boxes):
            # [image_idx, class_prediction, prob_score, [x1, y1, x2, y2], image_path]
            if class_id == -1 and score == 0:
                continue
            detection = [image_idx, class_id, score, bbox, image_path]
            detections.append(detection)

        return detections, ground_truths

    def reset(self):
        self.detections = []
        self.ground_truths = []

    def update(self, output):
        preds, targets, image_infos = output

        _iter_detections, _iter_ground_truths = [], []
        for pred, target, image_info in zip(preds, targets, image_infos):
            _detections, _ground_truths = self._get_bboxes(pred, target, image_info[0])
            self.detections.extend(_detections)
            self.ground_truths.extend(_ground_truths)
            _iter_detections.extend(_detections)
            _iter_ground_truths.extend(_ground_truths)

        return self.eval_fn(_iter_detections, _iter_ground_truths)

    def compute(self):
        metric = self.eval_fn(self.detections, self.ground_truths)
        return metric


class MeanAveragePrecision(nn.Module):
    def __init__(
        self,
        classes: Dict[str, int],
        iou_threshold: float = 0.5,
        method: str = "every_point_interpolation",  # or '11_point_interpolation'
        print_detail_mAP: bool = False,
        print_FP_files: bool = False,
    ) -> None:
        super(MeanAveragePrecision, self).__init__()
        self.classes = {
            class_id: class_name for class_name, class_id in classes.items()
        }
        self.iou_threshold = iou_threshold
        self.method = method
        # verbose
        self.print_detail_mAP = print_detail_mAP
        self.print_FP_files = print_FP_files

    def forward(self, detections: list, ground_truths: list) -> dict:
        r"""
        Args
            detections: list with all detections ([image_id, class_id, confidence, [x1, y1, x2, y2], image_path])
            ground_truths: list with all ground_truths ([image_id, class_id, 1., [x1, y1, x2, y2], image_path])
        Outputs:
            A list of dictionaries. Each dictionary contains information and metrics of each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total ground truths']: total number of ground truth positives;
            dict['total detections']: total number of detections;
            dict['total TP']: total number of True Positive detections;
            dict['total FP']: total number of False Negative detections;
        """
        results = []

        class_indices = sorted(map(int, self.classes.keys()))
        for class_id in class_indices:
            # get only detection of class_id -> class_detections
            class_detections = [
                detection for detection in detections if detection[1] == class_id
            ]
            num_detections = len(class_detections)

            # get only ground truth of class_id -> class_groundtruths
            class_groundtruths = [
                groundtruth
                for groundtruth in ground_truths
                if groundtruth[1] == class_id
            ]
            num_groundtruths = len(class_groundtruths)

            # initialize TP, FP with all 0 values and FP_image_paths is empty list.
            TP = [0] * num_detections
            FP = [0] * num_detections
            FP_image_paths = []

            # create dictionary with amount of ground truths for each image (same image id)
            # Ex: amount_class_groundtruths = {0: [0,0,0], 1: [0,0,0,0,0]}
            amount_class_groundtruths = Counter(
                [groundtruth[0] for groundtruth in class_groundtruths]
            )
            amount_class_groundtruths = {
                image_id: [0] * num_groundtruths
                for image_id, num_groundtruths in amount_class_groundtruths.items()
            }

            # sort class_detections by decreasing confidence.
            class_detections = sorted(
                class_detections, key=lambda detection: detection[2], reverse=True
            )

            for id_detection, detection in enumerate(class_detections):
                # collect grounth truths which have same image_id with detection -> image_groundtruths
                image_groundtruths = [
                    groundtruth
                    for groundtruth in class_groundtruths
                    if groundtruth[0] == detection[0]
                ]

                id_groundtruth_max, iou_max = 0, 0.0
                for id_groundtruth, groundtruth in enumerate(image_groundtruths):
                    iou = self._iou(boxA=detection[3], boxB=groundtruth[3])
                    if iou > iou_max:
                        iou_max = iou
                        id_groundtruth_max = id_groundtruth

                # assign class_detection (detection) as true positive/don't care/false positive
                if iou_max >= self.iou_threshold:
                    if amount_class_groundtruths[detection[0]][id_groundtruth_max] == 0:
                        TP[id_detection] = 1  # count as true positive
                        amount_class_groundtruths[detection[0]][
                            id_groundtruth_max
                        ] == 1  # flag as already 'seen'
                    else:
                        FP[id_detection] = 1  # count as false positive
                        # add FP_image_path
                        if detection[4] not in FP_image_paths:
                            FP_image_paths.append(detection[4])
                else:
                    FP[id_detection] = 1  # count as false positive
                    # add FP_image_path
                    if detection[4] not in FP_image_paths:
                        FP_image_paths.append(detection[4])

            # compute precision, recall and average precision
            cumulative_TP = np.cumsum(TP)
            cumulative_FP = np.cumsum(FP)

            if num_groundtruths > 0:
                precisions = np.divide(
                    cumulative_TP, (cumulative_FP + cumulative_TP)
                ).tolist()
                recalls = (cumulative_TP / num_groundtruths).tolist()
            else:
                precisions, recalls = [0] * num_detections, [0] * num_detections

            if self.method == "every_point_interpolation":
                average_precision, mrecalls, mprecisions = (
                    self.every_points_interpolated_AP(
                        recalls=recalls, precisions=precisions
                    )
                )
            elif self.method == "elevent_point_interpolation":
                average_precision, mrecalls, mprecisions = (
                    self.eleven_points_interpolated_AP(
                        recalls=recalls, precisions=precisions
                    )
                )
            else:
                raise RuntimeError("Interpolation Method is Wrong.")

            # if predictions have no bounding boxes of label and detection, ap will be 1.
            if (num_groundtruths == 0) and (num_detections == 0):
                average_precision = 1.0

            result = {
                "average_precision": average_precision,
                "class_name": self.classes[class_id],
                "recalls": recalls,
                "precisions": precisions,
                "interpolated_recalls": mrecalls,
                "interpolated_precisions": mprecisions,
                "total_detections": num_detections,
                "total_groundtruths": num_groundtruths,
                "total_TP": sum(TP),
                "total_FP": sum(FP),
                "FP_image_paths": FP_image_paths,
            }

            results.append(result)

        average_precisions = []
        average_precision_stats = PrettyTable(
            [
                "Class Name",
                "Total TP",
                "Total FP",
                "Total GroundTruths",
                "Total Detections",
                f"Average Precision (IoU={self.iou_threshold})",
            ]
        )

        mAP = (
            sum(average_precisions) / len(average_precisions)
            if len(average_precisions)
            else 0.0
        )

        if self.print_FP_files:
            FP_file_stats = PrettyTable(["Class Name", "FP File Path"])
            for result in results:
                class_name = result["class_name"]
                for file_path in result["FP_image_paths"]:
                    FP_file_stats.add_row([class_name, file_path])
            print(FP_file_stats)

        if self.print_detail_mAP:
            for result in results:
                average_precisions.append(result["average_precision"])
                average_precision_stats.add_row(
                    [
                        result["class_name"],
                        result["total_TP"],
                        result["total_FP"],
                        result["total_groundtruths"],
                        result["total_detections"],
                        round(result["average_precision"], 4),
                    ]
                )

            print(average_precision_stats)

            return mAP, results

        return mAP

    def every_points_interpolated_AP(
        self, recalls: List[float], precisions: List[float]
    ) -> Tuple[float, List[float], List[float]]:
        r"""every-point interpolated average precision"""

        mrecalls = [0.0] + recalls + [1.0]
        mprecisions = [0.0] + precisions + [0.0]

        for i in range(len(mprecisions) - 1, 0, -1):  # range(start, end=0, step=-1)
            mprecisions[i - 1] = max(mprecisions[i - 1], mprecisions[i])

        average_precision = 0.0
        for i in range(1, len(mrecalls)):  # range(start, end, step=1)
            if mrecalls[i] != mrecalls[i - 1]:
                average_precision += (mrecalls[i] - mrecalls[i - 1]) * mprecisions[i]

        return average_precision, mrecalls[0:-1], mprecisions[0:-1]

    def eleven_points_interpolated_AP(
        self, recalls: List[float], precisions: List[float]
    ) -> Tuple[float, List[float], List[float]]:
        r"""11-point interpolated average precision"""

        interp_rhos = []
        valid_recalls = []

        recall_values = np.linspace(start=0, stop=1, num=11).tolist()
        # for each recall_values (0, 0.1, 0.2, ... , 1)
        for r in recall_values[::-1]:
            # obtain all recall values higher or equal than r
            arg_greater_recalls = np.argwhere(np.array(recalls) >= r)

            precision_max = 0.0
            # if there are recalls above r
            if arg_greater_recalls.size != 0:
                precision_max = max(precisions[arg_greater_recalls.min() :])

            valid_recalls.append(r)
            interp_rhos.append(precision_max)

        # by definition AP = sum(max(precision whose recall is above r)) / 11
        average_precision = sum(interp_rhos) / 11

        # generating values for the plot
        mrecalls = [valid_recalls[0]] + valid_recalls + [0.0]
        mprecisions = [0.0] + interp_rhos + [0.0]

        pairs = []
        for i in range(len(mrecalls)):
            pair = (mrecalls[i], mprecisions[i - 1])
            if pair not in pairs:
                pairs.append(pair)
            pair = (mrecalls[i], mprecisions[i])
            if pair not in pairs:
                pairs.append(pair)

        mrecalls = [pair[0] for pair in pairs]
        mprecisions = [pair[1] for pair in pairs]

        return average_precision, mrecalls, mprecisions

    def _iou(self, boxA: List[float], boxB: List[float]) -> float:
        r"""Calculates intersection over union
        Args:
            boxA: List[box[x_min, y_min, x_max, y_max]]
            boxB: List[box[x_min, y_min, x_max, y_max]]
        Return:
            iou: float, intersection over union of boxA, boxB
        """
        iou = 0.0
        boxA = geometry.box(*[boxA[0], boxA[1], boxA[2] + 1, boxA[3] + 1])
        boxB = geometry.box(*[boxB[0], boxB[1], boxB[2] + 1, boxB[3] + 1])
        if boxA.intersects(boxB):
            iou = boxA.intersection(boxB).area / boxA.union(boxB).area

        return iou
