import cv2
import torch
import torchvision
import numpy as np

from pathlib import Path
from ..module import Module
from ignite.engine import Events
from typing import Dict, Tuple, List, Optional


class MaskPredictor(Module):
    def __init__(
        self,
        alpha: float = 0.3,
        image_size: Optional[Tuple[int, int]] = (800, 800),  # w, h
        evaluator_name: str = None,
        classes: Dict[str, List] = None,
        score_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        use_pad_to_square: bool = False,
        output_dir: str = None,
        binary_threshold: float = 0.5,
        output_transform=lambda x: x
    ) -> None:
        super(MaskPredictor, self).__init__()
        self.alpha = alpha
        self.classes = classes
        self.image_size = image_size
        self.iou_threshold = iou_threshold
        self.evaluator_name = evaluator_name
        self.score_threshold = score_threshold
        self.binary_threshold = binary_threshold
        self.output_transform = output_transform
        self.use_pad_to_square = use_pad_to_square
        self.output_dir = Path(output_dir)

    def init(self):
        assert self.evaluator_name in self.frame, f'The frame does not have {self.evaluator_name}'
        self._attach(self.frame[self.evaluator_name].engine)

    def reset(self):
        pass

    def update(self, output):
        preds, image_infos = output

        image_paths = [image_info[0] for image_info in image_infos]
        original_sizes = [image_info[1] for image_info in image_infos]

        for pred, image_path, original_size in zip(preds, image_paths, original_sizes):
            save_dir = self.output_dir.joinpath(Path(image_path).parent.stem)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

            save_path = str(save_dir.joinpath(Path(image_path).name))

            image = cv2.imread(image_path)

            labels, boxes = pred['labels'], pred['boxes']
            scores, masks = pred['scores'], pred['masks']

            if self.score_threshold:
                indices = scores > self.score_threshold
                labels, boxes = labels[indices], boxes[indices]
                scores, masks = scores[indices], masks[indices]

            if self.iou_threshold:
                indices = torchvision.ops.nms(boxes, scores, self.iou_threshold)
                labels, boxes = labels[indices], boxes[indices]
                scores, masks = scores[indices], masks[indices]

            boxes = boxes.data.cpu().numpy().tolist()
            labels = labels.data.cpu().numpy().tolist()
            scores = scores.data.cpu().numpy().tolist()
            masks = (masks > self.binary_threshold).to(torch.float32)
            masks = masks.squeeze(dim=1).cpu().numpy()

            if self.classes:
                classes = {
                    label: [cls_name, color] for cls_name, (color, label) in self.classes.items()
                }

            font_scale = max(original_size) / 1200
            box_thickness = max(original_size) // 400
            text_thickness = max(original_size) // 600

            if self.use_pad_to_square:
                original_size = (max(original_size), max(original_size))

            fx, fy = 1, 1
            if self.image_size is not None:
                fx, fy = original_size[0] / self.image_size[0], original_size[1] / self.image_size[1]

            for (label, box, score, mask) in zip(labels, boxes, scores, masks):
                if label == 0:
                    continue

                color = classes[label][1] if self.classes else [0, 0, 255]
                class_name = classes[label][0] if self.classes else str(label)
                x1, y1, x2, y2 = np.int32([box[0] * fx, box[1] * fy, box[2] * fx, box[3] * fy])

                cv2.rectangle(
                    img=image, pt1=(x1, y1), pt2=(x2, y2),
                    color=color, thickness=box_thickness
                )

                title = f"{class_name}: {score:.4f}"
                w_text, h_text = cv2.getTextSize(
                    title, cv2.FONT_HERSHEY_PLAIN, font_scale, text_thickness
                )[0]

                cv2.rectangle(
                    img=image, pt1=(x1, y1 + int(1.6 * h_text)), pt2=(x1 + w_text, y1),
                    color=color, thickness=-1
                )

                cv2.putText(
                    img=image, text=title, org=(x1, y1 + int(1.3 * h_text)),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=font_scale,
                    color=(255, 255, 255), thickness=text_thickness, lineType=cv2.LINE_AA
                )

                if self.image_size is not None:
                    mask = cv2.resize(mask, dsize=original_size, interpolation=cv2.INTER_NEAREST)
                    mask = mask[:image.shape[0], :image.shape[1]]

                image[mask == 1] = (
                    image[mask == 1] * (1. - self.alpha) + np.array(color, dtype=np.float) * self.alpha
                ).astype(np.uint8)

            cv2.imwrite(save_path, image)

    def compute(self):
        pass

    def started(self, engine):
        self.reset()

    @torch.no_grad()
    def iteration_completed(self, engine):
        output = self.output_transform(engine.state.output)
        self.update(output)

    def completed(self, engine):
        self.compute()

    def _attach(self, engine):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed)
        if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)