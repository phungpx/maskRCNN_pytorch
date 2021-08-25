import cv2
import torch
import random
import numpy as np

from pathlib import Path
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class CoCoDataset(Dataset):
    def __init__(self, data_dir, set_name, mean, std, min_size, max_size, transforms=None):
        super(CoCoDataset, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.transforms = transforms if transforms else []
        self.std = torch.tensor(std, dtype=torch.float).reshape(1, 1, 3)
        self.mean = torch.tensor(mean, dtype=torch.float).reshape(1, 1, 3)

        self.image_dir = Path(data_dir).joinpath(set_name)
        self.coco = COCO(annotation_file=str(Path(data_dir).joinpath('annotations', f'instances_{set_name}.json')))
        self.image_indices = self.coco.getImgIds()

        self.class2idx = dict()
        self.coco_label_to_label = dict()
        self.label_to_coco_label = dict()
        categories = self.coco.loadCats(ids=self.coco.getCatIds())
        categories = sorted(categories, key=lambda x: x['id'])
        for category in categories:
            self.label_to_coco_label[len(self.class2idx)] = category['id']
            self.coco_label_to_label[category['id']] = len(self.class2idx)
            self.class2idx[category['name']] = len(self.class2idx)

        self.idx2class = {class_idx: class_name for class_name, class_idx in self.class2idx.items()}

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx):
        image, image_info = self._load_image(image_idx=idx)
        boxes, labels = self._load_annotation(image_idx=idx)
        if not len(boxes) and not len(labels):
            print(image_info[0])
            # raise

        image, boxes, scale = self._resize(image, boxes)

        bboxes = [BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=label)
                  for box, label in zip(boxes, labels)]
        bboxes = BoundingBoxesOnImage(bounding_boxes=bboxes, shape=image.shape)
        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, bboxes = transform(image=image, bounding_boxes=bboxes)

        bboxes = bboxes.on(image)

        boxes = [[bbox.x1, bbox.y1, bbox.x2, bbox.y2] for bbox in bboxes.bounding_boxes]
        labels = [bbox.label for bbox in bboxes.bounding_boxes]

        # Convert to Torch Tensor
        labels = torch.from_numpy(np.asarray(labels)).to(torch.int64)
        boxes = torch.from_numpy(np.asarray(boxes)).to(torch.float32)
        image_id = torch.tensor([idx])

        # Target
        target = {}
        target['boxes'] = boxes
        target['scale'] = scale
        target['labels'] = labels
        target['image_id'] = image_id

        # Sample
        sample = np.ascontiguousarray(image)
        sample = torch.from_numpy(sample)
        sample = (sample.float().div(255.) - self.mean) / self.std
        sample = sample.permute(2, 0, 1).contiguous()

        return sample, target, image_info

    def _load_image(self, image_idx):
        image_info = self.coco.loadImgs(ids=self.image_indices[image_idx])[0]
        image_path = str(self.image_dir.joinpath(image_info['file_name']))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image_info = [image_path, image.shape[1::-1]]

        return image, image_info

    def _load_annotation(self, image_idx):
        boxes, labels = [], []
        annot_indices = self.coco.getAnnIds(imgIds=self.image_indices[image_idx], iscrowd=False)
        if not len(annot_indices):
            return boxes, labels

        annot_infos = self.coco.loadAnns(ids=annot_indices)
        for idx, annot_info in enumerate(annot_infos):
            # some annotations have basically no width or height, skip them.
            if annot_info['bbox'][2] < 1 or annot_info['bbox'][3] < 1:
                continue

            bbox = self._xywh2xyxy(annot_info['bbox'])
            label = self.coco_label_to_label[annot_info['category_id']]
            boxes.append(bbox)
            labels.append(label)

        return boxes, labels

    def _xywh2xyxy(self, box):
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        return box

    def _image_aspect_ratio(self, image_index):
        image_info = self.coco.loadImgs(self.image_indices[image_index])[0]
        return float(image_info['width']) / float(image_info['height'])

    def _num_classes(self):
        return len(list(self.idx2class.keys()))

    def _resize(self, image, bboxes):
        height, width = image.shape
        min_dim, max_dim = min(image.shape), max(image.shape)

        scale = self.min_size / min_dim
        if max_dim * scale > self.max_size:
            scale = self.max_size / max_dim

        scaled_width = int(round(scale * width))
        scaled_height = int(round(scale * height))
        scaled_image = cv2.resize(image, dsize=(scaled_width, scaled_height))

        padded_width = scaled_width + (32 - scaled_width % 32)
        padded_height = scaled_height + (32 - scaled_height % 32)
        padded_image = np.zeros(shape=(padded_height, padded_width, image.shape[2]), dtype=image.dtype)
        padded_image[:scaled_height, :scaled_width, :] = scaled_image

        bboxes = [(np.array(bbox) * scale).tolist() for bbox in bboxes]

        return image, bboxes, scale
