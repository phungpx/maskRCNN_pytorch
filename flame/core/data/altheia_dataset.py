import cv2
import torch
import random
import numpy as np
import imgaug.augmenters as iaa
import xml.etree.ElementTree as ET

from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Optional
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class AltheiaDataset(Dataset):
    def __init__(
        self,
        dirnames: List[str] = None,
        image_size: Tuple[int, int] = (800, 800),
        image_patterns: List[str] = ['*.jpg'],
        label_patterns: List[str] = ['*.xml'],
        classes: Dict[str, int] = None,
        mean: List[float] = [0., 0., 0.],
        std: List[float] = [1., 1., 1.],
        transforms: Optional[List] = None
    ) -> None:
        super(AltheiaDataset, self).__init__()
        self.classes = classes
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)
        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)

        self.transforms = transforms if transforms else []

        image_paths, label_paths = [], []
        for dirname in dirnames:
            for image_pattern in image_patterns:
                image_paths.extend(Path(dirname).glob('**/{}'.format(image_pattern)))
            for label_pattern in label_patterns:
                label_paths.extend(Path(dirname).glob('**/{}'.format(label_pattern)))

        image_paths = natsorted(image_paths, key=lambda x: str(x.stem))
        label_paths = natsorted(label_paths, key=lambda x: str(x.stem))

        self.data_pairs = [[image, label] for image, label in zip(image_paths, label_paths) if image.stem == label.stem]

        self.pad_to_square = iaa.PadToSquare(position='right-bottom', pad_cval=0)
        self.image_resizer = iaa.Resize(size=image_size, interpolation='cubic')
        self.mask_resizer = iaa.Resize(size=image_size, interpolation='nearest')

        print(f'{Path(dirnames[0]).stem}: {len(self.data_pairs)}')

    def __len__(self):
        return len(self.data_pairs)

    def to_4points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        x1, y1 = points[0][0], points[0][1]
        x2, y2 = points[1][0], points[1][1]
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def get_annotation(self, label_path: str, classes: dict) -> Dict:
        root = ET.parse(str(label_path)).getroot()
        page = root.find('{}Page'.format(''.join(root.tag.partition('}')[:2])))
        w, h = int(page.get('imageWidth')), int(page.get('imageHeight'))

        annots = []
        for card_type, label in classes.items():
            regions = root.findall('.//*[@value=\"{}\"]/../..'.format(card_type))
            regions += root.findall('.//*[@name=\"{}\"]/../..'.format(card_type))
            for region in regions:
                points = [
                    [int(float(coord)) for coord in point.split(',')]
                    for point in region[0].get('points').split()
                ]

                if len(points) == 2:
                    points = self.to_4points(points)
                elif len(points) <= 1:
                    continue

                x1 = min([point[0] for point in points])
                y1 = min([point[1] for point in points])
                x2 = max([point[0] for point in points])
                y2 = max([point[1] for point in points])

                if w >= x2 > x1 and h >= y2 > y1:
                    mask = np.zeros(shape=(h, w), dtype=np.uint8)
                    annots.append(
                        {
                            'mask': cv2.fillPoly(img=mask, pts=np.int32([points]), color=1),
                            'label': label,
                            'box': (x1, y1, x2, y2)
                        }
                    )

        if not len(annots):
            annots.append(
                {
                    'mask': np.zeros(shape=(h, w), dtype=np.uint8),
                    'label': -1,
                    'box': (0, 0, 1, 1)
                }
            )

        return annots

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict, Tuple[str, Tuple[int, int]]]:
        image_path, label_path = self.data_pairs[idx]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annots = self.get_annotation(label_path=str(label_path), classes=self.classes)
        boxes = [annot['box'] for annot in annots]
        labels = [annot['label'] for annot in annots]
        masks = [annot['mask'] for annot in annots]

        image_info = (str(image_path), image.shape[1::-1])  # image path, (w, h)

        # create BoundingBoxesOnImage
        bboxes = BoundingBoxesOnImage(
            bounding_boxes=[
                BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=label)
                for box, label in zip(boxes, labels)
            ],
            shape=image.shape
        )

        # create SegmentationMapsOnImage
        masks = [
            SegmentationMapsOnImage(arr=mask, shape=image.shape[:2])
            for mask in masks
        ]

        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            fixed_transform = transform.to_deterministic()  # fix random setting in each transforms
            image, bboxes = fixed_transform(image=image, bounding_boxes=bboxes)
            masks = [fixed_transform(segmentation_maps=mask) for mask in masks]

        masks = [mask.get_arr() for mask in masks]

        if self.image_size is not None:
            # Pad to square and then Rescale image, masks and bounding boxes
            image, bboxes = self.pad_to_square(image=image, bounding_boxes=bboxes)
            masks = [self.pad_to_square(image=mask) for mask in masks]
            image, bboxes = self.image_resizer(image=image, bounding_boxes=bboxes)
            masks = [self.mask_resizer(image=mask) for mask in masks]

        bboxes = bboxes.on(image)

        # Convert from Bouding Box Object to boxes, labels list
        labels = [bbox.label for bbox in bboxes.bounding_boxes]
        boxes = [[bbox.x1, bbox.y1, bbox.x2, bbox.y2] for bbox in bboxes.bounding_boxes]

        # Convert masks to tensor
        masks = [torch.from_numpy(mask) for mask in masks]
        masks = torch.stack(masks, dim=0).to(torch.uint8)

        # Convert to Torch Tensor
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)  # suppose all instances are not crowd
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Target
        target = {
            'image_id': image_id,
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'area': area,
            'iscrowd': iscrowd,
        }

        # Image
        sample = torch.from_numpy(np.ascontiguousarray(image))
        sample = sample.permute(2, 0, 1).contiguous()
        sample = (sample.float().div(255.) - self.mean) / self.std

        return sample, target, image_info
