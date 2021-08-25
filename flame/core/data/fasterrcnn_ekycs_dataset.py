import cv2
import torch
import random
import numpy as np
import imgaug.augmenters as iaa
import xml.etree.ElementTree as ET

from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class EkycsDataset(Dataset):
    def __init__(self, dirname, classes, image_size, image_patterns, label_patterns, transforms=None):
        super(EkycsDataset, self).__init__()
        self.classes = classes
        self.image_size = image_size
        self.transforms = transforms if transforms else []
        self.pad_to_square = iaa.PadToSquare(position='right-bottom')

        image_paths, label_paths = [], []
        for image_pattern in image_patterns:
            image_paths.extend(Path(dirname).glob('**/{}'.format(image_pattern)))
        for label_pattern in label_patterns:
            label_paths.extend(Path(dirname).glob('**/{}'.format(label_pattern)))

        image_paths = natsorted(image_paths, key=lambda x: str(x.stem))
        label_paths = natsorted(label_paths, key=lambda x: str(x.stem))

        if len(image_paths) != len(label_paths):
            raise ValueError('images {} and masks {} must be the same length.'.format(len(image_paths), len(label_paths)))

        self.data_pairs = [[image_path, label_path] for image_path, label_path in zip(image_paths, label_paths)]

        print(f'{Path(dirname).stem}: {len(self.data_pairs)}')

    def __len__(self):
        return len(self.data_pairs)

    def _get_masks(self, xml_path, classes):
        root = ET.parse(str(xml_path)).getroot()
        namespace = ''.join(root.tag.partition('}')[:2])
        page = root.find('{}Page'.format(namespace))
        width = int(page.get('imageWidth'))
        height = int(page.get('imageHeight'))

        masks, labels = [], []
        for card_type, label in classes.items():
            regions = root.findall('.//*[@name=\"{}\"]/../..'.format(card_type))
            for region in regions:
                points = [[int(float(coord)) for coord in point.split(',')] for point in region[0].get('points').split()]
                assert len(points) >= 4, 'Length of points must be greater than or equal 4.'
                mask = np.zeros(shape=(height, width), dtype=np.uint8)
                cv2.fillPoly(img=mask, pts=np.int32([points]), color=(255, 255, 255))
                masks.append(mask)
                labels.append(label)

        if len(masks) and len(labels):
            masks, labels = np.stack(masks, axis=-1), np.asarray(labels, dtype=np.int64)

        return masks, labels

    def __getitem__(self, idx):
        image_path, label_path = self.data_pairs[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks, labels = self._get_masks(xml_path=str(label_path), classes=self.classes)
        if (not len(masks)) and (not len(labels)):
            raise ValueError('image {} has no label.'.format(image_path.stem))

        image_info = [str(image_path), image.shape[1::-1]]

        # create SegmentationMapsOnImage
        masks = [SegmentationMapsOnImage(masks[:, :, i], image.shape[:2]) for i in range(masks.shape[-1])]

        # transform masks and image
        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            _transform = transform.to_deterministic()
            image = _transform(image=image)
            masks = [_transform(segmentation_maps=mask) for mask in masks]
        masks = [mask.get_arr() for mask in masks]

        # padding image, masks to square and then resize image and masks
        image = cv2.resize(self.pad_to_square(image=image), dsize=self.image_size)
        masks = [cv2.resize(self.pad_to_square(image=mask), dsize=self.image_size) for mask in masks]
        masks = np.stack(masks, axis=-1)

        # get boxes in masks
        boxes = []
        for i in range(masks.shape[-1]):
            mask = masks[:, :, i]
            pos = np.where(mask == 255)
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert to Torch Tensor
        labels = torch.from_numpy(labels).to(torch.int64)
        boxes = torch.from_numpy(np.asarray(boxes)).to(torch.float32)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros(len(labels)).to(torch.int64)
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # mask
        masks = torch.from_numpy(masks)
        masks = torch.floor_divide(masks, 255)
        masks = masks.permute(2, 0, 1).to(torch.uint8)

        # target
        target = {'boxes': boxes, 'labels': labels, 'area': areas, 'iscrowd': iscrowd, 'image_id': image_id}

        # image
        sample = torch.from_numpy(np.ascontiguousarray(image))
        sample = sample.permute(2, 0, 1).contiguous()
        sample = sample.float().div(255.)

        return sample, target, image_info
