import cv2
import torch
import random
import numpy as np
import imgaug.augmenters as iaa
import xml.etree.ElementTree as ET

from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class VOCDataset(Dataset):
    def __init__(self, input_dir, image_pattern, label_pattern, classes, image_size, transforms=None):
        super(VOCDataset, self).__init__()
        self.classes = classes
        self.image_size = image_size
        self.pad_to_square = iaa.PadToSquare(position='right-bottom')
        self.transforms = transforms if transforms else []
        image_paths = natsorted(list(Path(input_dir).glob(f'{image_pattern}')), key=lambda x: str(x.stem))
        label_paths = natsorted(list(Path(input_dir).glob(f'{label_pattern}')), key=lambda x: str(x.stem))
        self.data_pairs = [[image, label] for image, label in zip(image_paths, label_paths)]
        print(f'{Path(input_dir).stem}: {len(self.data_pairs)}')

    def __len__(self):
        return len(self.data_pairs)

    def _get_label_info(self, label_path):
        tree = ET.parse(str(label_path))
        image_info = {'image_name': tree.find('filename').text,
                      'height': int(tree.find('size').find('height').text),
                      'width': int(tree.find('size').find('width').text),
                      'depth': int(tree.find('size').find('depth').text)}
        label_info = []
        objects = tree.findall('object')
        for obj in objects:
            bndbox = obj.find('bndbox')
            bbox = np.int32([bndbox.find('xmin').text,
                             bndbox.find('ymin').text,
                             bndbox.find('xmax').text,
                             bndbox.find('ymax').text])
            label_name = obj.find('name').text
            label_info.append({'label': label_name, 'bbox': bbox})

        return image_info, label_info

    def __getitem__(self, idx):
        image_path, label_path = self.data_pairs[idx]
        sample_info, label_info = self._get_label_info(label_path)
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = [label['bbox'] for label in label_info]
        labels = [self.classes[label['label']] for label in label_info]
        image_info = [str(image_path), image.shape[1::-1]]

        # Pad to square to keep object's ratio
        bbs = BoundingBoxesOnImage([BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=label)
                                    for box, label in zip(boxes, labels)], shape=image.shape)
        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, bbs = transform(image=image, bounding_boxes=bbs)

        # Rescale image and bounding boxes
        image, bbs = self.pad_to_square(image=image, bounding_boxes=bbs)
        sample, bbs = iaa.Resize(size=self.image_size)(image=image, bounding_boxes=bbs)
        bbs = bbs.on(sample)

        # Convert from Bouding Box Object to boxes, labels list
        boxes = [[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs.bounding_boxes]
        labels = [bb.label for bb in bbs.bounding_boxes]

        # Convert to Torch Tensor
        labels = torch.from_numpy(np.asarray(labels)).to(torch.int64)
        boxes = torch.from_numpy(np.asarray(boxes)).to(torch.float32)
        image_id = torch.tensor([idx])

        # # Target
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id

        # Image
        sample = torch.from_numpy(np.ascontiguousarray(sample))
        sample = sample.permute(2, 0, 1).contiguous()
        sample = sample.float().div(255.)

        return sample, target, image_info


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    label_dir = './VOC2007/Annotations/'
    image_dir = './VOC2007/JPEGImages/'
    image_pattern = '*.jpg'
    label_pattern = '*.xml'
    classes2idx = {'aeroplane': 0,
                   'bicycle': 1,
                   'bird': 2,
                   'boat': 3,
                   'bottle': 4,
                   'bus': 5,
                   'car': 6,
                   'cat': 7,
                   'chair': 8,
                   'cow': 9,
                   'diningtable': 10,
                   'dog': 11,
                   'horse': 12,
                   'motorbike': 13,
                   'person': 14,
                   'pottedplant': 15,
                   'sheep': 16,
                   'sofa': 17,
                   'train': 18,
                   'tvmonitor': 19}
    transforms = []
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    compound_coef = 0  # B0
    image_size = 512 + compound_coef * 128
    voc_set = VOCDataset(image_dir=image_dir,
                         label_dir=label_dir,
                         image_pattern=image_pattern,
                         label_pattern=label_pattern,
                         classes=classes2idx,
                         image_size=image_size,
                         transforms=transforms)

    print(f'number of dataset: {len(voc_set)}')

    voc_loader = DataLoader(voc_set, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).unsqueeze(dim=1).unsqueeze(dim=1)  # (3, 1, 1)
    std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).unsqueeze(dim=1).unsqueeze(dim=1)  # (3, 1, 1)
    idx2class = {idx: label_name for label_name, idx in classes2idx.items()}
    for i, voc in enumerate(iter(voc_loader)):
        samples, targets, sample_infos = voc
        for sample, target in zip(samples, targets):
            image = ((sample * std + mean) * 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            boxes = target['boxes'].data.cpu().numpy().astype(np.int32)
            labels = target['labels'].data.cpu().numpy().astype(np.int32)
            for box, label in zip(boxes, labels):
                if label != -1:
                    image = np.ascontiguousarray(image)
                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 255, 0), thickness=1)
                    cv2.putText(image,
                                text=idx2class[int(label)],
                                org=tuple(box[:2]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.001 * max(image.shape[0], image.shape[1]),
                                color=(0, 0, 255),
                                thickness=1,
                                lineType=cv2.LINE_AA)
            cv2.imshow('image', image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        if i == 5:
            break
