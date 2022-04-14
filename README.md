# MaskRCNN

# Documents
1. The best explanation, I've ever read.
```bash
https://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/
```

2. Pytorch Setting.
```bash
https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
```

3. RoI Align vs RoI Pooling
* RoI Pooling
```bash
https://erdem.pl/2020/02/understanding-region-of-interest-ro-i-pooling
```
* RoI Align
```bash
https://erdem.pl/2020/02/understanding-region-of-interest-part-2-ro-i-align
```

4. Non-maxmimum Suppression (NMS)
```bash
https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
```

5. mAP
```bash
https://github.com/rafaelpadilla/Object-Detection-Metrics
```

# Todo
- [x] [Support Labelme format](https://github.com/phungpx/maskRCNN_pytorch/blob/main/flame/core/data/labelme_dataset.py)
- [x] [Support Altheia format](https://github.com/phungpx/maskRCNN_pytorch/blob/main/flame/core/data/altheia_dataset.py)
- [x] [Support Pascal VOC format](https://github.com/phungpx/maskRCNN_pytorch/blob/main/flame/core/data/pascal_dataset.py)

# MaskRCNN
* [Faster RCNN](https://github.com/phungpx/maskRCNN_pytorch/blob/main/flame/core/model/maskRCNN/faster_rcnn.py)
```python3
import cv2
import torch 

from flame.core.model.maskRCNN.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from flame.core.model.maskRCNN.faster_rcnn import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
model.eval().to(device)

# sample
image = cv2.imread('/home/phungpx/Downloads/dog.jpg')
sample = torch.from_numpy(image).to(device)
sample = sample.float().div(255.)
sample = sample.permute(2, 0, 1).contiguous()

with torch.no_grad():
	preds = model([sample])
```
* [Mask RCNN](https://github.com/phungpx/maskRCNN_pytorch/blob/main/flame/core/model/maskRCNN/mask_rcnn.py)
```python3

```

# Usage
> Training
```bash
CUDA_VISIBLE_DEVICE=0,1,2,3 python -m flame configs/...
```

> Testing
```bash
CUDA_VISIBLE_DEVICE=0,1,2,3 python -m flame configs/...
```

# Indicate
1. Backbones

2. Anchor Generation
<div align="center">
	<img src="https://user-images.githubusercontent.com/61035926/163119122-f73dd80f-6a5c-498d-a049-584661f2ad63.png" width="700">
</div>

3. RPN (Region Proposal Network)
<div align="center">
	<img src="https://user-images.githubusercontent.com/61035926/163118898-bc59196b-e9fd-4c0d-b14b-e0f8d96c067f.png" width="700">
</div>

<div align="center">
	<img src="https://user-images.githubusercontent.com/61035926/163119041-364abdc8-fa6c-4ce3-ab91-307439db1db2.png" width="700">
</div>

4. RPN Loss

4. RoI Head (Region of Interest)

5. RoI Loss
