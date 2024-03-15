# MaskRCNN

# Todo
- [x] [Mask RCNN Resnet 50 FPN](https://github.com/phungpx/maskRCNN_pytorch/blob/main/flame/core/model/maskrcnn/maskrcnn_resnet50_fpn_v2.py)
- [x] [Mask RCNN MobilenetV3 Large FPN](https://github.com/phungpx/maskRCNN_pytorch/blob/main/flame/core/model/maskrcnn/maskrcnn_mobilenetv3_large_fpn.py)
- [x] [Mask RCNN MobilenetV2](https://github.com/phungpx/maskRCNN_pytorch/blob/main/flame/core/model/maskrcnn/maskrcnn_mobilenetv2.py)
- [x] [Faster RCNN Resnet 50 FPN](https://github.com/phungpx/maskRCNN_pytorch/blob/main/flame/core/model/maskrcnn/fasterrcnn_resnet50_fpn_v2.py)
- [x] [Faster RCNN MobilenetV3 FPN](https://github.com/phungpx/maskRCNN_pytorch/blob/main/flame/core/model/maskrcnn/fasterrcnn_mobilenetv3_fpn.py)
- [x] [mAP](https://github.com/phungpx/maskRCNN_pytorch/blob/main/flame/core/metric/mAP.py)
- [x] [COCO Evaluator](https://github.com/phungpx/maskRCNN_pytorch/blob/main/flame/core/metric/COCO_eval.py)
- [x] [Support Labelme format](https://github.com/phungpx/maskRCNN_pytorch/blob/main/flame/core/data/labelme_dataset.py)
- [x] [Support Altheia format](https://github.com/phungpx/maskRCNN_pytorch/blob/main/flame/core/data/altheia_dataset.py)
- [x] [Support Pascal VOC format](https://github.com/phungpx/maskRCNN_pytorch/blob/main/flame/core/data/pascal_dataset.py)

# Usage
> Training
```bash
CUDA_VISIBLE_DEVICE=0,1,2,3 python -m flame configs/...
```

> Testing
```bash
CUDA_VISIBLE_DEVICE=0,1,2,3 python -m flame configs/...
```
# Papers
1. [Fast RCNN](https://arxiv.org/pdf/1504.08083.pdf)
2. [Faster RCNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)
3. [Mask RCNN](https://arxiv.org/pdf/1703.06870.pdf)

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
