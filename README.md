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
- [x] Support Labelme format.
- [x] Support Altheia format.
- [x] Support Pascal VOC format.

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

# Usage
> Training
```bash
CUDA_VISIBLE_DEVICE=0,1,2,3 python -m flame configs/...
```

> Testing
```bash
CUDA_VISIBLE_DEVICE=0,1,2,3 python -m flame configs/...
```
