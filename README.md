# [ICCV 2023] GAFlow: Incorporating Gaussian Attention into Optical Flow

<h4 align="center">Ao Luo<sup>1</sup>, Fan Fang<sup>2</sup>, Xin Li<sup>2</sup>, Lang Nie<sup>3</sup>, Chunyu Lin<sup>3</sup>, Haoqiang Fan<sup>1</sup>, and Shuaicheng Liu<sup>4,1</sup></h4>
<h4 align="center">1. Megvii Research &emsp; 2. Group 42 &emsp; 3.Beijing Jiaotong University</h4>
<h4 align="center">4. University of Electronic Science and Technology of China</h4>

This project provides the official implementation of '[**GAFlow: Incorporating Gaussian Attention into Optical Flow**]()'.

## Abstract
Optical flow, or the estimation of motion fields from image sequences, is one of the fundamental problems in computer vision. Unlike most pixel-wise tasks that aim at achieving consistent representations of the same category, optical flow raises extra demands for obtaining local discrimination and smoothness, which yet is not fully explored by existing approaches. In this paper, we push Gaussian Attention (GA) into the optical flow models to accentuate local properties during representation learning and enforce the motion affinity during matching. Specifically, we introduce a novel Gaussian-Constrained Layer (GCL) which can be easily plugged into existing Transformer blocks to highlight the local neighborhood that contains fine-grained structural information. Moreover, for reliable motion analysis, we provide a new Gaussian-Guided Attention Module (GGAM) which not only inherits properties from Gaussian distribution to instinctively revolve around the neighbor fields of each point but also is empowered to put the emphasis on contextually related regions during matching. Our fully-equipped model, namely Gaussian Attention Flow network (GAFlow), naturally incorporates a series of novel Gaussian-based modules into the conventional optical flow framework for reliable motion analysis. Extensive experiments on standard optical flow datasets consistently demonstrate the exceptional performance of the proposed approach in terms of both generalization ability evaluation and online benchmark testing. 


## Overview

Our GAFlow contains two pivotal modules: i) Gaussian-Constrained Layer (GCL) to enhance the local information. It help capture more discriminative features for optical flow estimation.
ii) Gaussian-Guided Attention Module, which is flexible to refine motion features by learning the deformable Gaussian attention.

![fig_2](https://github.com/LA30/GAFlow/assets/47421121/0fdb8ca3-3ad2-48ca-8e8f-742eec402670)

## Requirements

Python 3.8 with following packages
```Shell
pytorch  1.11.0+cu102
torchvision  0.12.0
numpy  1.23.5
matplotlib  3.7.1
opencv-python  4.7.0.72
einops  0.6.1
easydict  1.10 
scipy  1.10.1
timm  0.9.2
tensorboard  2.13.0
cupy-cuda100  9.6.0

pip3 install natten -f https://shi-labs.com/natten/wheels/cu102/torch1.11/index.html
```
(The code has been tested on Cuda 12.0.)


## Usage

1. The trained weights are available on [GoogleDrive](https://drive.google.com/drive/folders/1_g3HWm6asi8CXEQrhJE2kv_PIXGbOAkv?usp=sharing). Put `*.pth` files into folder `./weights`.

2. Download [Sintel](http://sintel.is.tue.mpg.de/) and [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) dataset, and set the root path of each class in `./core/datasets.py`.


3. Evaluation on Sintel and KITTI
```Shell
./eval_sintel_kitti.sh
```

## Acknowledgement

The code is built based on [RAFT](https://github.com/princeton-vl/RAFT), [GMFlowNet](https://github.com/xiaofeng94/GMFlowNet) and [SKFlow](https://github.com/littlespray/SKFlow). We thank the authors for their contributions.
