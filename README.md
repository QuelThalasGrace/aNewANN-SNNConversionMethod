# aNewANN-SNNConversionMethod
This is a project for our paper A New ANN-SNN Conversion Method with High Accuracy, Low Latency and Good Robustness in IJCAI2023

### Tips:
1. Since it involves another article being submitted, we only provide the code of the ANN part for the time being, and we will provide the code for the conversion part later.
2. The trained parameter file is too large to be uploaded. If any readers need to reproduce our work, please contact us through the email in the paper.

### About the training process
Our idea is mainly divided into two steps, the first step is to train the corresponding ANNs based on the StepReLU activation function, and the second step is to load the trained parameter files to the SNNs. For details of the StepReLU activation function, you can refer to the relevant code in the ann_StepReLU.py file.

You can also experiment on other network structures, but we only provide the training process of the resnet18 network in the train.py file. When you do this, you just need to define the network structure in the usual way, then replace all ReLU functions in the network with StepReLU functions, and specify the corresponding parameters. The util.py file mainly contains image enhancement methods for different data sets and methods for adding noise. When you use it, you can call it directly in the transform method like this:
~~~python
import torch
import torch.nn as nn
from torch.autograd import gradcheck, Variable
import torchvision
from torchvision import models,datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import random
from PIL import Image
from ann_StepReLU import *

        
train_transform= transforms.Compose(
         [transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),  # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪
         # 得到的图像为制定的大小； （即先随机采集，然后对裁剪得到的图像缩放为同一大小） 默认scale=(0.8, 1.0)
         transforms.RandomRotation(degrees=15),  # 随机旋转函数-依degrees随机旋转一定角度
         transforms.ColorJitter(),  # 改变颜色的，随机从-0.5 0.5之间对颜色变化
         transforms.RandomResizedCrop(224),  # 随机长宽比裁剪
         transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
         #AddNoise(0.95,0.5),
         transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
         # 归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。
         transforms.Normalize(mean=[0.485, 0.456, 0.406],   # 对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc
                              std=[0.229, 0.224, 0.225])])
test_transform = transforms.Compose(
        [transforms.Resize(256),  # 是按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。
         transforms.CenterCrop(224),  # 依据给定的size从中心裁剪
         AddNoise(0.95,0.5),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])


Taking care of the above, you can train ANNs based on the SteReLU activation function that perform well.
