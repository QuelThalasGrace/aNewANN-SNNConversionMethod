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
         #AddNoise(0.95,0.5),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
net=resnet18()
net.fc=nn.Linear(in_features=512,out_features=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root='.\dataset'

trainset = datasets.CIFAR10(root=root, train=True, download=True,transform=train_transform)
testset=datasets.CIFAR10(root=root, train=False, download=True,transform=test_transform)

trainloader=torch.utils.data.DataLoader(trainset, batch_size=64,shuffle=True, num_workers=0)
testloader=torch.utils.data.DataLoader(testset, batch_size=64,shuffle=False, num_workers=0)

Net=net
# Net.load_state_dict(torch.load('./varientResNet18-205.pt'),strict=True)
Net=Net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Net.parameters(), lr=0.001)
for epoch in range(200):
    for i,(inputs,labels) in enumerate(trainloader):
        inputs=inputs.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=Net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        if i%100==0:
            print('Epoch: {} Minibatch: {} loss: {:.3f}' .format(epoch + 1, i + 1, loss.item()))

torch.save(Net.state_dict(),'./varientResNet18-605.pt')
with torch.no_grad():
    correct=0
    total=0
    for data in testloader:
        images,labels=data
        images,labels=images.to(device),labels.to(device)
        outputs=Net(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    print('Accuracy of the network on the 10000 test images:{:.4f}'.format(100*correct/total))

