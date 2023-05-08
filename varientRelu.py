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
# class varientrelu(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, threshold1):
#         if x.requires_grad:
#             ctx.save_for_backward(x)
#
#         output = x.clone()
#         output[output > 0] += threshold1
#         output[output <= 0] = 0
#
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         if ctx.needs_input_grad[0]:
#             grad_input[ctx.saved_tensors[0] <= 0] = 0
#
#         return grad_input, None
#
#
# class VarientRelu(nn.Module):
#     def __init__(self, threshold1):
#         super(VarientRelu, self).__init__()
#
#         self.threshold1 = threshold1
#
#     def forward(self, x):
#         return varientrelu.apply(x, self.threshold1)
# 自定义一个类,对图片每个像素随机添加黑白噪声
class AddNoise(object):
    """
    Args:
        s(float): 噪声率
        p (float): 执行该操作的概率
    """

    def __init__(self, s=0.5, p=0.9):
        assert isinstance(s, float) or (isinstance(p, float))  # 判断输入参数格式是否正确
        self.s = s
        self.p = p

    # transform 会调用该方法
    def __call__(self, img):  # 使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用并执行对应的代码。
        """
       （PIL全称 Python Imaging Library，是 Python 平台一个功能非常强大而且简单易用的图像处理库。python3叫pillow）
       Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        # 如果随机概率小于 seld.p，则执行 transform
        if random.uniform(0, 1) < self.p:  # random.uniform(参数1，参数2) 返回参数1和参数2之间的任意值
            # 把 image 转为 array
            img_ = np.array(img).copy()
            # 获得 shape，高*宽*通道数
            h, w, c = img_.shape
            # 信噪比
            signal_pct = self.s
            # 噪声的比例 = 1 -信噪比
            noise_pct = (1 - self.s)
            # 选择的值为 (0, 1, 2)，每个取值的概率分别为 [signal_pct, noise_pct/2., noise_pct/2.]
            # 1 为白噪声，2 为 黑噪声
            #numpy.random.choice(a, size=None, replace=True, p=None)解释
            #从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
            #replace:True表示可以取相同数字，False表示不可以取相同数字
            #数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255  # 白噪声
            img_[mask == 2] = 0  # 黑噪声
            # 再转换为 image
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        # 如果随机概率大于 seld.p，则直接返回原图
        else:
            return img
class steprelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta, steps_num):
        if x.requires_grad:
            ctx.save_for_backward(x)
        ctx.theta = theta
        ctx.steps_num = steps_num

        output = x.clone()
        output[output <= 0] = 0
        for i in range(steps_num):
            index = (theta*i < output) & (output <= theta*(i+1))
            output[index] = i * theta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()

        if ctx.needs_input_grad[0]:
            grad_input[ctx.saved_tensors[0] <= 0] = 0

        return grad_input, None

class StepRelu(nn.Module):
    def __init__(self, theta, steps_num):
        super(StepRelu, self).__init__()

        self.theta = theta
        self.steps_num = steps_num

    def forward(self, x):
        return steprelu.apply(x, self.theta, self.steps_num)

def downsample(in_chan, out_chan):
    downsample1 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(out_chan, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            )
    return downsample1
class BasicBlock(nn.Module):
    def __init__(self, in_chan=256, out_chan=512,stride=1,downsample=None):
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False),
        # nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1=nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1=nn.BatchNorm2d(out_chan, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.relu=nn.ReLU(inplace=True)
        self.relu=StepRelu(theta=0.05,steps_num=1)
        self.conv2=nn.Conv2d(out_chan, out_chan, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2=nn.BatchNorm2d(out_chan, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.downsample=downsample



    def forward(self, x):
        identity=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        if self.downsample is not None:
            identity=self.downsample(x)
        out+=identity
        out=self.relu(out)
        return out

class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()

        self.conv1=nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1=nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.relu=nn.ReLU(inplace=True)
        self.relu = StepRelu(theta=0.05, steps_num=1)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1=nn.Sequential(
            BasicBlock(64,64),
            BasicBlock(64,64)
        )
        self.layer2=nn.Sequential(
            BasicBlock(64,128,2,downsample=downsample(64,128)),
            BasicBlock(128,128)
        )
        self.layer3=nn.Sequential(
            BasicBlock(128,256,2,downsample=downsample(128,256)),
            BasicBlock(256,256)
        )
        self.layer4=nn.Sequential(
            BasicBlock(256,512,2,downsample=downsample(256,512)),
            BasicBlock(512,512)
        )
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc=nn.Linear(in_features=512, out_features=1000, bias=True)
    def forward(self,x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.avgpool(out)
        out=out.reshape(out.shape[0],-1)
        out=self.fc(out)
        return out
# relu = StepRelu(0.5, 3)
# relu.train()
#
# input = torch.randn((2, 5), requires_grad=True)*2
# print(input)
#
# output = relu(input)
# print(output)
# net=models.resnet18(pretrained=False)
# print(net)
net=resnet18()
net.fc=nn.Linear(in_features=512,out_features=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trans_transform= transforms.Compose(
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
test1_transform = transforms.Compose(
        [transforms.Resize(256),  # 是按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。
         transforms.CenterCrop(224),  # 依据给定的size从中心裁剪
         AddNoise(0.95,0.5),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

root='D:\\finalproject\CancerDetection-master\dataset'
trainset = datasets.CIFAR10(root=root, train=True, download=True,transform=trans_transform)
testset=datasets.CIFAR10(root=root, train=False, download=True,transform=test_transform)
testset1=datasets.CIFAR10(root=root, train=False, download=True,transform=test1_transform)
trainloader=torch.utils.data.DataLoader(trainset, batch_size=64,shuffle=True, num_workers=0)
testloader=torch.utils.data.DataLoader(testset, batch_size=64,shuffle=False, num_workers=0)
testloader1=torch.utils.data.DataLoader(testset1, batch_size=64,shuffle=False, num_workers=0)
Net=net
# Net.load_state_dict(torch.load('D:\\finalproject\CancerDetection-master\\varientResNet18-205.pt'),strict=True)
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

torch.save(Net.state_dict(),'D:\\finalproject\CancerDetection-master\\varientResNet18-605.pt')
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

with torch.no_grad():
    correct=0
    total=0
    for data in testloader1:
        images,labels=data
        images,labels=images.to(device),labels.to(device)
        outputs=Net(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    print('Accuracy of the network on the 10000 test images with noise:{:.4f}'.format(100*correct/total))