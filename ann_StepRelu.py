import torch
import torch.nn as nn
from torch.autograd import gradcheck, Variable


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
        self.relu=StepRelu(theta=0.05,steps_num=20)
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
        self.relu = StepRelu(theta=0.05, steps_num=20)
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
