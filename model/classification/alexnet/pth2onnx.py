import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision

def main():
    net = torchvision.models.alexnet(pretrained= True)
    net.eval()
    net = net.to('cuda:0')
    tmp = torch.ones(1,3,224,224).to('cuda:0')
    torch.onnx.export(net, tmp, './net/alexnet.onnx',opset_version=11)

if __name__ == '__main__':
    main()
