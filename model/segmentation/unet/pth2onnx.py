import os
import torch
from torch import nn as nn
from torch.nn import functional as F
from unet_model import UNet
def main():
    net = UNet(n_channels=3, n_classes=1)
    net = net.to('cuda:0')
    net.load_state_dict(torch.load(os.getcwd()+'/net/pth/unet_carvana_scale1_epoch5.pth', map_location='cuda:0'))
    net.eval()
    tmp = torch.ones(1,3,224,224).to('cuda:0')
    torch.onnx.export(net, tmp, './net/onnx/unet.onnx',opset_version=11)

if __name__ == '__main__':
    main()
