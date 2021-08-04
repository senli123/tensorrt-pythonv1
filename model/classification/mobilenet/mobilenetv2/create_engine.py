import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import tensorrt as trt
from utils.get_trtwts import load_weights
from cailb import *
import argparse


def add_batch_norm_2d(network, weight_map, input, layer_name, eps):
    gamma = weight_map[layer_name + ".weight"]
    beta = weight_map[layer_name + ".bias"]
    mean = weight_map[layer_name + ".running_mean"]
    var = weight_map[layer_name + "running_var"]    
    var = np.sqrt(var + eps)

    scale = gamma / var
    shift = -mean / var * gamma + beta
    return network.add_scale(   input = input, 
                                mode = trt.ScaleMode.CHANNEL,
                                shift = shift,
                                scale = scale)  

def conv_bn_relu(network, weight_map, input, outch, ksize, s, g, lname):
    p = (ksize -1)//2
    conv1 =  network.add_convolution(input =input, 
                                    num_output_maps = outch,
                                    kernel_size = (ksize, ksize),
                                    kernel = weight_map[lname + "0.weight"],
                                    bias = trt.Weights())
    assert conv1
    conv1.stride = (s,s)
    conv1.padding = (p, p)
    conv1.num_groups = g
    bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + "1", EPS)
    assert bn1
    relu1 = network.add_activation(bn1.get_output(0),type = trt.ActivationType.RELU)
    assert relu1
     


