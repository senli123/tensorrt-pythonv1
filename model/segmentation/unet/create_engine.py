import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import tensorrt as trt
from utils.get_trtwts import load_weights
#from cailb import *
import argparse
from TensorRT.layer_common import *
EPS = 1e-5
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
def doubleConv(network, weight_map, input, outch, ksize, layer_name, midch):
    conv1 = network.add_convolution(input = input,
                                    num_output_maps = midch,
                                    kernel_shape = (ksize, ksize),
                                    kernel = weight_map[layer_name + ".double_conv.0.weight"],
                                    bias = weight_map[layer_name + ".double_conv.0.bias"])
    assert conv1
    conv1.stride = (1, 1)
    conv1.padding = (1, 1)

    bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), layer_name + ".double_conv.1", 0)
    assert bn1

    relu1 = network.add_activation(bn1.get_output(0), type = trt.ActivationType.kLEAKY_RELU)
    assert relu1
    
    conv2 = network.add_convolution(input = relu1.getOutput(0),
                                    num_output_maps = outch,
                                    kernel_shape = (3, 3),
                                    kernel = weight_map[layer_name + ".double_conv.3.weight"],
                                    bias = weight_map[layer_name + ".double_conv.3.bias"])

    assert conv2
    conv2.stride = (1, 1)
    conv2.padding = (1, 1)

    bn2 = add_batch_norm_2d(network, weight_map, conv2.get_output(0), layer_name + ".double_conv.4", 0)
    assert bn2
    
    relu2 = network.add_activation(bn2.get_output(0), type = trt.ActivationType.kLEAKY_RELU)
    assert relu2

    return relu2

def down(network, weight_map, input, outch, layer_name):
    pool1 = network.add_pooling(input = input, type = trt.PoolingType.MAX ,window_size=trt.DimsHW(2, 2))
    assert pool1
    dcov1 = doubleConv(network, weight_map, pool1.getOutput(0), outch, 3, layer_name +".maxpool_conv.1", outch)
    assert dcov1
    return dcov1

def add_batch_norm_2d(network, weight_map, input, layer_name):
    gamma = weight_map[layer_name + ".weight"]
    beta = weight_map[layer_name + ".bias"]
    mean = weight_map[layer_name + ".running_mean"]
    var = weight_map[layer_name + ".running_var"]
    var = np.sqrt(var + EPS)

    scale = gamma / var
    shift = -mean / var * gamma + beta
    return network.add_scale(input = input,
                             mode = trt.ScaleMode.CHANNEL,
                             shift = shift,
                             scale = scale)

def add_dense_layer(network, input, weight_map, lname):
    bn1 = add_batch_norm_2d(network, weight_map, input, lname + ".norm1")

    relu1 = network.add_activation(bn1.get_output(0), type = trt.ActivationType.RELU)
    assert relu1

    conv1 = network.add_convolution(input = relu1.get_output(0),
                                    num_output_maps = 128,
                                    kernel_shape = (1, 1),
                                    kernel = weight_map[lname + ".conv1.weight"],
                                    bias = trt.Weights())
    assert conv1
    conv1.stride = (1, 1)

    bn2 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + ".norm2")

    relu2 = network.add_activation(bn2.get_output(0), type = trt.ActivationType.RELU)
    assert relu2

    conv2 = network.add_convolution(input = relu2.get_output(0),
                                    num_output_maps = 32,
                                    kernel_shape = (3, 3),
                                    kernel = weight_map[lname + ".conv2.weight"],
                                    bias = trt.Weights())
    assert conv2
    conv2.stride = (1, 1)
    conv2.padding = (1, 1)

    return conv2