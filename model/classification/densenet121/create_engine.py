import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import tensorrt as trt
from utils.get_trtwts import load_weights
from cailb import *
import argparse

EPS = 1e-5
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
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

def add_transition(network, input, weight_map, outch, lname):
    bn1 = add_batch_norm_2d(network, weight_map, input, lname + ".norm")

    relu1 = network.add_activation(bn1.get_output(0), type = trt.ActivationType.RELU)
    assert relu1

    conv1 = network.add_convolution(input = relu1.get_output(0),
                                    num_output_maps = outch,
                                    kernel_shape = (1, 1), 
                                    kernel = weight_map[lname + ".conv.weight"], 
                                    bias = trt.Weights())
    assert conv1
    conv1.stride = (1, 1)

    pool1 = network.add_pooling(input = conv1.get_output(0),
                                type = trt.PoolingType.AVERAGE,
                                window_size = trt.DimsHW(2, 2))
    assert pool1
    pool1.stride_nd = (2, 2)
    pool1.padding_nd = (0, 0)

    return pool1

def add_dense_block(network, input, weight_map, num_dense_layers, lname):
    input_tensors = [None for _ in range(num_dense_layers + 1)]
    input_tensors[0] = input
    c = add_dense_layer(network, input, weight_map, lname + ".denselayer" + str(1))
    for i in range(1, num_dense_layers):
        input_tensors[i] = c.get_output(0)
        concat = network.add_concatenation(input_tensors[: i+1])
        assert concat
        c = add_dense_layer(network, concat.get_output(0), weight_map, lname + ".denselayer" + str(i+1))
    
    input_tensors[num_dense_layers] = c.get_output(0)
    concat = network.add_concatenation(input_tensors)
    assert concat

    return concat


def create_engine(builder, config, dt, INPUT_BLOB_NAME, OUTPUT_BLOB_NAME, max_batch_size, INPUT_H, INPUT_W, OUTPUT_SIZE, WEIGHT_PATH, DATA_TYPE, calib):
    weight_map = load_weights(WEIGHT_PATH)
    network = builder.create_network()

    data = network.add_input(INPUT_BLOB_NAME, dt, (3, INPUT_H, INPUT_W))
    assert data

    conv0 = network.add_convolution(input = data,
                                    num_output_maps = 64,
                                    kernel_shape = (7, 7),
                                    kernel = weight_map["features.conv0.weight"],
                                    bias = trt.Weights())
    assert conv0
    conv0.stride = (2, 2)
    conv0.padding = (3, 3)

    bn0 = add_batch_norm_2d(network, weight_map, conv0.get_output(0), "features.norm0")

    relu0 = network.add_activation(bn0.get_output(0), type = trt.ActivationType.RELU)
    assert relu0

    pool0 = network.add_pooling(input = relu0.get_output(0), 
                                type = trt.PoolingType.MAX,
                                window_size =trt.DimsHW(3, 3))
    assert pool0
    pool0.stride_nd = (2, 2)
    pool0.padding_nd = (1, 1)

    dense1 = add_dense_block(network, pool0.get_output(0), weight_map, 6, "features.denseblock1")
    transition1 = add_transition(network, dense1.get_output(0), weight_map, 128, "features.transition1") 

    dense2 = add_dense_block(network, transition1.get_output(0), weight_map, 12, "features.denseblock2")
    transition2 = add_transition(network, dense2.get_output(0), weight_map, 256, "features.transition2") 

    dense3 = add_dense_block(network, transition2.get_output(0), weight_map, 24, "features.denseblock3")
    transition3 = add_transition(network, dense3.get_output(0), weight_map, 512, "features.transition3") 

    dense4 = add_dense_block(network, transition3.get_output(0), weight_map, 16, "features.denseblock4")
    
    bn5 = add_batch_norm_2d(network, weight_map, dense4.get_output(0), "features.norm5")
    relu5 = network.add_activation(bn5.get_output(0), type = trt.ActivationType.RELU)

    pool5 = network.add_pooling(relu5.get_output(0), type=trt.PoolingType.AVERAGE, window_size=trt.DimsHW(7, 7))

    fc1 = network.add_fully_connected(input=pool5.get_output(0),
                                    num_outputs=OUTPUT_SIZE,
                                    kernel=weight_map["classifier.weight"],
                                    bias=weight_map["classifier.bias"])
    assert fc1
    fc1.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(fc1.get_output(0))

    # Build Engine
    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = 1 << 20
    if DATA_TYPE == "fp16":
        builder.fp16_mode = True
    elif DATA_TYPE == "int8":
        builder.int8_mode = True
        builder.int8_calibrator = calib
    engine = builder.build_engine(network, config)
    #or  engine = builder.build_cuda_engine(network)
    del network
    del weight_map
    return engine



def API_to_model(INPUT_BLOB_NAME, OUTPUT_BLOB_NAME, BATCH_SIZE, INPUT_H, INPUT_W, OUTPUT_SIZE, 
                WEIGHT_PATH, ENGINE_PATH, DATA_TYPE, FILE_PATH, DATA_SET, LABEL_SET, CACHE_FILE, MEAN, STD):
    loader.process_data( FILE_PATH, DATA_SET, LABEL_SET, INPUT_H, MEAN, STD)
    dataset = loader()
    cailb = DenseNetEntropyCailbrator.get_data(dataset.load_data(),CACHE_FILE,BATCH_SIZE)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    with create_engine(builder, config, trt.float32, INPUT_BLOB_NAME, OUTPUT_BLOB_NAME, 
                        BATCH_SIZE, INPUT_H, INPUT_W, OUTPUT_SIZE, WEIGHT_PATH,DATA_TYPE, 
                        cailb) as engine, engine.create_execution_context() as context:
        check_accuracy(context, BATCH_SIZE, test_set = dataset.load_data(), test_labels=dataset.load_labels(), output_size=OUTPUT_SIZE)
        with open(ENGINE_PATH, "wb") as f:
            f.write(engine.serialize())
        del engine
        del builder
        del config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b","--batch_size", type = int, default = 1)
    parser.add_argument("-h","--height", type = int, default = 224)
    parser.add_argument("-w","--width", type = int, default = 224)
    parser.add_argument("-s","--output_size", type = int, default = 1000)
    parser.add_argument("-i","--input_name", type = str, default = "data")
    parser.add_argument("-o","--output_name", type = str, default = "prob")
    parser.add_argument("-wpath","--weight_path", type = str, default = "./densenet121.wts")
    parser.add_argument("-epath","--engine_path", type = str, default = "./densenet121.engine")
    parser.add_argument("-t", "--engine_datatype",help="Optional.fp32,fp16,int8",type = str, default = "fp32")
    parser.add_argument("-f", "--file_path",type = str, default = None)
    parser.add_argument("-d", "--data_set",type = str, default = None)
    parser.add_argument("-l", "--label_set",type = str, default = None)
    parser.add_argument("-c", "--cache_file",type = str, default = None)
    parser.add_argument("-mean", "--mean_list",type = list, default = [0.485, 0.456, 0.406])
    parser.add_argument("-std", "--std_list",type = list, default = [0.229, 0.224, 0.225])
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    INPUT_H = args.height
    INPUT_W = args.width
    OUTPUT_SIZE = args.output_size
    INPUT_BLOB_NAME = args.input_name
    OUTPUT_BLOB_NAME = args.output_name
    WEIGHT_PATH = args.weight_path
    ENGINE_PATH = args.engine_path
    DATA_TYPE = args.engine_datatype
    FILE_PATH = args.file_path
    DATA_SET = args.data_set
    LABEL_SET = args.label_set
    CACHE_FILE = args.cache_file
    MEAN = args.mean_list
    STD = args.std_list
    API_to_model(INPUT_BLOB_NAME, OUTPUT_BLOB_NAME, BATCH_SIZE, INPUT_H, INPUT_W, OUTPUT_SIZE,
                WEIGHT_PATH, ENGINE_PATH, DATA_TYPE, FILE_PATH, DATA_SET, LABEL_SET, CACHE_FILE, MEAN, STD)
    