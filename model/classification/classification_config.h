#pragma once
#include "classification_interface.h"

//------------------mobilenetv2相关---------------------------
# define MOBILENETV2_RESIZE 224
# define MOBILENETV2_BATCHSIZE 1
# define MOBLIENETV2_meanVals { 0.485, 0.456, 0.406}
# define MOBLIENETV2_normVals {0.229,  0.224,  0.225}
# define MOBILENETV2_CLASS_NUM 1000
# define MOBILENETV2_TOPNUMS 5 
# define MOBILENETV2_ONNXFILENAME "/workspace/lisen/_bushu/tensorrt-python/model_zoo/onnx/mobilenet_v2.onnx"   //onnx模型名
# define MOBILENETV2_BINFILENAME  "/workspace/lisen/_bushu/tensorrt-python/model_zoo/trt/mobilenet_v2.trt"   //bin模型名
# define MOBILENETV2_CUDAID  1              //GPUid
# define MOBILENETV2_INPUTTENSORNAMES "input.1" //输入节点名
# define MOBILENETV2_OUTPUTTENSORNAMES "543" //输出节点名
# define MOBILENETV2_FP16 false
# define MOBILENETV2_INT8 false

extern const classify_config mobilenetv2_config;

//------------------alexnet相关---------------------------
# define ALEXNET_RESIZE 224
# define ALEXNET_BATCHSIZE 1
# define ALEXNET_meanVals { 0.485, 0.456, 0.406}
# define ALEXNET_normVals {0.229,  0.224,  0.225}
# define ALEXNET_CLASS_NUM 1000
# define ALEXNET_TOPNUMS 5 
# define ALEXNET_ONNXFILENAME "/workspace/lisen/_bushu/tensorrt-python/model_zoo/onnx/alexnet.onnx"   //onnx模型名
# define ALEXNET_BINFILENAME  "/workspace/lisen/_bushu/tensorrt-python/model_zoo/trt/alexnet.trt"   //bin模型名
# define ALEXNET_CUDAID  1              //GPUid
# define ALEXNET_INPUTTENSORNAMES "input.1" //输入节点名
# define ALEXNET_OUTPUTTENSORNAMES "36" //输出节点名
# define ALEXNET_FP16 false
# define ALEXNET_INT8 false
extern const classify_config alexnet_config;