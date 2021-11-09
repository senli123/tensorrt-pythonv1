#pragma once
#include "segmentation_interface.h"
//------------------unet---------------------------
# define UNET_ONNX_PATH "/workspace/lisen/deploy/ncnn/ncnn_demo/model_zoo/bin_param/unet-sim.param"   
# define UNET_BIN_PATH  "/workspace/lisen/deploy/ncnn/ncnn_demo/model_zoo/bin_param/unet-sim.bin"   
# define UNET_INPUTTENSORNAMES "input" //输入节点名
# define UNET_OUTPUTTENSORNAMES "output" //输出节点名
# define UNET_CUDAID  1    
# define UNET_INPUT_SIZE 640
# define UNET_BATCHSIZE 1
# define UNET_meanVals { 0, 0, 0}
# define UNET_normVals {1 / 255.f, 1 / 255.f, 1 / 255.f}
# define UNET_CLASS_NUM 2
# define UNET_CONFTHRE  0.5
# define UNET_FP16 false
# define UNET_INT8 false
extern const segmentation_config unet_config;

//------------------refinenet---------------------------
# define REFINENET_ONNX_PATH "/workspace/lisen/_bushu/tensorrt-python/model_zoo/onnx/refinenet.onnx"   
# define REFINENET_BIN_PATH  "/workspace/lisen/_bushu/tensorrt-python/model_zoo/trt/refinenet.trt"   
# define REFINENET_INPUTTENSORNAMES "input.1" //输入节点名
# define REFINENET_OUTPUTTENSORNAMES "593" //输出节点名
# define REFINENET_CUDAID  1    
# define REFINENET_INPUT_SIZE 512
# define REFINENET_BATCHSIZE 1
# define REFINENET_meanVals { 0.485, 0.456, 0.406}
# define REFINENET_normVals {0.229, 0.224, 0.225}
# define REFINENET_CLASS_NUM 11
# define REFINENET_CONFTHRE  0.5
# define REFINENET_FP16 false
# define REFINENET_INT8 false
extern const segmentation_config refinenet_config;