#pragma once
#include "segmentation_interface.h"
//------------------YOLOV5---------------------------
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