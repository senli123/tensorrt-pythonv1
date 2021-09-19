#pragma once
#include "detection_interface.h"
//------------------YOLOV5---------------------------
# define YOLOV5_ONNXFILENAME "/workspace/lisen/_bushu/tensorrt-python/model_zoo/onnx/yolov5s.onnx"   //onnx模型名
# define YOLOV5_BINFILENAME  "/workspace/lisen/_bushu/tensorrt-python/model_zoo/trt/yolov5s.trt"   //bin模型名
# define YOLOV5_INPUTTENSORNAMES "images" //输入节点名
# define YOLOV5_OUTPUTTENSORNAMES "397,458,519" //输出节点名
# define YOLOV5_CUDAID  1              //GPUid
# define YOLOV5_INPUT_SIZE 640
# define YOLOV5_BATCH_SIZE 1
# define YOLOV5_meanVals { 0.485, 0.456, 0.406}
# define YOLOV5_normVals {0.229,  0.224,  0.225}
# define YOLOV5_ITEM_NUM 85
# define YOLOV5_CONFTHRE  0.25
# define YOLOV5_IOUTHRE   0.5
# define YOLOV5_NET_GRID  {80,40,20}
# define YOLOV5_ANCHOR_NUM 3
# define YOLOV5_FP16 false
# define YOLOV5_INT8 false

extern const detection_config yolov5_config;

//------------------YOLOX---------------------------
# define YOLOX_ONNXFILENAME "/workspace/lisen/_bushu/tensorrt-python/model_zoo/onnx/yolox_s.onnx"   //onnx模型名
# define YOLOX_BINFILENAME  "/workspace/lisen/_bushu/tensorrt-python/model_zoo/trt/yolox_s.trt"   //bin模型名
# define YOLOX_INPUTTENSORNAMES "images" //输入节点名
# define YOLOX_OUTPUTTENSORNAMES "output" //输出节点名
# define YOLOX_CUDAID  1              //GPUid
# define YOLOX_INPUT_SIZE 640
# define YOLOX_BATCH_SIZE 1
# define YOLOX_meanVals { 0.485, 0.456, 0.406}
# define YOLOX_normVals {0.229,  0.224,  0.225}
# define YOLOX_ITEM_NUM 80
# define YOLOX_CONFTHRE  0.25
# define YOLOX_IOUTHRE   0.5
# define YOLOX_NET_GRID  {80,40,20}
# define YOLOX_ANCHOR_NUM 3
# define YOLOX_FP16 false
# define YOLOX_INT8 false

extern const detection_config yolox_config;