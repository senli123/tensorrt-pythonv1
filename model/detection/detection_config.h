#pragma once
#include "detection_interface.h"
//------------------YOLOV5---------------------------
# define YOLOV5_ONNXFILENAME "/workspace/lisen/_bushu/tensorrt-python/model_zoo/onnx/yolov5s.onnx"   //onnx模型名
# define YOLOV5_BINFILENAME  "/workspace/lisen/_bushu/tensorrt-python/model_zoo/trt/yolov5s.trt"   //bin模型名
# define YOLOV5_INPUTTENSORNAMES "images" //输入节点名
# define YOLOV5_OUTPUTTENSORNAMES "397,458,519" //输出节点名
# define YOLOV5_CUDAID  1              //GPUid
# define YOLOV5_INPUT_H 640
# define YOLOV5_INPUT_W 640
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
# define YOLOX_INPUT_H 640
# define YOLOX_INPUT_W 640
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

//------------------CenterNet---------------------------
# define CENTERNET_ONNXFILENAME "/workspace/lisen/_bushu/tensorrt-python/model_zoo/onnx/ctdet_coco_dlav0_1x_v1.onnx"   //onnx模型名
# define CENTERNET_BINFILENAME  "/workspace/lisen/_bushu/tensorrt-python/model_zoo/trt/ctdet_coco_dlav0_1x_v1.trt"   //bin模型名
# define CENTERNET_INPUTTENSORNAMES "input" //输入节点名
# define CENTERNET_OUTPUTTENSORNAMES "hv_max,hm,wh,reg" //输出节点名
# define CENTERNET_CUDAID  0              //GPUid
# define CENTERNET_INPUT_H 512
# define CENTERNET_INPUT_W 512
# define CENTERNET_BATCH_SIZE 1
# define CENTERNET_meanVals {0.408, 0.447, 0.470}
# define CENTERNET_normVals {0.289, 0.274, 0.278}
// # define CENTERNET_meanVals { 0.485, 0.456, 0.406}
// # define CENTERNET_normVals {0.229,  0.224,  0.225}
# define CENTERNET_ITEM_NUM 80
# define CENTERNET_CONFTHRE  0.25
# define CENTERNET_IOUTHRE   0.5
# define CENTERNET_NET_GRID  {80,40,20}
# define CENTERNET_ANCHOR_NUM 3
# define CENTERNET_FP16 false
# define CENTERNET_INT8 false
# define CENTERNET_TOP 100
extern const detection_config centernet_config;

//------------------Fcos---------------------------
# define FCOS_ONNXFILENAME "/workspace/lisen/_bushu/tensorrt-python/model_zoo/onnx/fcos_v1.onnx"   //onnx模型名
# define FCOS_BINFILENAME  "/workspace/lisen/_bushu/tensorrt-python/model_zoo/trt/fcos_v1.trt"   //bin模型名
# define FCOS_INPUTTENSORNAMES "input" //输入节点名
# define FCOS_OUTPUTTENSORNAMES "700,643,697,810,753,807,920,863,917,1030,973,1027,1140,1083,1137" //输出节点名
# define FCOS_CUDAID  0              //GPUid
# define FCOS_INPUT_H 800
# define FCOS_INPUT_W 1216
# define FCOS_BATCH_SIZE 1
# define FCOS_meanVals {102.9801, 115.9465, 122.7717}
# define FCOS_normVals {1.0, 1.0, 1.0}
# define FCOS_ITEM_NUM 80
# define FCOS_CONFTHRE  0.25
# define FCOS_IOUTHRE   0.5
# define FCOS_NET_GRID  {8,16,32,64,128}
# define FCOS_ANCHOR_NUM 3
# define FCOS_FP16 false
# define FCOS_INT8 false
# define FCOS_TOP 100
extern const detection_config fcos_config;

//------------------retinanet---------------------------
# define RETINANET_ONNXFILENAME "/workspace/lisen/_bushu/tensorrt-python/model_zoo/onnx/retinanet.onnx"   //onnx模型名
# define RETINANET_BINFILENAME  "/workspace/lisen/_bushu/tensorrt-python/model_zoo/trt/retinanet.trt"   //bin模型名
# define RETINANET_INPUTTENSORNAMES "input" //输入节点名
# define RETINANET_OUTPUTTENSORNAMES "591,592,609,610,627,628,645,646,663,664" //输出节点名
# define RETINANET_CUDAID  0              //GPUid
# define RETINANET_INPUT_H 1333
# define RETINANET_INPUT_W 800
# define RETINANET_BATCH_SIZE 1
# define RETINANET_meanVals {123.675, 116.28, 103.53}
# define RETINANET_normVals {58.395, 57.12, 57.375}
# define RETINANET_ITEM_NUM 80
# define RETINANET_CONFTHRE  0.25
# define RETINANET_IOUTHRE   0.5
# define RETINANET_NET_GRID  {8,16,32,64,128}
# define RETINANET_ANCHOR_NUM 9
# define RETINANET_FP16 false
# define RETINANET_INT8 false
# define RETINANET_TOP 100
extern const detection_config retinanet_config;