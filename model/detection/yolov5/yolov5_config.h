#pragma once
//------------------模型相关---------------------------
# define YOLOV5_RESIZE 640
# define YOLOV5_PADDINGGRAY 114
# define YOLOV5_BATCHSIZE 1
# define YOLOV5_STRIDE   {8,16,32}
# define YOLOV5_CONFTHRE  0.25
# define YOLOV5_IOUTHRE   0.5
# define YOLOV5_ANCHORNUM 3
# define YOLOV5_CLASSNUM 10
# define YOLOV5_BASE_ANCHOR_WIDTH {{10,16,33}, {30,62,59}, {116,156,373}}
# define YOLOV5_BASE_ANCHOR_HEIGHT {{13,30,23}, {61, 45, 119}, {90, 198, 326}}
//------------------tensorRT相关---------------------------
# define YOLOV5_ONNXFILENAME "/workspace/lisen/_bushu/models/onnx/add_flj_no_pifeng_10cls_mix.onnx"   //onnx模型名
# define YOLOV5_BINFILENAME  "/workspace/lisen/_bushu/models/onnx/add_flj_no_pifeng_10cls_mix.trt"   //bin模型名
# define YOLOV5_CUDAID  1              //GPUid
# define YOLOV5_INPUTTENSORNAMES "images" //输入节点名
# define YOLOV5_OUTPUTTENSORNAMES "output" //输出节点名
# define YOLOV5_INPUTDIM 4                //输入维度
# define YOLOV5_OUTPUTDIM 5               //输出维度
# define YOLOV5_FP16 false
# define YOLOV5_INT8 false