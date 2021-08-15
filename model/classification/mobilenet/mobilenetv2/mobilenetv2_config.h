#pragma once
//------------------模型相关---------------------------
# define MOBILENETV2_RESIZE 224
# define MOBILENETV2_BATCHSIZE 1
# define MOBILENETV2_MEAN_R 0.485
# define MOBILENETV2_MEAN_G 0.456
# define MOBILENETV2_MEAN_B 0.406
# define MOBILENETV2_STD_R 0.229
# define MOBILENETV2_STD_G 0.224
# define MOBILENETV2_STD_B 0.225
# define MOBILENETV2_CLASS_NUM 1000
# define MOBILENETV2_TOPNUMS 5 
# define MOBILENETV2_DEBUG_PATH "/workspace/lisen/_bushu/tensorrt-python/debug/mobilenetv2_"
//------------------tensorRT相关---------------------------
# define MOBILENETV2_ONNXFILENAME "/workspace/lisen/deploy/tensorRT/myproject/tensorrt-pythonv1/model_zoo/onnx/mobilenet_v2.onnx"   //onnx模型名
# define MOBILENETV2_BINFILENAME  "/workspace/lisen/deploy/tensorRT/myproject/tensorrt-pythonv1/model_zoo/trt/mobilenet_v2.trt"   //bin模型名
# define MOBILENETV2_CUDAID  0              //GPUid
# define MOBILENETV2_INPUTTENSORNAMES "input.1" //输入节点名
# define MOBILENETV2_OUTPUTTENSORNAMES "536" //输出节点名
# define MOBILENETV2_INPUTDIM 4                //输入维度
# define MOBILENETV2_OUTPUTDIM 5               //输出维度
# define MOBILENETV2_FP16 false
# define MOBILENETV2_INT8 false