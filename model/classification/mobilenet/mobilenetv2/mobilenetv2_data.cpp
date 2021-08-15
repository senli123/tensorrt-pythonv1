#include "mobilenetv2_data.h"
TensorRT_data Params_Init::MobilenetV2_tensorRT_data={
                        MOBILENETV2_ONNXFILENAME,   //onnx模型名
                        MOBILENETV2_BINFILENAME,    //bin模型名
                        MOBILENETV2_CUDAID,                //GPUid
                        MOBILENETV2_INPUTTENSORNAMES, //输入节点名
                        MOBILENETV2_OUTPUTTENSORNAMES , //输出节点名
                        MOBILENETV2_BATCHSIZE,                 //batch_size
                        MOBILENETV2_RESIZE,                  //输入图片的高
                        MOBILENETV2_RESIZE,                  //输入图片的宽
                        MOBILENETV2_FP16,
                        MOBILENETV2_INT8
};