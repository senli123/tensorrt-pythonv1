#pragma once
#include <string>
typedef struct tag_TensorRT_data
{
    std::string OnnxFileName;   //onnx模型名
	std::string BinFileName;    //bin模型名
	int CudaID;                //GPUid
	std::string InputTensorNames; //输入节点名
	std::string OutputTensorNames; //输出节点名
	int BatchSize;                 //batch_size
	//int InputDim;                //输入维度
	//int OutputDim;               //输出维度
	int inputH;                  //输入图片的高
	int inputW;                  //输入图片的宽
	bool fp16;
	bool int8;
}TensorRT_data;


