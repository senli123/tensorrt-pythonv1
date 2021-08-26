#pragma once
#include <opencv2/opencv.hpp>
#include "Infer_data_def.h"
#include "tensorRT.h"
typedef struct tag_image_info
{
	std::vector<int> indexs;
	std::vector<float> scores;    
} image_info;

typedef struct tag_classify_config
{
	char onnx_path[100];   //onnx模型名
	char bin_path[100];
    char input_name[100];
    char output_name[100];
    int cuda_id;
    int input_dim;
    int output_dim;
    int input_size;
    int batch_size;
    float meanVals[3];
    float normVals[3];
    int class_num;
    int output_num;
    bool FP16;
    bool INT8;
} classify_config;


//------------------tensorRT相关---------------------------
class ClassifierInterface: public TensorRT_Interface
{
public:
    virtual bool Model_build(const classify_config &input_config)=0;
    //模型推断
    virtual bool Model_infer(std::vector<cv::Mat> &bgr_imgs,std::vector<image_info> &outputinfos)=0;
public:
}; 