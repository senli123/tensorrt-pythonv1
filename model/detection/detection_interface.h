#pragma once
#include <opencv2/opencv.hpp>
#include "Infer_data_def.h"
#include "tensorRT.h"
typedef struct tag_InstanceInfo 
{
	cv::Rect rect;
	float score;            
	int class_id;   
} InstanceInfo;

typedef struct tag_ClassInfo 
{
	std::vector<cv::Rect> o_rect;
	std::vector<float> o_rect_cof;
} ClassInfo;

typedef struct tag_detection_config
{
	char onnx_path[100];   //onnx模型名
	char bin_path[100];
    char input_name[100];
    char output_name[100];
    int cuda_id;
    int input_size;
    int batch_size;
    float meanVals[3];
    float normVals[3];
    int item_num;
    float confthre;
    float iouthre;
    int net_grid[3];
    int anchor_num;
    bool FP16;
    bool INT8;
} detection_config;


//------------------tensorRT相关---------------------------
class DetectionInterface: public TensorRT_Interface
{
public:
    virtual bool Model_build(const detection_config &input_config)=0;
    //模型推断
    virtual bool Model_infer(std::vector<cv::Mat> &bgr_imgs,std::vector<std::vector<InstanceInfo>> &defect_info)=0;
public:
}; 