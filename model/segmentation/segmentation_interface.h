#pragma once
#include <opencv2/opencv.hpp>
#include "Infer_data_def.h"
#include "tensorRT.h"
typedef struct tag_segmentation_config
{
	char onnx_path[100];  
	char bin_path[100];
    char input_name[100];
    char output_name[100];
    int cuda_id;
    int input_size;
    int batch_size;
    float meanVals[3];
    float normVals[3];
    int class_num;
    float confthre;
    bool FP16;
    bool INT8;
} segmentation_config;
	
class SegmentationInterface: public TensorRT_Interface
{
public:
    virtual bool Model_build(const segmentation_config &input_config)=0;
    //模型推断
    virtual bool Model_infer(std::vector<cv::Mat> &bgr_imgs,std::vector<cv::Mat> &mask_imgs)=0;
public:
};