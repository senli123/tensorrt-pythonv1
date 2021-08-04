#pragma once
#include "tensorRT.h"
#include "mobilenetv2_data.h"
#include "classification_utils.h"
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
class MobilenetV2 :public TensorRT_Interface
{
public:
    MobilenetV2():TensorRT_Interface(Params_Init::MobilenetV2_tensorRT_data)
    {};
    //模型构建
    bool Model_Build();
    //模型推断
    bool Model_Infer(std::vector<cv::Mat> &bgr_imgs,std::vector<image_info>& outputinfo);
private:
    //图像预处理
    bool PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &After_Handle_img);
    bool PostProcess(std::vector<image_info>& outputinfo); 
private:
    ClassificationUtils utils;
};