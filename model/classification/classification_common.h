#pragma once
#include "tools.h"
#include "classification_interface.h"
#include "classification_utils.h"
#include "utils.h"
class ClassificationCommon :public ClassifierInterface
{
public:
    //模型构建
    bool Model_build(const classify_config &input_config);
    //模型推断
    bool Model_infer(std::vector<cv::Mat> &bgr_imgs,std::vector<image_info> &outputinfos);
    
private:
    bool PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &rgb_channel_img);
    bool PostProcess(std::vector<image_info>& outputinfos);
private:
    ClassificationUtils utils;
    classify_config config;
    
};
