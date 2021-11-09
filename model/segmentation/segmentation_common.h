#pragma once
#include "tools.h"
#include "segmentation_interface.h"
//#include "segmentation_utils.h"
#include "utils.h"
class SegmentationCommon :public SegmentationInterface
{
public:
    //模型构建
    bool Model_build(const segmentation_config &input_config);
    //模型推断
    bool Model_infer(std::vector<cv::Mat> &bgr_imgs,std::vector<cv::Mat> &mask_imgs);
    
private:
    bool PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &rgb_channel_img);
    bool PostProcess(std::vector<float*> &outputs,std::vector<cv::Mat> &mask_imgs, std::vector<int> &height_list, std::vector<int> &width_list);
private:
    segmentation_config config;
   
};