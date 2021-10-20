#pragma once
#include "tools.h"
#include "detection_interface.h"
#include "utils.h"
#include "dete_utils.h"
class Fcos : public DetectionInterface
{
public:
    //模型构建
    bool Model_build(const detection_config &input_config);
    //模型推断
    bool Model_infer(std::vector<cv::Mat> &bgr_imgs,std::vector<std::vector<InstanceInfo>> &output_infos);
private:
    bool PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &rgb_channel_img);
    bool PostProcess(std::vector<std::vector<InstanceInfo>> &output_infos, std::vector<int> height_list, std::vector<int> width_list);
private:
    detection_config config;
};