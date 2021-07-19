#pragma once
#include <opencv2/opencv.hpp>
#include "yolov5_config.h"
#include "dete_utils.h"
class Yolov5_Utils
{
public:
	bool PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &After_Handle_img, 
    int &img_width, int &img_height);
    bool PostProcess(float *output_preds, std::vector<std::vector<InstanceInfo>> &defect_info, 
    int &img_width, int &img_height);
private:
	void Detection_Enlarge_imgbgr2rgb(cv::Mat &bgr_img, cv::Mat &crop_img_resize_rgb, 
    int &img_width, int &img_height);
private:
    DetectionUtils dete_utils;
};