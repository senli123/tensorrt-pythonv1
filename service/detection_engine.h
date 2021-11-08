#pragma once
#include <stdio.h>
#include "service_interface.h"
#include "config_operator.h"
#include "tools.h"
#include "yolov5/yolov5_infer.h"
#include "yolox/yolox_infer.h"
#include "centernet/centernet_infer.h"
#include "fcos/fcos_infer.h"
#include "retinanet/retinanet_infer.h"
#include "register.h"
#include "detection_config.h"
#include "utils.h"
class DetectionEngine
{
public:
    bool init(std::string model_name,std::string config_name);
    bool run(std::vector<cv::Mat> &imgs);
    bool uninit();
private:
    bool ParseConfig(std::string config_name, detection_config &config);
private:
    DetectionInterface* model;
    Debug_utils utils;
};
