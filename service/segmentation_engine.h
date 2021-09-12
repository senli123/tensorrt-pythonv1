// #pragma once
// #include <stdio.h>
// #include "service_interface.h"
// #include "config_operator.h"
// #include "tools.h"
// #include "segmentation_common.h"
// #include "register.h"
// #include "segmentation_config.h"
// #include "utils.h"
// class SegmentationEngine
// {
// public:
//     bool init(std::string model_name,std::string config_name);
//     bool run(std::vector<cv::Mat> &img);
//     bool uninit();
// private:
//     bool ParseConfig(std::string config_name, segmentation_config &config);
// private:
//     SegmentationInterface* model;
//     Debug_utils utils;
// };