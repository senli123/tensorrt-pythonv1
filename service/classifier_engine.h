#pragma once
#include <stdio.h>
#include "service_interface.h"
#include "config_operator.h"
#include "tools.h"
#include "classification_common.h"
#include "register.h"
#include "classification_config.h"
//#include "squeezenet/squeezenet_infer.h"
class ClassifierEngine
{
public:
    bool init(std::string model_name,std::string config_name);
    bool run(std::vector<cv::Mat> &imgs);
    bool uninit();
private:
    bool ParseConfig(std::string config_name, classify_config &config);
private:
    ClassifierInterface* model;
    Debug_utils utils;
};
