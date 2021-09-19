#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "tools.h"
#include "detection_engine.h"
#include "config_operator.h"
int main(int, char**) {  
    cv::Mat bgr_img = cv::imread("/workspace/lisen/_bushu/tensorrt-python/data/bus.jpg");
    if (bgr_img.empty())
    {
        LOG(ERROR)<<"img is empty!";
        return -1;
    }
    std::string model_name = "Yolox";
    std::string config_name = "yolox_config";
    DetectionEngine engine;
    bool err;
    err = engine.init(model_name,config_name);
    if(!err)
    {
        LOG(ERROR)<<"engine init failed!";
        return -1;
    }
    std::vector<cv::Mat> bgr_imgs;
    bgr_imgs.push_back(bgr_img);
    err = engine.run(bgr_imgs);
    if(!err)
    {
        LOG(ERROR)<<"engine run failed!";
        return -1;
    }
    err = engine.uninit();
    return 0;
}
