#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "detection_interface.h"
// class DetectionUtils
// {
// public:
//     bool NMS(std::vector<InstanceInfo> predictions, float iou_thre,std::vector<InstanceInfo>& outputinfo);
//     bool ScaleCoords(int net_require_width, int net_require_height, int src_img_width, int src_img_height, InstanceInfo& prediction);

// };

class DetectionUtils
{
public:
    ~DetectionUtils(){}
    DetectionUtils(const DetectionUtils&)=delete;
    DetectionUtils& operator=(const DetectionUtils&)=delete;
    static DetectionUtils& get_instance(){
        static DetectionUtils instance;
        return instance;

    }
private:
    DetectionUtils(){};
	
public:
template <typename T>
	T sigmoid(T x);
	bool NMS(std::vector<std::vector<InstanceInfo>> &output_infos,std::vector<std::map<int,ClassInfo>> &classinfo, 
	std::vector<int> &height_list,std::vector<int> &width_list,float &cof_threshold, float &nmsThreshold,
	int &inputW, int &inputH);
    bool Update_coords(int img_width, int img_height, int resize_w, int resize_h, cv::Rect &rect);


};
template <typename T>
T DetectionUtils::sigmoid(T x){
    return 1 / (1+exp(-x));
}