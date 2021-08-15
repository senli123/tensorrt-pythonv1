#pragma once
#include <algorithm>
#include <vector>
typedef struct tag_InstanceInfo 
{
	float x1;           
	float y1;
	float x2;
	float y2;
	float score;            
	int class_id;   
} InstanceInfo;

class DetectionUtils
{
public:
    bool NMS(std::vector<InstanceInfo> predictions, float iou_thre,std::vector<InstanceInfo>& outputinfo);
    bool ScaleCoords(int net_require_width, int net_require_height, int src_img_width, int src_img_height, InstanceInfo& prediction);

};