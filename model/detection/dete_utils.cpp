#include "dete_utils.h"
// double DetectionUtils::sigmoid(double x)
// {
//     return (1 / (1 + exp(-x)));
// }
bool DetectionUtils::NMS(std::vector<std::vector<InstanceInfo>> &output_infos,std::vector<std::map<int,ClassInfo>> &classinfo, 
std::vector<int> &height_list,std::vector<int> &width_list,float &cof_threshold, float &nmsThreshold,int &inputW, int &inputH)
{
    try
    {
        for (int i = 0; i < classinfo.size(); i++)
        {
            int img_height = height_list[i];
            int img_width = width_list[i];
            std::vector<InstanceInfo> batch_instances;
            std::map<int,ClassInfo> batch_class_info_map = classinfo[i];
            std::map<int, ClassInfo>::reverse_iterator  iter;
            for(iter = batch_class_info_map.rbegin();iter != batch_class_info_map.rend();iter++)
            {
                int class_index = iter->first;
                ClassInfo batch_class_info = iter->second;
                std::vector<int> final_id;
                cv::dnn::NMSBoxes(batch_class_info.o_rect,batch_class_info.o_rect_cof,cof_threshold,nmsThreshold,final_id);
                for(int id=0; id<final_id.size(); id++)
                {
                    cv::Rect resize_rect = batch_class_info.o_rect[final_id[id]];
                    float score = batch_class_info.o_rect_cof[final_id[id]];
                    Update_coords(img_width, img_height, inputW, inputH, resize_rect);
                    InstanceInfo instance;
                    instance.rect = resize_rect;
                    instance.score = score;
                    instance.class_id = class_index;
                    batch_instances.push_back(instance);

                }
            }
            output_infos.push_back(batch_instances);
        }
        
    }
    catch(const std::exception& e)
    {
        return false;
    }
    return true;
    
}
// bool DetectionUtils::Update_coords(int img_width, int img_height, int resize_w, int resize_h, cv::Rect &rect)
// {
//     try
//     {
//         float x1 = rect.x;
//         float y1 = rect.y;
//         float x2 = rect.x + rect.width;
//         float y2 = rect.y + rect.height;
//         float gain_x = float(resize_w) /float(img_width);
//         float gain_y = float(resize_h) /float(img_height);
//         x1 /= gain_x;
//         y1 /= gain_y;
//         x2 /= gain_x;
//         y2 /= gain_y;
//         x1 = std::max(float(0),x1);
//         y1 = std::max(float(0),y1);
//         x2 = std::min(float(img_width -1),x2);
//         y2 = std::min(float(img_height-1),y2);
//         rect.x = x1;
//         rect.y = y1;
//         rect.width = x2-x1;
//         rect.height = y2-y1;
//     }
//     catch(const std::exception& e)
//     {
//         return false;
//     }
//     return true;

// }
bool DetectionUtils::Update_coords(int img_width, int img_height, int resize_w, int resize_h, cv::Rect &rect)
{
    try
    {
        float x1 = rect.x;
        float y1 = rect.y;
        float x2 = rect.x + rect.width;
        float y2 = rect.y + rect.height;
        float gain_x = float(resize_w) /float(img_width);
        float gain_y = float(resize_h) /float(img_height);
        float gain = std::min(gain_x,gain_y);
        float pad_w = (resize_w - img_width * gain) / 2.0;
	    float pad_h = (resize_h - img_height * gain) / 2.0;
        x1 -= pad_w;
        y1 -= pad_h;
        x2 -= pad_w;
        y2 -= pad_h;
        x1 /= gain;
        y1 /= gain;
        x2 /= gain;
        y2 /= gain;
        x1 = std::max(float(0),x1);
        y1 = std::max(float(0),y1);
        x2 = std::min(float(img_width -1),x2);
        y2 = std::min(float(img_height-1),y2);
        rect.x = x1;
        rect.y = y1;
        rect.width = x2-x1;
        rect.height = y2-y1;
    }
    catch(const std::exception& e)
    {
        return false;
    }
    return true;

}
// inline static bool CmpPrediction(InstanceInfo& inst_a, InstanceInfo& inst_b)
// {
// 	return inst_a.score > inst_b.score;
// }
// int FindBestPredIndex(int* cal_flag, int number)
// {
// 	for (int i = 0; i < number; ++i)
// 	{
// 		if (0 == cal_flag[i])
// 		{
// 			return i;
// 		}
// 	}
// 	return -1;
// }
// float CalOverlap(const InstanceInfo& prediction_a, const InstanceInfo& prediction_b)
// {
// 	float overlap = -1;
// 	float over_w = std::min(prediction_b.x2, prediction_a.x2) - std::max(prediction_b.x1, prediction_a.x1) + 1;
// 	if (over_w > 0)
// 	{
// 		float over_h = std::min(prediction_b.y2, prediction_a.y2) - std::max(prediction_b.y1, prediction_a.y1) + 1;
// 		if (over_h > 0)
// 		{
// 			float area_a = (prediction_a.y2 - prediction_a.y1 + 1) * (prediction_a.x2 - prediction_a.x1 + 1);
// 			float area_b = (prediction_b.y2 - prediction_b.y1 + 1) * (prediction_b.x2 - prediction_b.x1 + 1);
// 			overlap = over_w * over_h / (area_a + area_b - over_w * over_h);
// 		}
// 	}
// 	return overlap;
// }

// int ClipCoords(InstanceInfo& prediction, int src_img_width, int src_img_height)
// {
// 	prediction.x1 = std::max(float(0), prediction.x1);
// 	prediction.y1 = std::max(float(0), prediction.y1);
// 	prediction.x2 = std::min(prediction.x2, float(src_img_width - 1));
// 	prediction.y2 = std::min(prediction.y2, float(src_img_height - 1));
// 	return 0;
// }
// bool DetectionUtils::NMS(std::vector<InstanceInfo> predictions, float iou_thre,std::vector<InstanceInfo>& outputinfo)
// {
//     try
//     {
//         std::sort(predictions.begin(), predictions.end(), CmpPrediction);
//         int pre_num = predictions.size();
//         int *cal_flag = new int[pre_num];
//         memset(cal_flag, 0, sizeof(int) * pre_num);
//         int count_cal_num = 0;
//         while (count_cal_num < pre_num)
//         {
//             int max_score_index = FindBestPredIndex(cal_flag, pre_num);
//             InstanceInfo max_score_pred = predictions[max_score_index];
//             cal_flag[max_score_index] = 1; 
//             ++count_cal_num;
//             for (int index = max_score_index + 1; index < pre_num; index++)
//             {
//                 if ( 1 == cal_flag[index])
//                 {
//                     continue;
//                 }
//                 InstanceInfo compare_pred = predictions[index];
//                 float overlap = CalOverlap(max_score_pred, compare_pred);
//                 if (overlap > iou_thre)
//                 {
//                     predictions[index].score = 0;
//                     cal_flag[index] = 1;
//                     ++count_cal_num;
//                 }     
//             }  
//         }
//         delete []cal_flag;
//         for (auto each_pred : predictions)
//         {
//             if (each_pred.score >0)
//             {
//                 outputinfo.push_back(each_pred);
//             }
            
//         }
        
//     }
//     catch(const std::exception& e)
//     {
//         printf("NMS process fail !");
//         return false;
//     }
//     return true;
    
// }
// bool DetectionUtils::ScaleCoords(int net_require_width, int net_require_height, int src_img_width, int src_img_height, InstanceInfo& prediction)
// {
//     try
//     {
//         float gain = std::min(float(net_require_width) / float(src_img_width), float(net_require_height) / float(src_img_height));
//         float pad_w = (net_require_width - src_img_width * gain) / 2.0;
//         float pad_h = (net_require_height - src_img_height * gain) / 2.0;
//         prediction.x1 -= pad_w;
//         prediction.x2 -= pad_w;
//         prediction.y1 -= pad_h;
//         prediction.y2 -= pad_h;
//         prediction.x1 /= gain;
//         prediction.x2 /= gain;
//         prediction.y1 /= gain;
//         prediction.y2 /= gain;
//         ClipCoords(prediction, src_img_width, src_img_height);
//     }
//     catch(const std::exception& e)
//     {
//         printf("ScaleCoords process fail !");
//         return false;
//     }
    
// 	return true;
// }