#include "yolov5_utils.h"
bool Yolov5_Utils::PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &After_Handle_img, 
    int &img_width, int &img_height)
{   
    try
    {
        cv::Mat crop_img_resize_rgb;
        Detection_Enlarge_imgbgr2rgb(bgr_img,crop_img_resize_rgb, img_width, img_height);
        int rows = crop_img_resize_rgb.rows;
        for (int row_idx = 0; row_idx < rows; row_idx++)
        {
            auto *pointer_R = After_Handle_img[0].ptr<float>(row_idx);
            auto *pointer_G = After_Handle_img[1].ptr<float>(row_idx);
            auto *pointer_B = After_Handle_img[2].ptr<float>(row_idx);
            auto *crop_img_resize_pointer = crop_img_resize_rgb.ptr<float>(row_idx);
            for (int col_idx = 0; col_idx < crop_img_resize_rgb.cols; col_idx++)
            {
                float a = float(crop_img_resize_pointer[col_idx * 3 + 0]) / 255;
                float b = float(crop_img_resize_pointer[col_idx * 3 + 1]) / 255;
                float c = float(crop_img_resize_pointer[col_idx * 3 + 2]) / 255;
                
                pointer_R[col_idx] = a;
                pointer_G[col_idx] = b;
                pointer_B[col_idx] = c;
            }
            
        }
    }
    catch(const std::exception& e)
    {
        printf("PreProcess fail !");
        return false;
    }
    return true;
    
}
bool Yolov5_Utils::PostProcess(float *output_preds, std::vector<std::vector<InstanceInfo>> &defect_info, 
    int &img_width, int &img_height)
{
    try
    {
        int base_anchor_width[3][3] = YOLOV5_BASE_ANCHOR_WIDTH;
        int base_anchor_height[3][3] = YOLOV5_BASE_ANCHOR_HEIGHT;
        int stride[3] = YOLOV5_STRIDE;
        // 一共三层输出，第一层是下采样8倍，第二层16倍，第三层32倍
        int level_width_array[3] ={
                                    YOLOV5_RESIZE /stride[0],
                                    YOLOV5_RESIZE /stride[1],
                                    YOLOV5_RESIZE /stride[2],
                                    };
        //计算每个层的size
        int level_one_size = pow(level_width_array[0], 2);
        int level_two_size = pow(level_width_array[1], 2);
        int level_three_size = pow(level_width_array[2], 2);
        //计算一个batch中预测结果的个数
        int nums_per_batch = YOLOV5_ANCHORNUM * (level_one_size + level_two_size + level_three_size);
        //计算输出的数字个数,每个预测结果有（class_num + 5）个数字
        int datas_per_batch = nums_per_batch * (YOLOV5_CLASSNUM + 5); 
        for(int index_batch = 0; index_batch < YOLOV5_BATCHSIZE; ++index_batch)
        {
            std::map<int, std::vector<InstanceInfo>> batch_info;
            for (int index_data = index_batch * datas_per_batch; index_data < (index_batch + 1) * datas_per_batch; index_data+= (YOLOV5_CLASSNUM +5))
            {
                InstanceInfo ins_info;
                //背景丢弃
                if (output_preds[index_data+4] < YOLOV5_CONFTHRE)
                {
                    continue;
                }
                //计算类别和得分
                float max_score = 0.0;
                int max_index = 0;
                for (int index_score = 0; index_score < YOLOV5_CLASSNUM; ++index_score)
                {
                    if (output_preds[index_data + 5 + index_score] > max_score)
                    {
                        max_score = output_preds[index_data + 5 + index_score];
                        max_index = index_score;
                    }
                    
                }
                //分类得分太低丢弃
                if (max_score * output_preds[index_data + 4] < YOLOV5_CONFTHRE)
                {
                    continue;
                }
                ins_info.score = max_score * output_preds[index_data + 4];
                ins_info.class_id = max_index;
                //计算bbox
                int current_level = -1; //哪一个输出层上
                int pos_in_current_level = -1;   // 在当前层上的位置，为了计算在当前层上的哪个base_anchor上（每层都有三个base_anchor）
                if ((index_data % datas_per_batch) < YOLOV5_ANCHORNUM * level_one_size * (YOLOV5_CLASSNUM + 5)) //在第一层
                {
                    current_level = 0;
                    pos_in_current_level =  index_data % datas_per_batch;
                }
                else if ((index_data % datas_per_batch) - YOLOV5_ANCHORNUM * level_one_size * (YOLOV5_CLASSNUM + 5)
                        < YOLOV5_ANCHORNUM * level_two_size * (YOLOV5_CLASSNUM + 5)) //在第一层
                {
                    current_level = 1;
                    pos_in_current_level =  index_data % datas_per_batch - YOLOV5_ANCHORNUM * level_one_size * (YOLOV5_CLASSNUM + 5);
                }
                else
                {
                    current_level = 2;
                    pos_in_current_level =  index_data % datas_per_batch - YOLOV5_ANCHORNUM * (level_one_size+ level_two_size) * (YOLOV5_CLASSNUM + 5);

                }
                int current_base = -1;   // 每一层预测三个实例，每个实例对应不同anchor_base
                int pos_in_current_base = -1;
                int level_size_array[3] =   { 
                                                (YOLOV5_CLASSNUM + 5) * level_one_size,
                                                (YOLOV5_CLASSNUM + 5) * level_two_size,
                                                (YOLOV5_CLASSNUM + 5) * level_three_size 
                                            };
                int base_size_array[3][3] = {   
                                                {0, (YOLOV5_CLASSNUM + 5) * level_one_size, (YOLOV5_CLASSNUM + 5) * level_one_size * 2},
                                                {0, (YOLOV5_CLASSNUM + 5) * level_two_size, (YOLOV5_CLASSNUM + 5) * level_two_size * 2},
                                                {0, (YOLOV5_CLASSNUM + 5) * level_three_size, (YOLOV5_CLASSNUM + 5) * level_three_size * 2}
                                            };
                if (pos_in_current_level < level_size_array[current_level])
                {
                    current_base = 0;
                }
                else if (pos_in_current_level < level_size_array[current_level] *2)
                {
                    current_base = 1;
                    
                }else
                {
                    current_base = 2;
                }
                pos_in_current_base = pos_in_current_level - base_size_array[current_level][current_base];
                int x_grid = pos_in_current_base / (YOLOV5_CLASSNUM +5) % level_width_array[current_level];
                int y_grid = pos_in_current_base / (YOLOV5_CLASSNUM +5) / level_width_array[current_level];

                float x1 = (output_preds[index_data] * 2 - 0.5 + x_grid) * stride[current_level];
                float y1 = (output_preds[index_data + 1] * 2 -0.5 + y_grid) * stride[current_level];
                float width = pow(output_preds[index_data + 2] * 2, 2) * base_anchor_width[current_level][current_base];
                float height = pow(output_preds[index_data + 3] * 2, 2) * base_anchor_height[current_level][current_base];

                ins_info.x1 = x1 - width / 2;
                ins_info.y1 = y1 - height / 2;
                ins_info.x2 = x1 + width / 2;
                ins_info.y2 = y1 + height /2; 
                dete_utils.ScaleCoords(YOLOV5_RESIZE, YOLOV5_RESIZE, img_width,img_height, ins_info); 
                if (batch_info.find(ins_info.class_id) != batch_info.end())
                {
                    batch_info[ins_info.class_id].push_back(ins_info);
                }
                else
                {
                    std::vector<InstanceInfo> first_vector = {ins_info};
                    batch_info.insert(std::pair<int,std::vector<InstanceInfo>>(ins_info.class_id, first_vector));
                }  
            }
            for (auto &it :batch_info)
            {
                std::vector<InstanceInfo> outputinfo;
                dete_utils.NMS(it.second,YOLOV5_IOUTHRE,outputinfo);
                defect_info.push_back(outputinfo);
            }
                                    
        }
    }
    catch(const std::exception& e)
    {
        printf("PostProcess fail !");
        return false;
    }
    return true;
    
}

/*
 *-----------------private------------------------
 */
void Yolov5_Utils::Detection_Enlarge_imgbgr2rgb(cv::Mat &bgr_img, cv::Mat &crop_img_resize_rgb, 
    int &img_width, int &img_height)
{
    //计算放大的比例
	float ratio = std::min(float(YOLOV5_RESIZE) / float(img_width), float(YOLOV5_RESIZE) / float(img_height));
	int new_unpad_width = round(img_width * ratio);
	int new_unpad_height = round(img_height * ratio);
	//进行初次放大
	cv::Mat img_resize;
	if (new_unpad_width != img_width || new_unpad_height != img_height)
	{
		cv::resize(bgr_img, img_resize, cv::Size(new_unpad_width, new_unpad_height), cv::INTER_LINEAR);
	}
	else
	{
		img_resize = bgr_img;
	}
	//计算在四周要补充的像素
	int top = std::max(0, (YOLOV5_RESIZE - new_unpad_height) / 2);
	int bottom = std::max(0, (YOLOV5_RESIZE - new_unpad_height) / 2);
	if (std::max(0, (YOLOV5_RESIZE - new_unpad_height) % 2) == 1)
	{
		bottom++;
	}
	int left = std::max(0, (YOLOV5_RESIZE - new_unpad_width) / 2);
	int right = std::max(0, (YOLOV5_RESIZE - new_unpad_width) / 2);
	if (std::max(0, (YOLOV5_RESIZE - new_unpad_width) % 2) == 1)
	{
		right++;
	}
	cv::Mat img_makeborder;
	cv::copyMakeBorder(img_resize, img_makeborder, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(YOLOV5_PADDINGGRAY, YOLOV5_PADDINGGRAY, YOLOV5_PADDINGGRAY));
	cv::Mat img_rgb;
	cvtColor(img_makeborder, img_rgb, cv::COLOR_BGR2RGB);
	img_rgb.convertTo(crop_img_resize_rgb, CV_32F);
}