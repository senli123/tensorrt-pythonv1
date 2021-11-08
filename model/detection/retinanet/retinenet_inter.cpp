#include "retinanet_infer.h"
float ratios[3] = { 0.5, 1.  ,2. };
float scales[3] = { 1. ,1.25992105 , 1.58740105 };
void get_anchor(float anchor[9][4],float sizes, float ratios[], float scales[])
{
	for (int j = 0; j < 3; j++)
	{
		float ratio = ratios[j];
		for (int i = 0; i < 3; i++)
		{
			float area = std::pow((scales[i] * sizes * 4),2);
			float w = std::sqrt(area/ ratio);
			float h = w * ratio;
			float x1 = 0 - w / 2;
			float y1 = 0 - h / 2;
			float x2 = w / 2;
			float y2 = h / 2;
			anchor[j * 3 + i][0] = x1;
			anchor[j * 3 + i][1] = y1;
			anchor[j * 3 + i][2] = x2;
			anchor[j * 3 + i][3] = y2;
		}

	}
}
 //模型构建
bool RetinaNet::Model_build(const detection_config &input_config)
{
    config = input_config;
    std::vector<char*> output_names = Utils::get_instance().split_name(config.output_name,",");
    TensorRT_data Tparams = {
        config.onnx_path,
        config.bin_path,
        config.cuda_id,
        config.input_name,
        output_names,
        config.batch_size,
        config.input_h,
        config.input_w,
        config.FP16,
        config.INT8
    };
    get_anchors();
    if (TensorRT_Interface::build(Tparams)) {
        LOG(INFO)<<"[RetinaNet::Model_build] build process succeed!";
        return true;
    }
    else {
        LOG(ERROR)<<"[RetinaNet::Model_build] build process failed!";
        return false;
    }
}
//模型推断
bool RetinaNet::Model_infer(std::vector<cv::Mat> &bgr_imgs,std::vector<std::vector<InstanceInfo>> &output_infos)
{
    output_infos.clear();
    TensorRT_Interface::img_size_clear();
    std::vector<int> height_list;
    std::vector<int> width_list;
    LOG(INFO)<<"[RetinaNet::Model_infer] PreProcess start!";
    for (int batch = 0; batch < bgr_imgs.size(); batch++)
    {
        cv::Mat bgr_img = bgr_imgs[batch];
        int height = bgr_img.rows;
        int width = bgr_img.cols;
        height_list.emplace_back(height);
        width_list.emplace_back(width);
        std::vector<cv::Mat> rgb_channel_img(3);
        if(!PreProcess(bgr_img,rgb_channel_img))
        {
            LOG(ERROR)<<"[RetinaNet::Model_infer] PreProcess failed!";
            std::ostringstream ostr;
		    ostr << "the index of error image: " << batch;
		    LOG(ERROR)<<ostr.str().c_str();
            return false;
            
        }
        if (!TensorRT_Interface::processInput(rgb_channel_img)) //在buffer中加入数据
		{
            LOG(ERROR)<<"[RetinaNet::Model_infer] processInput process failed!";
            std::ostringstream ostr;
		    ostr << "the index of error image: " << batch;
		    LOG(ERROR)<<ostr.str().c_str();
			return false;
		}
    }
    LOG(INFO)<<"[RetinaNet::Model_infer] PreProcess succeed!";
    Debug_utils::set_time(PREPROCESS);
    LOG(INFO)<<"[RetinaNet::Model_infer] infer start!";
    if (!TensorRT_Interface::infer())  //对图片进行推断
	{
		LOG(ERROR)<<"[RetinaNet::Model_infer] infer process failed!";
        return false;
	}
    LOG(INFO)<<"[RetinaNet::Model_infer] infer succeed!";
    Debug_utils::set_time(INFER);
    LOG(INFO)<<"[RetinaNet::Model_infer] Postprocess start!";
    if (!PostProcess(output_infos,height_list,width_list))
    {
        LOG(ERROR)<<"[RetinaNet::Model_infer] Postprocess failed!";
		return false;
    }
    LOG(INFO)<<"[RetinaNet::Model_infer] Postprocess succeed!";
    Debug_utils::set_time(POSTPROCESS);
    return true;
}
bool RetinaNet::PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &rgb_channel_img)
{
    try
    {
        cv::Mat rgb_img,resize_img;
        cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
        bool err = _imresize(rgb_img,resize_img);
        if (!err)
        {
            LOG(ERROR)<<"[RetinaNet::PreProcess] _imresize failed!";
            return false;
        }
        cv::Mat rgb_resize_img;
        resize_img.convertTo(rgb_resize_img,CV_32F);
        _normalize(rgb_resize_img,rgb_channel_img);
    }
    catch(const std::exception& e)
    {
        return false;
    }
    return true;
} 

bool RetinaNet::_imresize(cv::Mat &rgb_img, cv::Mat &resize_img)
{
    int img_width = rgb_img.cols;
    int img_height = rgb_img.rows;
    //计算放大比例
    float ratio = std::min( (float)config.input_h /(float)img_height, (float)config.input_w /(float)img_height);
    int new_width = round(img_width * ratio);
    int new_height = round(img_height * ratio);
    //进行初次放大
    cv::Mat re_img;
    cv::resize(rgb_img,re_img,cv::Size(new_width, new_height), cv::INTER_LINEAR);
    //计算在四周要补充的像素
    int top = std::max(0,(config.input_h  - new_height) /2);
    int bottom = std::max(0, (config.input_h  - new_height) / 2);
    if (std::max(0, (config.input_h  - new_height) % 2) == 1)
	{
		bottom++;
	}
    int left = std::max(0,(config.input_w - new_width) /2);
    int right = std::max(0, (config.input_w - new_width) / 2);
    if (std::max(0, (config.input_w - new_width) % 2) == 1)
	{
		right++;
	}
    cv::copyMakeBorder(re_img, resize_img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return true;
}
bool RetinaNet::_normalize(cv::Mat &rgb_resize_img, std::vector<cv::Mat> &rgb_channel_img)
{
        rgb_resize_img = rgb_resize_img - cv::Scalar(config.meanVals[0], config.meanVals[1], config.meanVals[2]);
        std::vector<float> v_std_ = {config.normVals[0], config.normVals[1], config.normVals[2]};
        cv::split(rgb_resize_img, rgb_channel_img);
        for (int i = 0; i < 3; i++)
        {
            rgb_channel_img[i].convertTo(rgb_channel_img[i], CV_32FC1, 1.0 / v_std_[i]);
        }
}
bool RetinaNet::PostProcess(std::vector<std::vector<InstanceInfo>> &output_infos, std::vector<int> height_list, std::vector<int> width_list)
{
    //循环每一个输出（即各种分辨率下），并得到该分辨率下的anchor
    //每一个batch送进nms的数据
    bool err;
    std::vector<std::map<int,ClassInfo>> classinfo(config.batch_size);
    int channel_num = config.item_num * 9;
    int channel_reg_num = 4 * 9;
    for(int scale_index = 0; scale_index < 5; scale_index++)
    {
        float* output_cls = outputs[scale_index * 2]; //batch ,(80 *9), h, w 
        float* output_reg = outputs[scale_index *2 +1]; //batch ,(4 *9), h, w 
        for(int batch_index = 0; batch_index < config.batch_size; batch_index ++)
        {
            for(int c_index = 0; c_index < channel_num ; c_index ++)
            {
                int h_num = (int)ceil((float)config.input_h / (float)(config.net_grid[scale_index]));
                int w_num = (int)ceil((float)config.input_w / (float)(config.net_grid[scale_index]));
                for(int h_index = 0; h_index < h_num ; h_index ++)
                {
                    for(int w_index = 0; w_index < w_num; w_index ++)
                    {
                        //计算当前在buffer中的位置
                        int index = batch_index * channel_num * h_num * w_num +
                        c_index * h_num * w_num + h_index * w_num + w_index;
                        float score = DetectionUtils::get_instance().sigmoid(output_cls[index]);
                        if( score >config.confthre)
                        {
                            //计算当前在哪个anchor上，对应的class_id是什么
                            int anchor_id = c_index / 80;
                            int class_id = c_index  % 80;
                            // int anchor_id = c_index % 9;
                            // int class_id = c_index  / 9;
                            //计算原始anchor
                            float anchor_x1 = anchors[scale_index][anchor_id][0] + (w_index) * config.net_grid[scale_index];
                            float anchor_y1 = anchors[scale_index][anchor_id][1] + (h_index) * config.net_grid[scale_index];
                            float anchor_x2 = anchors[scale_index][anchor_id][2] + (w_index) * config.net_grid[scale_index];
                            float anchor_y2 = anchors[scale_index][anchor_id][3] + (h_index) * config.net_grid[scale_index];
                            //计算网络输出4个值的位置
                            int output_x_index = batch_index * channel_reg_num * h_num * w_num + 
                            (anchor_id * 4)* h_num * w_num + h_index * w_num + w_index;
                            int output_y_index = batch_index * channel_reg_num * h_num * w_num + 
                            (anchor_id * 4 + 1) * h_num * w_num + h_index * w_num + w_index;
                            int output_w_index = batch_index * channel_reg_num * h_num * w_num + 
                            (anchor_id * 4 + 2) * h_num * w_num + h_index * w_num + w_index;
                            int output_h_index = batch_index * channel_reg_num * h_num * w_num + 
                            (anchor_id * 4 + 3) * h_num * w_num + h_index * w_num + w_index;
                            float output_x = output_reg[output_x_index] * 1.0;
                            float output_y = output_reg[output_y_index] * 1.0;
                            float output_w = output_reg[output_w_index] * 1.0;
                            float output_h = output_reg[output_h_index] * 1.0;
                            // output_x = (anchor_x1 + anchor_x2) * 0.5 + output_x * (anchor_x2 - anchor_x1);
                            // output_y = (anchor_y1 + anchor_y2) * 0.5 + output_y * (anchor_y2 - anchor_y1);
                            output_x = (w_index) * config.net_grid[scale_index] + output_x * (anchor_x2 - anchor_x1);
                            output_y = (h_index) * config.net_grid[scale_index] + output_y * (anchor_y2 - anchor_y1);
                            output_w =  exp(output_w)*(anchor_x2 - anchor_x1) *config.net_grid[scale_index];
                            output_h =  exp(output_h)*(anchor_y2 - anchor_y1) *config.net_grid[scale_index];
                            cv::Rect rect = cv::Rect(round(output_x),round(output_y),round(output_w),round(output_h));
                            if (classinfo[batch_index].find(class_id)!=classinfo[batch_index].end())
                            {
                                classinfo[batch_index][class_id].o_rect.push_back(rect);
                                classinfo[batch_index][class_id].o_rect_cof.push_back(score);
                            }else{
                                ClassInfo new_class;
                                new_class.o_rect = {rect};
                                new_class.o_rect_cof = {score};
                                classinfo[batch_index].insert(std::pair<int,ClassInfo>(class_id,new_class));

                            }
                        }

                    }
                }
            }
        }

    }
    //每张图类内做nms，并输出真实框
    err = DetectionUtils::get_instance().NMS(output_infos, classinfo,
                                            height_list, width_list,
                                            config.confthre, config.iouthre,
                                            config.input_w,config.input_h);
    if (!err)
    {
        LOG(ERROR)<<"[RetinaNet::PostProcess] NMS process failed!";
        return false;
    }
    return true;

}
bool RetinaNet::get_anchors()
{
    // if(sizeof(config.net_grid) / sizeof(config.net_grid[0])!=5)
    // {
    //     LOG(ERROR)<<"[RetinaNet::get_anchors] the size of net_grid must be 5!";
    //     return false;
    // }
    for (int i = 0; i < 5; i++)
	{
		get_anchor(anchors[i], config.net_grid[i], ratios, scales);
	}
    return true;
}
