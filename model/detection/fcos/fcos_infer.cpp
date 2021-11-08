#include "fcos_infer.h"
    //模型构建
bool Fcos:: Model_build(const detection_config &input_config)
{
    config = input_config;
	std::vector<char*> output_names = Utils::get_instance().split_name(config.output_name, ",");
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
	if (TensorRT_Interface::build(Tparams)) {
		LOG(INFO) << "[Fcos::Model_build] build process succeed!";
		return true;
	}
	else {
		LOG(ERROR) << "[Fcos::Model_build] build process failed!";
		return false;
	}
}
    //模型推断
bool Fcos::Model_infer(std::vector<cv::Mat> &bgr_imgs,std::vector<std::vector<InstanceInfo>> &output_infos)
{
    output_infos.clear();
	TensorRT_Interface::img_size_clear();
	std::vector<int> height_list;
	std::vector<int> width_list;
	LOG(INFO) << "[Fcos::Model_infer] PreProcess start!";
	for (int batch = 0; batch < bgr_imgs.size(); batch++)
	{
		cv::Mat bgr_img = bgr_imgs[batch];
		int height = bgr_img.rows;
		int width = bgr_img.cols;
		height_list.emplace_back(height);
		width_list.emplace_back(width);
		std::vector<cv::Mat> rgb_channel_img(3);
		if (!PreProcess(bgr_img, rgb_channel_img))
		{
			LOG(ERROR) << "[Fcos::Model_infer] PreProcess failed!";
			std::ostringstream ostr;
			ostr << "the index of error image: " << batch;
			LOG(ERROR) << ostr.str().c_str();
			return false;

		}
		if (!TensorRT_Interface::processInput(rgb_channel_img)) //在buffer中加入数据
		{
			LOG(ERROR) << "[Fcos::Model_infer] processInput process failed!";
			std::ostringstream ostr;
			ostr << "the index of error image: " << batch;
			LOG(ERROR) << ostr.str().c_str();
			return false;
		}
	}
	LOG(INFO) << "[Fcos::Model_infer] PreProcess succeed!";
	Debug_utils::set_time(PREPROCESS);
	LOG(INFO) << "[Fcos::Model_infer] infer start!";
	if (!TensorRT_Interface::infer())  //对图片进行推断
	{
		LOG(ERROR) << "[Fcos::Model_infer] infer process failed!";
		return false;
	}
	LOG(INFO) << "[Fcos::Model_infer] infer succeed!";
	Debug_utils::set_time(INFER);
	LOG(INFO) << "[Fcos::Model_infer] Postprocess start!";
	if (!PostProcess(output_infos, height_list, width_list))
	{
		LOG(ERROR) << "[Fcos::Model_infer] Postprocess failed!";
		return false;
	}
	LOG(INFO) << "[Fcos::Model_infer] Postprocess succeed!";
	Debug_utils::set_time(POSTPROCESS);
	return true;
}
bool  Fcos::PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &rgb_channel_img)
{
	try
	{
		// cv::Mat rgb_img;
		// cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
		// cv::resize(rgb_img, rgb_img, cv::Size(config.input_w, config.input_h));
		// cv::Mat rgb_resize_img;
		// rgb_img.convertTo(rgb_resize_img, CV_32F);
		// rgb_resize_img = rgb_resize_img / 255.0f;
		// cv::split(rgb_resize_img, rgb_channel_img);
         cv::Mat rgb_img,img_resize,Process_img;
        //cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
        cv::resize(bgr_img, img_resize, cv::Size(config.input_w, config.input_h));
        img_resize.convertTo(img_resize, CV_32F);
        //img_resize = img_resize / 255.0;
        Process_img= img_resize - cv::Scalar(config.meanVals[0], config.meanVals[1], config.meanVals[2]);
        std::vector<float> v_std_ = {config.normVals[0], config.normVals[1], config.normVals[2]};
        cv::split(Process_img, rgb_channel_img);
        for (int i = 0; i < 3; i++)
        {
            rgb_channel_img[i].convertTo(rgb_channel_img[i], CV_32FC1, 1.0 / v_std_[i]);
        }
        
	}
	catch (const std::exception& e)
	{
		return false;
	}
	return true;
}
bool Fcos::PostProcess(std::vector<std::vector<InstanceInfo>> &output_infos, std::vector<int> height_list, std::vector<int> width_list)
{
    //循环每一个batch
    //循环每一个scale，得到cls,cess,wh,
    //循环cls的channel层，找到每个点最大的对应通道,再把对应score与cess相同位置的值相乘得到实际score,
    //最终记录class,score,w,h,然后按照socre排序，找到前k个且socre>thre的点
    //找到这些点对应的框
    //nms筛选框最终输出
    bool err;
    std::vector<std::map<int,ClassInfo>> classinfo(config.batch_size);
    //求scale的大小
    //int len =sizeof(config.net_grid) /sizeof(config.net_grid[0]);
    int len = outputs.size()/3;
    for(int batch_idx=0; batch_idx < config.batch_size; batch_idx++)
    {
        //记录当前batch上所有的预选点信息
        std::vector<std::vector<float>> points_info; // index,score,x1,y1,x2,y2
        for(int scale_idx = 0; scale_idx < len; scale_idx++)
        {
            int scale = config.net_grid[scale_idx];
            //取到每一个batch上的wh,cls,cess
            float* wh = outputs[scale_idx * 3];  //batch * 4 * h * w
            float* cls = outputs[scale_idx * 3 + 1]; //batch * 80 * h * w
            float* cess = outputs[scale_idx * 3 + 2]; //batch * 1 * h * w

            int h = (int)ceil((float)config.input_h / (float)scale);
            int w = (int)ceil((float)config.input_w / (float)scale);
            int length = h * w;
            for(int h_idx=0; h_idx < h; h_idx++)
            {
                for(int w_idx=0; w_idx < w; w_idx++)
                {
                    int max_idx = 0;
                    float score = 0.0f;
                    int start_idx =  batch_idx *  h * w +  h_idx * w + w_idx;
                    for(int class_idx = 0; class_idx < config.item_num; class_idx++)
                    {
                        float temp_score =  DetectionUtils::get_instance().sigmoid(cls[start_idx + class_idx * length]);
                        if(temp_score>score)
                        {
                            max_idx = class_idx;
                            score = temp_score;
                        }
                    }
                    //计算实际的socre,idx,x1,y1,x2,y2
                    score *=  DetectionUtils::get_instance().sigmoid(cess[start_idx]);
                    //score *= cess[start_idx];
                    float x1 = (float)w_idx - (float)wh[start_idx];
                    float y1 = (float)h_idx - (float)wh[start_idx + length];
                    float x2 = (float)w_idx + (float)wh[start_idx + length * 2];
                    float y2 = (float)h_idx + (float)wh[start_idx + length * 3];
                    std::vector<float> point_info;
                    point_info.push_back((float)max_idx);
                    point_info.push_back(score);
                    point_info.push_back(x1);
                    point_info.push_back(y1);
                    point_info.push_back(x2);
                    point_info.push_back(y2);
                    points_info.push_back(point_info);
                }
            }
        }
        //当前batch上的所有scale循环之后取出满足条件的预选点
        std::sort(points_info.begin(), points_info.end(), 
                [](const std::vector<float>& a, const std::vector<float>& b) { return a[1] > b[1]; });
        int iters = std::min<int>(points_info.size(),config.top_num);
        int real_num = 0;
        for (int j = 0; j < iters; j++)
        {
            if (points_info[j][1]<config.confthre)
            {
                break;
            }
            real_num++;
        }
        //把满足条件的前real_num个预选点insert到最终的vector中
        std::vector<std::vector<float>> real_batch_fscore_max(points_info.begin(), points_info.begin() + real_num);
        //把该batch上的框做数据结构的转换
        std::map<int,ClassInfo> batch_classinfo;
        for(int k = 0; k < real_batch_fscore_max.size(); k++)
        {
            std::vector<float> info = real_batch_fscore_max[k];
            int point_idx = (int)info[0];
            float point_score = info[1];
            float point_w = info[4] - info[2];
            float point_h = info[5] - info[3];
            float point_x = info[2] + point_w/2;
            float point_y = info[3] + point_h /2;
            cv::Rect rect = cv::Rect(point_x, point_y, point_w, point_h);
            if (classinfo[batch_idx].find(point_idx)!=classinfo[batch_idx].end())
            {
                classinfo[batch_idx][point_idx].o_rect.push_back(rect);
                classinfo[batch_idx][point_idx].o_rect_cof.push_back(point_score);
            }else{
                ClassInfo new_class;
                new_class.o_rect = {rect};
                new_class.o_rect_cof = {point_score};
                classinfo[batch_idx].insert(std::pair<int,ClassInfo>(point_idx,new_class));

            }
        }
        
    }
    err = DetectionUtils::get_instance().NMS(output_infos, classinfo,
                                            height_list, width_list,
                                            config.confthre, config.iouthre,
                                            config.input_w,config.input_h);
    if (!err)
    {
        LOG(ERROR)<<"[Fcos::PostProcess] NMS process failed!";
        return false;
    }
    return true;
    
    // "wh","cls","cess"
    //     700,643,697  8
    //     810,753,807  16
    //     920,863,917  32
    //     1030,973,1027 64
    //     1140,1083,1137 128
}