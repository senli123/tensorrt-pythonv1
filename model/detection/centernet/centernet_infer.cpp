#include "centernet_infer.h"
//模型构建
bool CenterNet::Model_build(const detection_config &input_config)
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
		LOG(INFO) << "[CenterNet::Model_build] build process succeed!";
		return true;
	}
	else {
		LOG(ERROR) << "[CenterNet::Model_build] build process failed!";
		return false;
	}
}
//模型推断
bool CenterNet::Model_infer(std::vector<cv::Mat> &bgr_imgs, std::vector<std::vector<InstanceInfo>> &output_infos)
{
	output_infos.clear();
	TensorRT_Interface::img_size_clear();
	std::vector<int> height_list;
	std::vector<int> width_list;
	LOG(INFO) << "[CenterNet::Model_infer] PreProcess start!";
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
			LOG(ERROR) << "[CenterNet::Model_infer] PreProcess failed!";
			std::ostringstream ostr;
			ostr << "the index of error image: " << batch;
			LOG(ERROR) << ostr.str().c_str();
			return false;

		}
		if (!TensorRT_Interface::processInput(rgb_channel_img)) //在buffer中加入数据
		{
			LOG(ERROR) << "[CenterNet::Model_infer] processInput process failed!";
			std::ostringstream ostr;
			ostr << "the index of error image: " << batch;
			LOG(ERROR) << ostr.str().c_str();
			return false;
		}
	}
	LOG(INFO) << "[CenterNet::Model_infer] PreProcess succeed!";
	Debug_utils::set_time(PREPROCESS);
	LOG(INFO) << "[CenterNet::Model_infer] infer start!";
	if (!TensorRT_Interface::infer())  //对图片进行推断
	{
		LOG(ERROR) << "[CenterNet::Model_infer] infer process failed!";
		return false;
	}
	LOG(INFO) << "[CenterNet::Model_infer] infer succeed!";
	Debug_utils::set_time(INFER);
	LOG(INFO) << "[CenterNet::Model_infer] Postprocess start!";
	if (!PostProcess(output_infos, height_list, width_list))
	{
		LOG(ERROR) << "[CenterNet::Model_infer] Postprocess failed!";
		return false;
	}
	LOG(INFO) << "[CenterNet::Model_infer] Postprocess succeed!";
	Debug_utils::set_time(POSTPROCESS);
	return true;
}
bool CenterNet::PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &rgb_channel_img)
{
	try
	{
		cv::Mat rgb_img;
		cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
		cv::resize(rgb_img, rgb_img, cv::Size(config.input_w, config.input_h));
		cv::Mat rgb_resize_img;
		rgb_img.convertTo(rgb_resize_img, CV_32F);
		rgb_resize_img = rgb_resize_img / 255.0f;
		cv::split(rgb_resize_img, rgb_channel_img);
	}
	catch (const std::exception& e)
	{
		return false;
	}
	return true;
}
bool CenterNet::PostProcess(std::vector<std::vector<InstanceInfo>> &output_infos, std::vector<int> height_list, std::vector<int> width_list)
{
	try
	{
		//先对比hv_max和hm,找到预选点的socre,index和坐标
	//再通过index找到wh和offset，计算真实bbox
	//每一个batch送进nms的数据
		bool err;
		float* hv_max = outputs[0];  //batch*80(类别)*128*128
		float* hm = outputs[1];  //batch*80(类别)*128*128
		float* wh = outputs[2];  //batch*2*128*128
		float* reg = outputs[3]; //batch*2*128*128
		//挑出_mns之后预选的点
		std::vector<std::vector<std::vector<float>>> fscore_max; //三层vector,最外层为batch,中间层为所有预选点，内层分别为class,score,h_index,w_index
		err = get_index(hv_max, hm, fscore_max);
		if (!err)
		{
			LOG(ERROR) << "[CenterNet::PostProcess] get_index failed!";
			return false;
		}
		//循环每个batch和每个预选点得到真正的框
		int spacial_size = (config.input_h / 4)*(config.input_w / 4);
		 float *scale0 = wh;
		// float *scale1 = wh + spacial_size;

		 float *offset0 = reg;
		// float *offset1 = reg + spacial_size;

		for (int batch_index = 0; batch_index < config.batch_size; batch_index++)
		{
			std::vector<InstanceInfo> batch_output_instance;
			std::vector<std::vector<float>> batch_instacnes = fscore_max[batch_index];
			for (int instance_num = 0; instance_num < batch_instacnes.size(); instance_num++)
			{
				std::vector<float> instance = batch_instacnes[instance_num];
				int class_id = (int)instance[0];
				float score = instance[1];
				float h_index = instance[2];
				float w_index = instance[3];
				int index = h_index * config.input_w / 4 + w_index;
				int x1 = (int)(w_index + offset0[index]) * 4;
				// int y1 = (int)(h_index + offset1[index]) * 4;
                int y1 = (int)(h_index + offset0[index + spacial_size]) * 4;
				int w = (int)scale0[index] * 4;
				//int h = (int)scale1[index] * 4;
                int h = (int)scale0[index + spacial_size] * 4;
				cv::Rect rect;
				rect.x = x1 - w/2;
				rect.y = y1 - h/2;
				rect.width = w;
				rect.height = h;
				err = DetectionUtils::get_instance().Update_coords(width_list[batch_index], height_list[batch_index], config.input_w, config.input_h, rect);
				InstanceInfo output_instance;
				output_instance.class_id = class_id;
				output_instance.score = score;
				output_instance.rect = rect;
				batch_output_instance.push_back(output_instance);
			}
			output_infos.push_back(batch_output_instance);
		}
	}
	catch (const std::exception&)
	{
		return true;
	}
	return true;

}
bool CenterNet::get_index(float* max_heatmap, float* heatmap, std::vector<std::vector<std::vector<float>>> &fscore_max)
{
	try
	{
		std::vector<std::vector<std::vector<float>>> temp_fscore_max;
		//循环batch
		int feature_size = config.input_h * config.input_w / 16;
		for (int batch = 0; batch < config.batch_size; batch++)
		{
			std::vector<std::vector<float>> batch_fscore_max;
			for (int class_index = 0; class_index < config.item_num; class_index++)
			{
				for (int h_index = 0; h_index < config.input_h/4; h_index++)
				{
					for (int w_index = 0; w_index < config.input_w/4; w_index++)
					{
						int index = batch * config.item_num * feature_size + class_index * feature_size + h_index * config.input_w/4 + w_index;
						if (max_heatmap[index] == heatmap[index]) //满足条件挑出对应的预选信息
						{
							//class, score, h_index, w_index
							std::vector<float> info;
							info.push_back(class_index);
							info.push_back(max_heatmap[index]);
							info.push_back(h_index);
							info.push_back(w_index);
							batch_fscore_max.push_back(info);
						}
					}
				}
			}
			temp_fscore_max.push_back(batch_fscore_max);	
		}
        //对每一个batch上面的预选点进行排序，挑选前top个，且score满足条件的点
        for (int i = 0; i < temp_fscore_max.size(); i++)
        {
            std::vector<std::vector<float>> temp_batch_fscore_max = temp_fscore_max[i];
            std::sort(temp_batch_fscore_max.begin(), temp_batch_fscore_max.end(), 
                [](const std::vector<float>& a, const std::vector<float>& b) { return a[1] > b[1]; });
            int iters = std::min<int>(temp_batch_fscore_max.size(),config.top_num);
            int real_num = 0;
            for (int j = 0; j < iters; j++)
            {
                if (temp_batch_fscore_max[j][1]<config.confthre)
                {
                    break;
                }
                real_num++;
            }
            //把满足条件的前real_num个预选点insert到最终的vector中
            std::vector<std::vector<float>> real_batch_fscore_max(temp_batch_fscore_max.begin(), temp_batch_fscore_max.begin() + real_num);
            fscore_max.push_back(real_batch_fscore_max);
        }
	}
	catch (const std::exception&)
	{
		return false;
	}
	
	return true;
}


