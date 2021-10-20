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
		config.input_size,
		config.input_size,
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
bool Fcos::PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &rgb_channel_img)
{

}