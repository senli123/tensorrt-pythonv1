#include "classification_common.h"
bool ClassificationCommon::Model_build(const classify_config &input_config)
{
    config = input_config;
    TensorRT_data Tparams = {
        config.onnx_path,
        config.bin_path,
        config.cuda_id,
        config.input_name,
        config.output_name,
        config.batch_size,
        config.input_size,
        config.input_size,
        config.FP16,
        config.INT8
    };
	if (TensorRT_Interface::build(Tparams)) {
        LOG(INFO)<<"[ClassificationCommon::Model_build] build process succeed!";
		return true;
	}
	else {
        LOG(ERROR)<<"[ClassificationCommon::Model_build] build process failed!";
		return false;
	}
}
    //模型推断
bool ClassificationCommon::Model_infer(std::vector<cv::Mat> &bgr_imgs, std::vector<image_info>& outputinfos)
{
    outputinfos.clear();
    LOG(INFO)<<"[ClassificationCommon::Model_infer] PreProcess start!";
    for (int batch = 0; batch < bgr_imgs.size(); batch++)
    {
        cv::Mat bgr_img = bgr_imgs[batch];
        std::vector<cv::Mat> After_Process_img(3);
        if(!PreProcess(bgr_img,After_Process_img))
        {
            LOG(ERROR)<<"[ClassificationCommon::Model_infer] PreProcess failed!";
            std::ostringstream ostr;
		    ostr << "the index of error image: " << batch;
		    LOG(ERROR)<<ostr.str().c_str();
            return false;
            
        }
        if (!TensorRT_Interface::processInput(After_Process_img)) //在buffer中加入数据
		{
            LOG(ERROR)<<"[ClassificationCommon::Model_infer] processInput process failed!";
            std::ostringstream ostr;
		    ostr << "the index of error image: " << batch;
		    LOG(ERROR)<<ostr.str().c_str();
			return false;
		}
    }
    LOG(INFO)<<"[ClassificationCommon::Model_infer] PreProcess succeed!";
    Debug_utils::set_time(PREPROCESS);
    LOG(INFO)<<"[ClassificationCommon::Model_infer] infer start!";
    if (!TensorRT_Interface::infer())  //对图片进行推断
	{
		LOG(ERROR)<<"[ClassificationCommon::Model_infer] infer process failed!";
        return false;
	}
    LOG(INFO)<<"[ClassificationCommon::Model_infer] infer succeed!";
    Debug_utils::set_time(INFER);
    LOG(INFO)<<"[ClassificationCommon::Model_infer] Postprocess start!";
    if (!PostProcess(outputinfos))
    {
        LOG(ERROR)<<"[ClassificationCommon::Model_infer] Postprocess failed!";
		return false;
    }
    LOG(INFO)<<"[ClassificationCommon::Model_infer] Postprocess succeed!";
    Debug_utils::set_time(POSTPROCESS);
    return true;
    
    
}
    //图像预处理
bool ClassificationCommon::PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &rgb_channel_img)
{
    try
    {
        cv::Mat rgb_img,img_resize,Process_img;
        cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
        cv::resize(rgb_img, img_resize, cv::Size(config.input_size, config.input_size));
        img_resize.convertTo(img_resize, CV_32F);
        img_resize = img_resize / 255.0;
        Process_img= img_resize - cv::Scalar(config.meanVals[0], config.meanVals[1], config.meanVals[2]);
        std::vector<float> v_std_ = {config.normVals[0], config.normVals[1], config.normVals[2]};
        cv::split(Process_img, rgb_channel_img);
        for (int i = 0; i < 3; i++)
        {
            rgb_channel_img[i].convertTo(rgb_channel_img[i], CV_32FC1, 1.0 / v_std_[i]);
        }
    }
    catch(...)
    {
        return false;
    }
    return true;
    // cv::dnn::blobFromImage(bgr_img, Process_img, MOBILENETV2_STD , cv::Size(MOBILENETV2_RESIZE, MOBILENETV2_RESIZE), cv::Scalar(MOBILENETV2_MEAN_R, MOBILENETV2_MEAN_G, MOBILENETV2_MEAN_B), true, false);
    // cv::split(Process_img, After_Handle_img);
}
bool ClassificationCommon::PostProcess(std::vector<image_info>& outputinfo)
{
    try
    {
        std::vector<std::vector<std::pair<float, int>>> index_infos;
        for (int batch = 0; batch < config.batch_size; batch++)
        {
            std::vector<std::pair<float, int>> index_info;
            for (int class_index = 0; class_index < config.class_num; class_index++)
            {
                index_info.push_back(std::make_pair(output[batch * config.class_num + class_index],class_index));
            }
            index_infos.push_back(index_info);
        }
        utils.TopNums(index_infos, config.class_num, outputinfo);
    }
    catch(...)
    {
        return false;
    }
    return true;
    
}