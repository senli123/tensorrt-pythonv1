#include "mobilenetv2.h"
bool MobilenetV2::Model_Build()
{
	if (TensorRT_Interface::build()) {
        LOG(INFO)<<"[MobilenetV2::Model_Build] build process succeed!";
		return true;
	}
	else {
        LOG(ERROR)<<"[MobilenetV2::Model_Build] build process failed!";
		return false;
	}
}
    //模型推断
bool MobilenetV2::Model_Infer(std::vector<cv::Mat> &bgr_imgs, std::vector<image_info>& outputinfo)
{
    outputinfo.clear();
    LOG(INFO)<<"[MobilenetV2::Model_Infer] PreProcess start!";
    for (int batch = 0; batch < bgr_imgs.size(); batch++)
    {
        cv::Mat bgr_img = bgr_imgs[batch];
        std::vector<cv::Mat> After_Process_img(3);
        if(!PreProcess(bgr_img,After_Process_img))
        {
            LOG(ERROR)<<"[MobilenetV2::Model_Infer] PreProcess failed!";
            std::ostringstream ostr;
		    ostr << "the index of error image: " << batch;
		    LOG(ERROR)<<ostr.str().c_str();
            return false;
            
        }
        if (!TensorRT_Interface::processInput(After_Process_img)) //在buffer中加入数据
		{
            LOG(ERROR)<<"[MobilenetV2::Model_Infer] processInput process failed!";
            std::ostringstream ostr;
		    ostr << "the index of error image: " << batch;
		    LOG(ERROR)<<ostr.str().c_str();
			return false;
		}
    }
    LOG(INFO)<<"[MobilenetV2::Model_Infer] PreProcess succeed!";
    Debug_utils::set_time(PREPROCESS);
    LOG(INFO)<<"[MobilenetV2::Model_Infer] infer start!";
    if (!TensorRT_Interface::infer())  //对图片进行推断
	{
		LOG(ERROR)<<"[MobilenetV2::Model_Infer] infer process failed!";
        return false;
	}
    LOG(INFO)<<"[MobilenetV2::Model_Infer] infer succeed!";
    Debug_utils::set_time(INFER);
    LOG(INFO)<<"[MobilenetV2::Model_Infer] Postprocess start!";
    if (!PostProcess(outputinfo))
    {
        LOG(ERROR)<<"[MobilenetV2::Model_Infer] Postprocess failed!";
		return false;
    }
    LOG(INFO)<<"[MobilenetV2::Model_Infer] Postprocess succeed!";
    Debug_utils::set_time(POSTPROCESS);
    return true;
    
    
}
    //图像预处理
bool MobilenetV2::PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &After_Handle_img)
{
    try
    {
        cv::Mat rgb_img,img_resize,Process_img;
        cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
        cv::resize(rgb_img, img_resize, cv::Size(MOBILENETV2_RESIZE, MOBILENETV2_RESIZE));
        img_resize.convertTo(img_resize, CV_32F);
        img_resize = img_resize / 255.0;
        Process_img= img_resize - cv::Scalar(MOBILENETV2_MEAN_R, MOBILENETV2_MEAN_G, MOBILENETV2_MEAN_B);
        std::vector<float> v_std_ = {MOBILENETV2_STD_R, MOBILENETV2_STD_G, MOBILENETV2_STD_B};
        cv::split(Process_img, After_Handle_img);
        for (int i = 0; i < 3; i++)
        {
            After_Handle_img[i].convertTo(After_Handle_img[i], CV_32FC1, 1.0 / v_std_[i]);
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
bool MobilenetV2::PostProcess(std::vector<image_info>& outputinfo)
{
    try
    {
        std::vector<std::vector<std::pair<float, int>>> index_infos;
        for (int batch = 0; batch < MOBILENETV2_BATCHSIZE; batch++)
        {
            std::vector<std::pair<float, int>> index_info;
            for (int class_index = 0; class_index < MOBILENETV2_CLASS_NUM; class_index++)
            {
                index_info.push_back(std::make_pair(output[batch * MOBILENETV2_CLASS_NUM + class_index],class_index));
            }
            index_infos.push_back(index_info);
        }
        utils.TopNums(index_infos, MOBILENETV2_TOPNUMS, outputinfo);
    }
    catch(...)
    {
        return false;
    }
    return true;
    
}