#include "segmentation_common.h"
bool SegmentationCommon::Model_build(const segmentation_config &input_config)
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
        config.input_size,
        config.input_size,
        config.FP16,
        config.INT8
    };
    if (TensorRT_Interface::build(Tparams))
    {
        LOG(INFO)<<"[SegmentationCommon::Model_Build] build process succeed!";
        return true;
    }else{
        LOG(ERROR)<<"[SegmentationCommon::Model_Build] build process failed!";
        return false;
    }
}
bool SegmentationCommon::Model_infer(std::vector<cv::Mat> &bgr_imgs,std::vector<cv::Mat> &mask_imgs)
{
    mask_imgs.clear();
    TensorRT_Interface::img_size_clear(); 
    LOG(INFO)<<"[SegmentationCommon::Model_Infer] PreProcess start!";
    std::vector<int> height_list;
    std::vector<int> width_list;
    for (int img_index = 0; img_index < bgr_imgs.size(); img_index++)
    {
        cv::Mat bgr_img = bgr_imgs[img_index];
        height_list.push_back(bgr_img.rows);
        width_list.push_back(bgr_img.cols);
        std::vector<cv::Mat> preprocess_img(3);
        if(!PreProcess(bgr_img, preprocess_img))
        {
            LOG(ERROR)<<"[SegmentationCommon::Model_Infer] PreProcess failed!";
            std::ostringstream ostr;
            ostr << "the index of error image: " << img_index;
            LOG(ERROR)<<ostr.str().c_str();
            return false;
        }
        if(!TensorRT_Interface::processInput(preprocess_img))
        {
            LOG(ERROR)<<"[SegmentationCommon::Model_Infer] processInput process failed!";
            std::ostringstream ostr;
            ostr << "the index of error image: " << img_index;
            LOG(ERROR)<<ostr.str().c_str();
            return false;
        }
    }
    LOG(INFO)<<"[SegmentationCommon::Model_Infer] PreProcess succeed!";
    Debug_utils::set_time(PREPROCESS);
    LOG(INFO)<<"[SegmentationCommon::Model_Infer] infer start!";
    if(!TensorRT_Interface::infer())
    {
        LOG(ERROR)<<"[SegmentationCommon::Model_Infer] infer process failed!";
        return false;
    }
    LOG(INFO)<<"[SegmentationCommon::Model_Infer] infer succeed!";
    Debug_utils::set_time(INFER);
    LOG(INFO)<<"[SegmentationCommon::Model_Infer] Postprocess start!";
    if(!PostProcess(this->outputs, mask_imgs, height_list, width_list))
    {
        LOG(ERROR)<<"[SegmentationCommon::Model_Infer] Postprocess failed!";
        return false;
    }
    LOG(INFO)<<"[SegmentationCommon::Model_Infer] Postprocess succeed!";
    Debug_utils::set_time(POSTPROCESS);
    return true;
}
bool SegmentationCommon::PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &rgb_channel_img)
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
}
bool SegmentationCommon::PostProcess(std::vector<float*> &outputs,std::vector<cv::Mat> &mask_imgs,
std::vector<int> &height_list, std::vector<int> &width_list)
{
    try
    {
        //循环batch_size,求出每个batch上的mask并push到mask_imgs中
        float* out = outputs[0];
        for (int batch = 0; batch < config.batch_size; batch++)
        {
            for (int y = 0; y < config.input_size; y++)
            {
                for (int x = 0; x < config.input_size; x++)
                {
                    //在该通道上计算每个值的softmax并求最大值index
                    cv::Mat score;
                    float denominator = 0;
                    for (int c = 0; c < config.class_num; c++)
                    {
                       float t = out[batch * config.input_size * config.input_size * config.class_num +
                       y * config.input_size + x + config.input_size * config.input_size * c];
                       score.push_back(t);
                    }
                    
                }
            }
            
        }
        
        std::vector<std::vector<std::pair<float, int>>> index_infos;
        for (int batch = 0; batch < config.batch_size; batch++)
        {
            std::vector<std::pair<float, int>> index_info;
            for (int class_index = 0; class_index < config.class_num; class_index++)
            {
                index_info.push_back(std::make_pair(outputs[0][batch * config.class_num + class_index],class_index));
            }
            index_infos.push_back(index_info);
        }
        
    }
    catch(...)
    {
        return false;
    }
    return true;
}