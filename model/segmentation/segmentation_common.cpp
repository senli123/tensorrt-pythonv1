#include "segmentation_common.h"
int mask_color[11][3] = { {0, 0, 0},
                            {255, 248, 220},
                            {100, 149, 237},
                            {102, 205, 170},
                            {205, 133, 63},
                            {160, 32, 240},
                            {255, 64, 64},
                            {139, 69, 19},
                            {255, 0,  0},
                            {0,  255, 0},
                            {0,  0, 255}};
                            
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
        float* out = outputs[0];   //batch,c,h,w
        for (int batch = 0; batch < config.batch_size; batch++)
        {
            //每次要返回的img
            cv::Mat mask_mat = cv::Mat(config.input_size / 4,config.input_size / 4,CV_8UC3); 
            for (int y = 0; y < config.input_size / 4; y++)
            {
                for (int x = 0; x < config.input_size / 4; x++)
                {
                    //在该通道上计算每个值的softmax并求最大值index
                    int start_index = batch * config.input_size / 4 * config.input_size / 4 * config.class_num + y * config.input_size / 4 + x;
                    int max_index  = 0;
                    float max_score = 0.0;
                    for (int c = 0; c < config.class_num; c++)
                    {
                        float t = out[c * config.input_size / 4 * config.input_size / 4 + start_index];
                        if (t > max_score)
                        {
                            max_score = t;
                            max_index = c;
                        }
                    }
                    mask_mat.at<cv::Vec3b>(y,x)[0] = mask_color[max_index][0];
                    mask_mat.at<cv::Vec3b>(y,x)[1] = mask_color[max_index][1];
                    mask_mat.at<cv::Vec3b>(y,x)[2] = mask_color[max_index][2]; 
                }
            }
            //batch结束后将该图片返回
            cv::Mat rgb_mask_img;
            cv::cvtColor(mask_mat, rgb_mask_img, cv::COLOR_RGB2BGR);
            mask_imgs.push_back(rgb_mask_img);
        }
        
    }
    catch(...)
    {
        return false;
    }
    return true;
}
