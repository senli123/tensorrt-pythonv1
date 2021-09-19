#include "yolox_infer.h"
const std::vector<int> strides = {8,16,32};
 //模型构建
bool Yolox::Model_build(const detection_config &input_config)
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
    if (TensorRT_Interface::build(Tparams)) {
        LOG(INFO)<<"[Yolox::Model_build] build process succeed!";
        return true;
    }
    else {
        LOG(ERROR)<<"[Yolox::Model_build] build process failed!";
        return false;
    }
}
//模型推断
bool Yolox::Model_infer(std::vector<cv::Mat> &bgr_imgs,std::vector<std::vector<InstanceInfo>> &output_infos)
{
    output_infos.clear();
    TensorRT_Interface::img_size_clear();
    std::vector<int> height_list;
    std::vector<int> width_list;
    LOG(INFO)<<"[Yolox::Model_infer] PreProcess start!";
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
            LOG(ERROR)<<"[Yolox::Model_infer] PreProcess failed!";
            std::ostringstream ostr;
		    ostr << "the index of error image: " << batch;
		    LOG(ERROR)<<ostr.str().c_str();
            return false;
            
        }
        if (!TensorRT_Interface::processInput(rgb_channel_img)) //在buffer中加入数据
		{
            LOG(ERROR)<<"[Yolox::Model_infer] processInput process failed!";
            std::ostringstream ostr;
		    ostr << "the index of error image: " << batch;
		    LOG(ERROR)<<ostr.str().c_str();
			return false;
		}
    }
    LOG(INFO)<<"[Yolox::Model_infer] PreProcess succeed!";
    Debug_utils::set_time(PREPROCESS);
    LOG(INFO)<<"[Yolox::Model_infer] infer start!";
    if (!TensorRT_Interface::infer())  //对图片进行推断
	{
		LOG(ERROR)<<"[Yolox::Model_infer] infer process failed!";
        return false;
	}
    LOG(INFO)<<"[Yolox::Model_infer] infer succeed!";
    Debug_utils::set_time(INFER);
    LOG(INFO)<<"[Yolox::Model_infer] Postprocess start!";
    if (!PostProcess(output_infos,height_list,width_list))
    {
        LOG(ERROR)<<"[Yolox::Model_infer] Postprocess failed!";
		return false;
    }
    LOG(INFO)<<"[Yolox::Model_infer] Postprocess succeed!";
    Debug_utils::set_time(POSTPROCESS);
    return true;
}
bool Yolox::PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &rgb_channel_img)
{
    try
    {
        //cv::Mat rgb_img;
        //cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
        cv::resize(bgr_img, bgr_img, cv::Size(config.input_size, config.input_size));
        // cv::Mat rgb_resize_img;
        bgr_img.convertTo(bgr_img,CV_32F);
        // rgb_resize_img = rgb_resize_img/255.0f;
        cv::split(bgr_img,rgb_channel_img);
    }
    catch(const std::exception& e)
    {
        return false;
    }
    return true;
}
bool Yolox::PostProcess(std::vector<std::vector<InstanceInfo>> &output_infos, std::vector<int> height_list, std::vector<int> width_list)
{
    //循环输出的每一个batch的所有框
    //每一个batch送进nms的数据
    bool err;
    std::vector<std::map<int,ClassInfo>> classinfo(config.batch_size);
    std::vector<GridAndStride> grid_strides;
    err = get_anchors(grid_strides);
    float* output = outputs[0];
    for (int batch_index = 0; batch_index < config.batch_size; batch_index++)
    {
        for (int anchor_idx = 0; anchor_idx < grid_strides.size(); anchor_idx++)
        {
            const int grid0 = grid_strides[anchor_idx].grid0;
            const int grid1 = grid_strides[anchor_idx].grid1;
            const int stride = grid_strides[anchor_idx].stride;

            const int basic_pos = (batch_index *  grid_strides.size() + anchor_idx) * (config.item_num + 5 );

            float x_center = (output[basic_pos + 0] + grid0) *stride;
            float y_center = (output[basic_pos + 1] + grid1) *stride;
            float w = exp(output[basic_pos + 2]) * stride;
            float h = exp(output[basic_pos + 3]) * stride;
            float x0 = x_center - w * 0.5f;
            float y0 = y_center - h * 0.5f;

            float box_objectness = output[basic_pos +4];
            for(int class_idx =0; class_idx < config.item_num; class_idx++)
            {
                float box_cls_score = output[basic_pos + 5 + class_idx];
                float box_prob = box_objectness * box_cls_score;
                if (box_prob>config.confthre)
                {
                    cv::Rect rect = cv::Rect(round(x0),round(y0),round(w),round(h));
                    if (classinfo[batch_index].find(class_idx)!=classinfo[batch_index].end())
                    {
                        classinfo[batch_index][class_idx].o_rect.push_back(rect);
                        classinfo[batch_index][class_idx].o_rect_cof.push_back(box_prob);
                    }else
                    {
                        ClassInfo new_class;
                        new_class.o_rect = {rect};
                        new_class.o_rect_cof = {box_prob};
                        classinfo[batch_index].insert(std::pair<int,ClassInfo>(class_idx,new_class));

                    }
                }
                
            }
        }
    }
    //每张图类内做nms，并输出真实框
    err = DetectionUtils::get_instance().NMS(output_infos, classinfo,
                                            height_list, width_list,
                                            config.confthre, config.iouthre,
                                            config.input_size,config.input_size);
    if (!err)
    {
        LOG(ERROR)<<"[Yolox::PostProcess] NMS process failed!";
        return false;
    }
    return true;

}
bool Yolox::get_anchors(std::vector<GridAndStride> &grid_strides)
{
    for(auto stride : strides)
    {
        int num_grid = config.input_size / stride;
        for(int g1 = 0; g1 < num_grid; g1++)
        {
            for(int g0 = 0; g0 < num_grid; g0++)
            {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
    return true;
}


