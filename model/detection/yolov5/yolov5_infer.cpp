#include "yolov5_infer.h"
const std::vector<int> grid = {80,40,20};
 //模型构建
bool Yolov5::Model_build(const detection_config &input_config)
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
        LOG(INFO)<<"[Yolov5::Model_build] build process succeed!";
        return true;
    }
    else {
        LOG(ERROR)<<"[Yolov5::Model_build] build process failed!";
        return false;
    }
}
//模型推断
bool Yolov5::Model_infer(std::vector<cv::Mat> &bgr_imgs,std::vector<std::vector<InstanceInfo>> &output_infos)
{
    output_infos.clear();
    TensorRT_Interface::img_size_clear();
    std::vector<int> height_list;
    std::vector<int> width_list;
    LOG(INFO)<<"[Yolov5::Model_infer] PreProcess start!";
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
            LOG(ERROR)<<"[Yolov5::Model_infer] PreProcess failed!";
            std::ostringstream ostr;
		    ostr << "the index of error image: " << batch;
		    LOG(ERROR)<<ostr.str().c_str();
            return false;
            
        }
        if (!TensorRT_Interface::processInput(rgb_channel_img)) //在buffer中加入数据
		{
            LOG(ERROR)<<"[Yolov5::Model_infer] processInput process failed!";
            std::ostringstream ostr;
		    ostr << "the index of error image: " << batch;
		    LOG(ERROR)<<ostr.str().c_str();
			return false;
		}
    }
    LOG(INFO)<<"[Yolov5::Model_infer] PreProcess succeed!";
    Debug_utils::set_time(PREPROCESS);
    LOG(INFO)<<"[Yolov5::Model_infer] infer start!";
    if (!TensorRT_Interface::infer())  //对图片进行推断
	{
		LOG(ERROR)<<"[Yolov5::Model_infer] infer process failed!";
        return false;
	}
    LOG(INFO)<<"[Yolov5::Model_infer] infer succeed!";
    Debug_utils::set_time(INFER);
    LOG(INFO)<<"[Yolov5::Model_infer] Postprocess start!";
    if (!PostProcess(output_infos,height_list,width_list))
    {
        LOG(ERROR)<<"[Yolov5::Model_infer] Postprocess failed!";
		return false;
    }
    LOG(INFO)<<"[Yolov5::Model_infer] Postprocess succeed!";
    Debug_utils::set_time(POSTPROCESS);
    return true;
}
bool Yolov5::PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &rgb_channel_img)
{
    try
    {
        cv::Mat rgb_img;
        cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
        cv::resize(rgb_img, rgb_img, cv::Size(config.input_size, config.input_size));
        cv::Mat rgb_resize_img;
        rgb_img.convertTo(rgb_resize_img,CV_32F);
        rgb_resize_img = rgb_resize_img/255.0f;
        cv::split(rgb_resize_img,rgb_channel_img);
    }
    catch(const std::exception& e)
    {
        return false;
    }
    return true;
}
bool Yolov5::PostProcess(std::vector<std::vector<InstanceInfo>> &output_infos, std::vector<int> height_list, std::vector<int> width_list)
{
    //循环每一个输出（即各种分辨率下），并得到该分辨率下的anchor
    //每一个batch送进nms的数据
    bool err;
    std::vector<std::map<int,ClassInfo>> classinfo(config.batch_size);
    int scale_index = 0;
    for ( auto &output : outputs)
    {
        int start_index=0;
        int grid = config.net_grid[scale_index];
        std::vector<int> anchors(6);
        err = get_anchors(grid,anchors);
        if (!err)
        {
            return false;
        }
        for (int batch_index = 0; batch_index < config.batch_size; batch_index++)
        {
            for (int anchor_index = 0; anchor_index < config.anchor_num; anchor_index++)
            {
                for (int y_grid = 0; y_grid < grid; y_grid++)
                {
                    for (int x_grid = 0; x_grid < grid; x_grid++)
                    {
                        double box_prob = output[start_index + 4];
                        box_prob = DetectionUtils::get_instance().sigmoid(box_prob);
                        if (box_prob<config.confthre)
                        {
                            start_index +=config.item_num;
                            continue;
                        }
                        double x =output[start_index + 0];
                        double y =output[start_index + 1];
                        double w =output[start_index + 2];
                        double h =output[start_index + 3];
                        double max_prob = 0;
                        int idx =0;
                        for (int class_index = 5; class_index < config.item_num; class_index++)
                        {
                            double tp = output[start_index + class_index];
                            tp = DetectionUtils::get_instance().sigmoid(tp);
                            if (tp>max_prob)
                            {
                                max_prob = tp;
                                idx = class_index -5;
                            }
                            
                        }
                        float cof = box_prob * max_prob;
                        //对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
                        if(cof<config.confthre)
                        {
                            start_index +=config.item_num;
                            continue;
                        }
                        x = ( DetectionUtils::get_instance().sigmoid(x)*2 - 0.5 + x_grid) * config.input_size / grid;
                        y = ( DetectionUtils::get_instance().sigmoid(y)*2 - 0.5 + y_grid) * config.input_size / grid;
                        w = pow(DetectionUtils::get_instance().sigmoid(w)*2, 2) * anchors[anchor_index *2];
                        h = pow(DetectionUtils::get_instance().sigmoid(h)*2, 2) * anchors[anchor_index *2 + 1];
                        double r_x = x - w/2;
                        double r_y = y - h/2;
                        cv::Rect rect = cv::Rect(round(r_x),round(r_y),round(w),round(h));
                        if (classinfo[batch_index].find(idx)!=classinfo[batch_index].end())
                        {
                            classinfo[batch_index][idx].o_rect.push_back(rect);
                            classinfo[batch_index][idx].o_rect_cof.push_back(cof);
                        }else{
                            ClassInfo new_class;
                            new_class.o_rect = {rect};
                            new_class.o_rect_cof = {cof};
                            classinfo[batch_index].insert(std::pair<int,ClassInfo>(idx,new_class));

                        }
                        start_index += config.item_num;
                        
                    }
            
                }
            
            }
            
        }
        scale_index += 1;    
    }
    //每张图类内做nms，并输出真实框
    err = DetectionUtils::get_instance().NMS(output_infos, classinfo,
                                            height_list, width_list,
                                            config.confthre, config.iouthre,
                                            config.input_size,config.input_size);
    if (!err)
    {
        LOG(ERROR)<<"[Yolov5::PostProcess] NMS process failed!";
        return false;
    }
    return true;

}
bool Yolov5::get_anchors(int net_grid,std::vector<int> &anchors)
{
    if (count(grid.begin(),grid.end(),net_grid)==0)
    {
        LOG(ERROR)<<"[Yolov5::get_anchors] net_grid must be 80,40 or 20!";
        return false;
    }
    int a80[6] = {10,13, 16,30, 33,23};
    int a40[6] = {30,61, 62,45, 59,119};
    int a20[6] = {116,90, 156,198, 373,326}; 
    if(net_grid == 80){
        anchors.insert(anchors.begin(),a80,a80 + 6);
    }
    else if(net_grid == 40){
        anchors.insert(anchors.begin(),a40,a40 + 6);
    }
    else if(net_grid == 20){
        anchors.insert(anchors.begin(),a20,a20 + 6);
    }
    return true;
}
// // long yolov5::Model_Build()
// // {
// //     if ( TensorRT_Interface::build())
// //     {
// //         return 0;
// //     }
// //     else
// //     {
// //         printf("yolov5 Model_Build fail !");
// //         return -1;
// //     }
// // }

// // long yolov5::Model_Infer(std::vector<cv::Mat> &bgr_imgs,std::vector<std::vector<InstanceInfo>> &defect_info)
// // {
// //     if (bgr_imgs.size() != YOLOV5_BATCHSIZE)
// //     {
// //         printf("the size of input vector must be equal to batch_size !");
// //         return -1;
// //     }
// //     defect_info.clear();
// //     TensorRT_Interface::img_size_clear();
// //     int img_width = bgr_imgs[0].cols;
// //     int img_height = bgr_imgs[0].rows;
// //     for(int idx = 0; idx <bgr_imgs.size(); ++idx)
// //     {
// //         //初始化预处理后的图片
// //         cv::Mat After_Handle_img_R(YOLOV5_RESIZE, YOLOV5_RESIZE, CV_32FC1, cv::Scalar(0, 0, 0));
// //         cv::Mat After_Handle_img_G(YOLOV5_RESIZE, YOLOV5_RESIZE, CV_32FC1, cv::Scalar(0, 0, 0));
// //         cv::Mat After_Handle_img_B(YOLOV5_RESIZE, YOLOV5_RESIZE, CV_32FC1, cv::Scalar(0, 0, 0));
// //         std::vector<cv::Mat> After_Handle_img = {
// //             After_Handle_img_R,After_Handle_img_G,After_Handle_img_B
// // 		};
// //         //图像预处理
// //         bool Preprocess_status = PreProcess(bgr_imgs[idx], After_Handle_img, img_width, img_height); //图像预处理
// //         if(!Preprocess_status)
// //         {
// //             printf("yolov5 Preprocess fail !");
// //             return -1;
// //         }
// //         if (!TensorRT_Interface::processInput(After_Handle_img)) //在buffer中加入数据
// // 		{
// // 			printf("yolov5 processInput fail !");
// // 			return -1;
// // 		}
// //     }
// //     if (!TensorRT_Interface::infer())  //对图片进行推断
// // 	{
// // 			printf("yolov5 infer fail !");
// // 			return -1;
// // 	}
// // 	if (!verifyOutput(defect_info,img_width, img_height))   //后处理
// // 	{
// // 		printf("yolov5 verifyOutput fail !");
// // 		return -1;
// // 	}
// //     return 0;
    

// // }
// // /*
// //  *-----------------private------------------------
// //  */
// // bool yolov5::PreProcess(cv::Mat& bgr_img, std::vector<cv::Mat> &After_Handle_img,
// //                     int &img_width, int &img_height)
// // {
// //     return yolov5_utils.PreProcess(bgr_img, After_Handle_img, img_width, img_height);
// // }
// // bool yolov5::verifyOutput(std::vector<std::vector<InstanceInfo>> &defect_info,int &img_width,int &img_height)
// // {
// //     return yolov5_utils.PostProcess(this->output, defect_info,img_width, img_height);
// // } 
