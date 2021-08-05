#include <iostream>
#include "mobilenetv2.h"
#include "alglogger.h"
#include "my_log.h"
#include "tools.h"
int main(int, char**) {
    if(!InitLogger("mobilenet"))
    {

        return -1;
    }
    Debug_utils utils;
    utils.init("mobilenet");
    cv::Mat img = cv::imread("/workspace/lisen/_bushu/tensorrt-python/data/img2.jpg");
    if (img.empty())
    {
        LOG(ERROR)<<"img is empty!";
        return -1;
    }
   
    std::vector<cv::Mat> bgr_imgs;
    std::vector<image_info> outputinfo;
    bgr_imgs.push_back(img);
    long err;
    MobilenetV2 mobilenetv2;
    err = mobilenetv2.Model_Build();
    if (!err)
    {
        LOG(ERROR)<<"build failed!";
        return -1;
    }
    Debug_utils::set_time(START);
    err = mobilenetv2.Model_Infer(bgr_imgs,outputinfo);
    if (!err)
    {
        LOG(ERROR)<<"Infer failed!";
        return -1;
    }
    Debug_utils::set_time(END);
    utils.save_time_path();
    utils.mean_time();
    for (size_t i = 0; i < outputinfo.size(); i++)
    {
        image_info info = outputinfo[i];
        std::cout<<"class index :"<<info.indexs.at(0)<<std::endl;
        std::cout<<"class score :"<<info.scores.at(0)<<std::endl;
    }


    // cv::Mat Process_img ;
    //  cv::dnn::blobFromImage(img, Process_img, 0.017 , cv::Size(224, 224), cv::Scalar(103.94, 166.78, 123.68), true, false);
    //  cv::imwrite("./1.bmp",Process_img);
    // return 0;

    
    UnitLogger();
    // LOG(ERROR)<<"[MobilenetV2::Model_Infer] PreProcess failed!";
    //         std::string save_error_path = MOBILENETV2_DEBUG_PATH + std::to_string(error_index) + ".bmp";
    //         error_index +=1;
    //         std::ostringstream ostr;
	// 	    ostr << "the path of error image: " << save_error_path;
    //         try
    //         {
    //             cv::imwrite(save_error_path,bgr_img);
    //         }
    //         catch(...)
    //         {
    //             LOG(ERROR)<<"[MobilenetV2::Model_Infer] save error image failed!";
    //         }
	// 	    LOG(ERROR)<<ostr.str().c_str();
    //         return false;
    return 0;


}
