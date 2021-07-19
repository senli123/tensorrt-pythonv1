#include "tensorRT.h"
#include "yolov5_utils.h"
#include "yolov5_data.h"
class yolov5 :public TensorRT_Interface
{
public:
    yolov5():TensorRT_Interface(Params_Init::YOLOV5_tensorRT_data)
    {};
    //模型构建
    long Model_Build();
    //模型推断
    long Model_Infer(std::vector<cv::Mat> &bgr_imgs,std::vector<std::vector<InstanceInfo>> &defect_info);
private:
    //图像预处理
    bool PreProcess(cv::Mat &bgr_img, std::vector<cv::Mat> &After_Handle_img,
                    int &img_width, int &img_height);
    bool verifyOutput(std::vector<std::vector<InstanceInfo>> &defect_info, int &img_width,int &img_height); 
private:
    Yolov5_Utils yolov5_utils;
};