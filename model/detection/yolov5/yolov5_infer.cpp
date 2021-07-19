#include "yolov5_infer.h"
long yolov5::Model_Build()
{
    if ( TensorRT_Interface::build())
    {
        return 0;
    }
    else
    {
        printf("yolov5 Model_Build fail !");
        return -1;
    }
}

long yolov5::Model_Infer(std::vector<cv::Mat> &bgr_imgs,std::vector<std::vector<InstanceInfo>> &defect_info)
{
    if (bgr_imgs.size() != YOLOV5_BATCHSIZE)
    {
        printf("the size of input vector must be equal to batch_size !");
        return -1;
    }
    defect_info.clear();
    TensorRT_Interface::img_size_clear();
    int img_width = bgr_imgs[0].cols;
    int img_height = bgr_imgs[0].rows;
    for(int idx = 0; idx <bgr_imgs.size(); ++idx)
    {
        //初始化预处理后的图片
        cv::Mat After_Handle_img_R(YOLOV5_RESIZE, YOLOV5_RESIZE, CV_32FC1, cv::Scalar(0, 0, 0));
        cv::Mat After_Handle_img_G(YOLOV5_RESIZE, YOLOV5_RESIZE, CV_32FC1, cv::Scalar(0, 0, 0));
        cv::Mat After_Handle_img_B(YOLOV5_RESIZE, YOLOV5_RESIZE, CV_32FC1, cv::Scalar(0, 0, 0));
        std::vector<cv::Mat> After_Handle_img = {
            After_Handle_img_R,After_Handle_img_G,After_Handle_img_B
		};
        //图像预处理
        bool Preprocess_status = PreProcess(bgr_imgs[idx], After_Handle_img, img_width, img_height); //图像预处理
        if(!Preprocess_status)
        {
            printf("yolov5 Preprocess fail !");
            return -1;
        }
        if (!TensorRT_Interface::processInput(After_Handle_img)) //在buffer中加入数据
		{
			printf("yolov5 processInput fail !");
			return -1;
		}
    }
    if (!TensorRT_Interface::infer())  //对图片进行推断
	{
			printf("yolov5 infer fail !");
			return -1;
	}
	if (!verifyOutput(defect_info,img_width, img_height))   //后处理
	{
		printf("yolov5 verifyOutput fail !");
		return -1;
	}
    return 0;
    

}
/*
 *-----------------private------------------------
 */
bool yolov5::PreProcess(cv::Mat& bgr_img, std::vector<cv::Mat> &After_Handle_img,
                    int &img_width, int &img_height)
{
    return yolov5_utils.PreProcess(bgr_img, After_Handle_img, img_width, img_height);
}
bool yolov5::verifyOutput(std::vector<std::vector<InstanceInfo>> &defect_info,int &img_width,int &img_height)
{
    return yolov5_utils.PostProcess(this->output, defect_info,img_width, img_height);
} 
