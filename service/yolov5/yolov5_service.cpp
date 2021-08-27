// #include "yolov5_service.h"

// long  Dete_Service::Init()
// {
//     if (0 != yolo.Model_Build())
//     {
//         printf("detection init fail !");\
//         return -1;
//     }
//     return 0;
    
// }
// long  Dete_Service::Run()
// {
//     std::string path = "/workspace/lisen/_bushu/data/1.bmp";
//     cv::Mat img = cv::imread(path);
//     std::vector<cv::Mat> input; 
//     std::vector<std::vector<InstanceInfo>> output;
//     input.push_back(img);
//     long detection_status = yolo.Model_Infer(input,output);
//     if (0 !=detection_status)
// 	{
// 		std::cout << "Detection failed !" << std::endl;
// 		return -1;
// 	}
//     for (int i = 0; i < output.size(); i++)
//     {
//         std::vector<InstanceInfo> img_info =  output[i];
//         for (int j = 0; j < img_info.size(); j++)
//         {
//             std::cout<<img_info[j].x1<<" ";
//             std::cout<<img_info[j].y1<<" ";
//             std::cout<<img_info[j].x2<<" ";
//             std::cout<<img_info[j].y2<<" ";
//             std::cout<<img_info[j].class_id<<" ";
//             std::cout<<img_info[j].score<<std::endl;
//         }
        
//     }
//     return 0;
    
// }