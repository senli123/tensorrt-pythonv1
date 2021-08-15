#include <iostream>
// #include <opencv2/highgui.hpp>
// #include <opencv2/opencv.hpp>
// #include "NvInfer.h"
// #include <cuda_runtime.h>
#include "yolov5_service.h"
int main(int, char**) {
    // int a =10;
    // int b = 2;
    // int c = a+b;
    //  std::cout <<c;
    // std::cout << "Hello, world!\n";
    Dete_Service dete;
    //LJ_Service dete;
    long Init_status = dete.Init();
    if (0 != Init_status)
    {
        printf("init fail");
        return -1;
    }
    for (size_t i = 0; i < 10; i++)
    {
        long Infer_status = dete.Run();
        if (0 != Infer_status )
        {
            printf("run fail");
            return -1;
        }
    }
    return 0;
    //std::cout<< atof("")<<std::endl;
    //return 0;
}
