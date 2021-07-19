#pragma once
#include "common.h"
#include "Infer_data_def.h"
#include <opencv2/opencv.hpp>
class TensorRT_Interface
{
public:
    TensorRT_Interface(const TensorRT_data &Tparams);
    bool build();
    bool processInput(std::vector<cv::Mat> &Batch_rbg_img);
    bool infer();
    template <typename T>
    bool verifyOutput(std::vector<T> &result_list);
    void img_size_clear();
    ~TensorRT_Interface()
    {
        this->engine->destroy();
        
        
    }
public:
    TensorRT_data Tparams;
    Logger global_logger;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
    BufferControl buffer;
    float *hostDataBuffer;
    float *output;
    int img_size = 0;


};
template <typename T>
bool TensorRT_Interface::verifyOutput(std::vector<T> &result_list)
{
    return true;
}