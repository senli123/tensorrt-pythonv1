#pragma once
#include "common.h"
#include "Infer_data_def.h"
#include <opencv2/opencv.hpp>
#include "my_log.h"
class TensorRT_Interface
{
public:
    bool build(const TensorRT_data &Tparams);
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
    std::vector<float*> outputs;
    int img_size = 0;


};
template <typename T>
bool TensorRT_Interface::verifyOutput(std::vector<T> &result_list)
{
    return true;
}