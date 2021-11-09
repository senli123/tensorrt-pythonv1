#include "segmentation_config.h"
const segmentation_config unet_config = 
{
    UNET_ONNX_PATH,
    UNET_BIN_PATH,  
    UNET_INPUTTENSORNAMES, 
    UNET_OUTPUTTENSORNAMES, 
    UNET_CUDAID,   
    UNET_INPUT_SIZE,
    UNET_BATCHSIZE,
    UNET_meanVals, 
    UNET_normVals,
    UNET_CLASS_NUM, 
    UNET_CONFTHRE,  
    UNET_FP16, 
    UNET_INT8, 
};
const segmentation_config refinenet_config = 
{
    REFINENET_ONNX_PATH,
    REFINENET_BIN_PATH,  
    REFINENET_INPUTTENSORNAMES, 
    REFINENET_OUTPUTTENSORNAMES, 
    REFINENET_CUDAID,   
    REFINENET_INPUT_SIZE,
    REFINENET_BATCHSIZE,
    REFINENET_meanVals, 
    REFINENET_normVals,
    REFINENET_CLASS_NUM, 
    REFINENET_CONFTHRE,  
    REFINENET_FP16, 
    REFINENET_INT8, 
};

