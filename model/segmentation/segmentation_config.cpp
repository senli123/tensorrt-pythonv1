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
