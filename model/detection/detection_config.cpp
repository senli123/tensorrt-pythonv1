#include "detection_config.h"
const detection_config yolov5_config = 
{
    YOLOV5_ONNXFILENAME,
    YOLOV5_BINFILENAME, 
    YOLOV5_INPUTTENSORNAMES, 
    YOLOV5_OUTPUTTENSORNAMES, 
    YOLOV5_CUDAID,            
    YOLOV5_INPUT_SIZE,
    YOLOV5_BATCH_SIZE, 
    YOLOV5_meanVals,
    YOLOV5_normVals, 
    YOLOV5_ITEM_NUM, 
    YOLOV5_CONFTHRE, 
    YOLOV5_IOUTHRE,  
    YOLOV5_NET_GRID,  
    YOLOV5_ANCHOR_NUM, 
    YOLOV5_FP16, 
    YOLOV5_INT8, 
};