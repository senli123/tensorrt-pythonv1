
// #include "segmentation_engine.h"
// REGISTER_CLASS(SegmentationCommon)
// REGISTER_SEG_CONFIG(unet_config)

// bool SegmentationEngine::init(std::string model_name,std::string config_name)
// {   //model初始化
//     bool err;
//     std::cout<<config_name<<std::endl;
//     model= SegmentationFactory::createSegmentationInterface(model_name);
//     if (model == nullptr)
//     {
//         std::cout << "input model_name not in :" << std::endl;
//         Store::getInstance()->get_keys();
//         return false;
//     }
//     //log初始化
//     if(!InitLogger(config_name.c_str()))
//     {
//         return false;
//     }
// #ifdef RELEASE
//     segmentation_config config;
//     err =  ParseConfig(config_name, config);
//     if (!err)
//     {
//         LOG(ERROR)<<"[SegmentationEngine::init]ParseConfig process failed!";
//         return false;
//     }
//     err = model->Model_build(config);
// #else
//     segmentation_config config= SegmentationConfigFactory::createStruct(config_name);
//     if (config.input_size == 0)
//     {
//         std::cout << "input config_name not in :" << std::endl;
//         Store::getInstance()->GetDetectionConfigKeys();
//         return false;
//     }
//     err = model->Model_build(config);
//     utils.init(model_name);
// #endif
//     if (!err)
//     {
//         LOG(ERROR)<<"[SegmentationEngine::init] build failed!";
//         return false;
//     }
//     return true;   
// }
// bool SegmentationEngine::run(std::vector<cv::Mat> &img)
// {
    
//     bool err;
//     cv::Mat mask_img(img.rows, img.cols, CV_32F);
//     Debug_utils::set_time(START);
//     err = model->Model_infer(img,mask_img);
//     if (!err)
//     {
//         LOG(ERROR)<<"[DetectionEngine::run]Infer failed!";
//         return false;
//     }
//     Debug_utils::set_time(END);
//     //可视化mask图
//     std::string path = "./mask.bmp";
//     Utils::get_instance().printMask(mask_img, path);
//     #ifdef RELEASE
//     #else
//         utils.save_time_path();
//         utils.mean_time();
//     #endif
//     return true;
// }
// bool SegmentationEngine::uninit()
// {
//     delete model;
//     return true;
// }
// bool SegmentationEngine::ParseConfig(std::string config_name, segmentation_config &config)
// {
//     try
//     {
//         long flag = ConfigOperator::getIns().init();
//         if (flag!=1)
//         {
//             LOG(ERROR)<<"config path is fault!";
//             return false;
//         }
        
//         std::string seg_bin_path;
//         std::string seg_param_path; 
//         std::string seg_input_name;
//         std::string seg_output_name;
//         int seg_input_size;
//         float seg_meanVals[3];
//         float seg_normVals[3];
//         int seg_class_num;
//         float seg_confthre;
       

//         if( ""!= ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_BIN_PATH))
//         {
//             seg_bin_path = ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_BIN_PATH).c_str();
//         }else{
//             LOG(ERROR)<<"parse bin path failed!";
//             return false;
//         }
//         if( ""!= ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_PARAM_PATH))
//         {
//             seg_param_path = ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_PARAM_PATH);
//         }else{
//             LOG(ERROR)<<"parse param path failed!";
//             return false;
//         }
//         if( ""!= ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_INPUT_NAME))
//         {
//             seg_input_name = ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_INPUT_NAME).c_str();
//         }else{
//             LOG(ERROR)<<"parse input name failed!";
//             return false;
//         }
//         if( ""!= ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_OUTPUT_NAME))
//         {
//             seg_output_name = ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_OUTPUT_NAME).c_str();
//         }else{
//             LOG(ERROR)<<"parse output name failed!";
//             return false;
//         }
//         if( ""!= ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_SIZE))
//         {
//             seg_input_size = atoi(ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_SIZE).c_str());
//         }else{
//             LOG(ERROR)<<"parse input size failed!";
//             return false;
//         }
        
//         //解析均值方差
//         std::string temp_meanVals;
//         if( ""!= ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_MEAMVALS))
//         {
//             temp_meanVals = ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_MEAMVALS);
//         }else{
//             LOG(ERROR)<<"parse meanVals failed!";
//             return false;
//         }
//         std::vector<std::string> meanVals_list=Utils::get_instance().split(temp_meanVals,",");
//         if(meanVals_list.size()<3)
//         {
//             LOG(ERROR)<<"the num of meanVals < 3!";
//             return false;
//         }
//         for (int index = 0; index < 3; index++)
//         {
//             seg_meanVals[index] = atof(meanVals_list[index].c_str());
//         }
//         std::string temp_normVals;
//         if( ""!= ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_NORMVALS))
//         {
//             temp_normVals = ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_NORMVALS);
//         }else{
//             LOG(ERROR)<<"parse normVals failed!";
//             return false;
//         }
//         std::vector<std::string> normVals_list=Utils::get_instance().split(temp_normVals,",");
//         if(normVals_list.size()<3)
//         {
//             LOG(ERROR)<<"the num of normVals < 3!";
//             return false;
//         }
//         for (int index = 0; index < 3; index++)
//         {
//             seg_normVals[index] = atof(normVals_list[index].c_str());
//         }
//         if( ""!= ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_ITEM_NUM))
//         {
//             seg_class_num = atoi(ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_ITEM_NUM).c_str());
//         }else{
//             LOG(ERROR)<<"parse class_num failed!";
//             return false;
//         }
//         if( ""!= ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_CONFTHRE))
//         {
//             seg_confthre = atof(ConfigOperator::getIns().getValue(CFG_SEGMENTATION, config_name + CFG_CONFTHRE).c_str());
//         }else{
//             LOG(ERROR)<<"parse confthre failed!";
//             return false;
//         }
        

//         strcpy(config.param_path,seg_param_path.c_str());
//         strcpy(config.bin_path,seg_bin_path.c_str());
//         strcpy(config.input_name,seg_input_name.c_str());
//         strcpy(config.output_name,seg_output_name.c_str());
//         config.input_size = seg_input_size;
//         memcpy(config.meanVals, seg_meanVals, sizeof(seg_meanVals));
//         memcpy(config.normVals, seg_normVals, sizeof(seg_normVals));
//         config.class_num = seg_class_num;
//         config.confthre = seg_confthre;
//     }
//     catch(...)
//     {
//         return false;
//     }
//     return true;
// }


