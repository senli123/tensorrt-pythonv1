#include "detection_engine.h"

REGISTER_CLASS(Yolov5)
REGISTER_DETE_CONFIG(yolov5_config)

bool DetectionEngine::init(std::string model_name,std::string config_name)
{   //model初始化
    bool err;
    std::cout<<config_name<<std::endl;
    model= DetectionFactory::createDetectionInterface(model_name);
    if (model == nullptr)
    {
        std::cout << "input model_name not in :" << std::endl;
        Store::getInstance()->get_keys();
        return false;
    }
    //log初始化
    if(!InitLogger(config_name.c_str()))
    {
        return false;
    }
#ifdef RELEASE
    detection_config config;
    err =  ParseConfig(config_name, config);
    if (!err)
    {
        LOG(ERROR)<<"[DetectionEngine::init]ParseConfig process failed!";
        return false;
    }
    err = model->Model_build(config);
#else
    detection_config config= DetectionConfigFactory::createStruct(config_name);
    if (config.input_size == 0)
    {
        std::cout << "input config_name not in :" << std::endl;
        Store::getInstance()->GetDetectionConfigKeys();
        return false;
    }
    err = model->Model_build(config);
    utils.init(model_name);
#endif
    if (!err)
    {
        LOG(ERROR)<<"[DetectionEngine::init] build failed!";
        return false;
    }
    return true;   
}
bool DetectionEngine::run(std::vector<cv::Mat> &imgs)
{
    
    bool err;
    std::vector<std::vector<InstanceInfo>> output_infos;
    Debug_utils::set_time(START);
    err = model->Model_infer(imgs,output_infos);
    if (!err)
    {
        LOG(ERROR)<<"[DetectionEngine::run]Infer failed!";
        return false;
    }
    Debug_utils::set_time(END);
    for (size_t i = 0; i < output_infos.size(); i++)
    {
        std::vector<InstanceInfo> outputinfo = output_infos[i];
        for(size_t j = 0; j < outputinfo.size(); j++)
        {
            InstanceInfo instance = outputinfo[j];
            LOG(INFO)<<"instance id :"<<instance.class_id;
            LOG(INFO)<<"instance score :"<<instance.score;
            LOG(INFO)<<"instance bbox :"<<instance.rect.x <<" "<<instance.rect.y<<" "<<instance.rect.width<<" "<<instance.rect.height;
        }
        std::string path = "./" + std::to_string(i) + ".bmp";
        Utils::get_instance().printBbox(imgs.at(i), outputinfo, path);
    }
    #ifdef RELEASE
    #else
        utils.save_time_path();
        utils.mean_time();
    #endif
    return true;
}
bool DetectionEngine::uninit()
{
    delete model;
    return true;
}
bool DetectionEngine::ParseConfig(std::string config_name, detection_config &config)
{
    try
    {
        long flag = ConfigOperator::getIns().init();
        if (flag!=1)
        {
            LOG(ERROR)<<"config path is fault!";
            return false;
        }
        std::string cla_onnx_path; 
        std::string cla_bin_path;
        std::string cla_input_name;
        std::string cla_output_name;
        int cla_cuda_id;
        int cla_input_size;
        int cla_batch_size;
        float cla_meanVals[3];
        float cla_normVals[3];
        int cla_class_num;
        int cla_output_num;
        bool cla_FP16;
        bool cla_INT8;


        if( ""!= ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_ONNX_PATH))
        {
            cla_onnx_path = ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_ONNX_PATH);
        }else{
            LOG(ERROR)<<"parse onnx path failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_BIN_PATH))
        {
            cla_bin_path = ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_BIN_PATH).c_str();
        }else{
            LOG(ERROR)<<"parse bin path failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_INPUT_NAME))
        {
            cla_input_name = ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_INPUT_NAME).c_str();
        }else{
            LOG(ERROR)<<"parse input name failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_OUTPUT_NAME))
        {
            cla_output_name = ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_OUTPUT_NAME).c_str();
        }else{
            LOG(ERROR)<<"parse output name failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_CUDA_ID))
        {
            cla_cuda_id = atoi(ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_CUDA_ID).c_str());
        }else{
            LOG(ERROR)<<"parse input size failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_INPUT_SIZE))
        {
            cla_input_size = atoi(ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_INPUT_SIZE).c_str());
        }else{
            LOG(ERROR)<<"parse input size failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_BATCH_SIZE))
        {
            cla_batch_size = atoi(ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_BATCH_SIZE).c_str());
        }else{
            LOG(ERROR)<<"parse input size failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_BATCH_SIZE))
        {
            cla_batch_size = atoi(ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_BATCH_SIZE).c_str());
        }else{
            LOG(ERROR)<<"parse batch size failed!";
            return false;
        }
        
        //解析均值方差
        std::string temp_meanVals;
        if( ""!= ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_MEAMVALS))
        {
            temp_meanVals = ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_MEAMVALS);
        }else{
            LOG(ERROR)<<"parse meanVals failed!";
            return false;
        }
        std::vector<std::string> meanVals_list=Utils::get_instance().split(temp_meanVals,",");
        if(meanVals_list.size()<3)
        {
            LOG(ERROR)<<"the num of meanVals < 3!";
            return false;
        }
        for (int index = 0; index < 3; index++)
        {
            cla_meanVals[index] = atof(meanVals_list[index].c_str());
        }
        std::string temp_normVals;
        if( ""!= ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_NORMVALS))
        {
            temp_normVals = ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_NORMVALS);
        }else{
            LOG(ERROR)<<"parse normVals failed!";
            return false;
        }
        std::vector<std::string> normVals_list=Utils::get_instance().split(temp_normVals,",");
        if(normVals_list.size()<3)
        {
            LOG(ERROR)<<"the num of normVals < 3!";
            return false;
        }
        for (int index = 0; index < 3; index++)
        {
            cla_normVals[index] = atof(normVals_list[index].c_str());
        }
        
        if( ""!= ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_CLASS_NUM))
        {
            cla_class_num = atoi(ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_CLASS_NUM).c_str());
        }else{
            LOG(ERROR)<<"parse class_num failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_TOP_NUM))
        {
            cla_output_num = atoi(ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_TOP_NUM).c_str());
        }else{
            LOG(ERROR)<<"parse output_num failed!";
            return false;
        }
        int temp_fp16,temp_int8;
        if( ""!= ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_FP16))
        {
            temp_fp16 = atoi(ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_FP16).c_str());
        }else{
            LOG(ERROR)<<"parse fp16 failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_INT8))
        {
            temp_int8 = atoi(ConfigOperator::getIns().getValue(CFG_CLASSIFICATION, config_name+CFG_INT8).c_str());
        }else{
            LOG(ERROR)<<"parse int8 failed!";
            return false;
        }
        if (temp_fp16 ==1 and temp_int8==1)
        {
            LOG(ERROR)<<"check the settings of fp16 & int8 !";
            return false;
        }
        cla_FP16 = temp_fp16;
        cla_INT8=temp_int8;
        strcpy(config.onnx_path,cla_onnx_path.c_str());
        strcpy(config.bin_path,cla_bin_path.c_str());
        strcpy(config.input_name,cla_input_name.c_str());
        strcpy(config.output_name,cla_output_name.c_str());
        config.cuda_id = cla_cuda_id;
        config.input_size = cla_input_size;
        config.batch_size = cla_batch_size;
        memcpy(config.meanVals,cla_meanVals,sizeof(cla_meanVals));
        memcpy(config.normVals,cla_normVals,sizeof(cla_normVals));
        config.class_num = cla_class_num;
        config.output_num = cla_output_num;
        config.FP16 = cla_FP16;
        config.INT8 = cla_INT8;
    }
    catch(...)
    {
        return false;
    }
    return true;
}

