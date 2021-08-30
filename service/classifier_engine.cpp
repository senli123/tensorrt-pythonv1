#include "classifier_engine.h"
REGISTER_CLASS(ClassificationCommon)
//REGISTER_CLASS(Squeezenet)
REGISTER_CLA_CONFIG(mobilenetv2_config)
REGISTER_CLA_CONFIG(alexnet_config)
//REGISTER_CLA_CONFIG(squeezenet_config)
std::vector<std::string> split(std::string str, std::string pattern)
{
    std::string::size_type pos; 
    std::vector<std::string> result;
    str += pattern;
    int size = str.size();
    for (int i = 0; i < size; i++)
    {
        pos = str.find(pattern, i);
        if (pos<size)
        {
            std::string s = str.substr(i, pos-i);
            result.push_back(s);
            i = pos + pattern.size()-1;
        }
        
    }
    return result;
}
bool ClassifierEngine::init(std::string model_name,std::string config_name)
{   //model初始化
    bool err;
    std::cout<<config_name<<std::endl;
    model= ClassifierFactory::createClassifierInterface(model_name);
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
    classify_config config;
    err =  ParseConfig(config_name, config);
    if (!err)
    {
        LOG(ERROR)<<"[ClassifierEngine::init]ParseConfig process failed!";
        return false;
    }
    err = model->Model_build(config);
#else
    classify_config config= ClassifyConfigFactory::createStruct(config_name);
    if (config.input_size == 0)
    {
        std::cout << "input config_name not in :" << std::endl;
        Store::getInstance()->GetClassifyConfigKeys();
        return false;
    }
    err = model->Model_build(config);
    utils.init(model_name);
#endif
    if (!err)
    {
        LOG(ERROR)<<"build failed!";
        return false;
    }
     return true;   
}
bool ClassifierEngine::run(std::vector<cv::Mat> &imgs)
{
    bool err;
    std::vector<image_info> outputinfos;
    Debug_utils::set_time(START);
    err = model->Model_infer(imgs,outputinfos);
    if (!err)
    {
       LOG(ERROR)<<"Infer failed!";
        return false;
    }
    Debug_utils::set_time(END);
    for (size_t i = 0; i < outputinfos.size(); i++)
    {
        image_info outputinfo = outputinfos[i];
        LOG(INFO)<<"class index :"<<outputinfo.indexs.at(0);
        LOG(INFO)<<"class score :"<<outputinfo.scores.at(0);
    }
    
    #ifdef RELEASE
    #else
        utils.save_time_path();
        utils.mean_time();
    #endif
    return true;
}
bool ClassifierEngine::uninit()
{
    delete model;
    return true;
}
bool ClassifierEngine::ParseConfig(std::string config_name, classify_config &config)
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
        std::vector<std::string> meanVals_list=split(temp_meanVals,",");
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
        std::vector<std::string> normVals_list=split(temp_normVals,",");
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

