#include "detection_engine.h"

REGISTER_CLASS(Yolov5)
REGISTER_CLASS(Yolox)
REGISTER_CLASS(CenterNet)
REGISTER_DETE_CONFIG(yolox_config)
REGISTER_DETE_CONFIG(centernet_config)
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
        std::string path = "./123" + std::to_string(i) + ".bmp";
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
        std::string dete_onnx_path; 
        std::string dete_bin_path;
        std::string dete_input_name;
        std::string dete_output_name;
        int dete_cuda_id;
        int dete_input_size;
        int dete_batch_size;
        float dete_meanVals[3];
        float dete_normVals[3];
        int dete_item_num;
        float dete_confthre;
        float dete_iouthre;
        int dete_net_grid[3];
        int dete_anchor_num;
        bool dete_FP16;
        bool dete_INT8;


        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_ONNX_PATH))
        {
            dete_onnx_path = ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_ONNX_PATH);
        }else{
            LOG(ERROR)<<"parse onnx path failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_BIN_PATH))
        {
            dete_bin_path = ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_BIN_PATH).c_str();
        }else{
            LOG(ERROR)<<"parse bin path failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_INPUT_NAME))
        {
            dete_input_name = ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_INPUT_NAME).c_str();
        }else{
            LOG(ERROR)<<"parse input name failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_OUTPUT_NAME))
        {
            dete_output_name = ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_OUTPUT_NAME).c_str();
        }else{
            LOG(ERROR)<<"parse output name failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_CUDA_ID))
        {
            dete_cuda_id = atoi(ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_CUDA_ID).c_str());
        }else{
            LOG(ERROR)<<"parse cuda id failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_INPUT_SIZE))
        {
            dete_input_size = atoi(ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_INPUT_SIZE).c_str());
        }else{
            LOG(ERROR)<<"parse input size failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_BATCH_SIZE))
        {
            dete_batch_size = atoi(ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_BATCH_SIZE).c_str());
        }else{
            LOG(ERROR)<<"parse batch size failed!";
            return false;
        }
        
        //解析均值方差
        std::string temp_meanVals;
        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_MEAMVALS))
        {
            temp_meanVals = ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_MEAMVALS);
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
            dete_meanVals[index] = atof(meanVals_list[index].c_str());
        }
        std::string temp_normVals;
        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_NORMVALS))
        {
            temp_normVals = ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_NORMVALS);
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
            dete_normVals[index] = atof(normVals_list[index].c_str());
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_ITEM_NUM))
        {
            dete_item_num = atoi(ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_ITEM_NUM).c_str());
        }else{
            LOG(ERROR)<<"parse item_num failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_CONFTHRE))
        {
            dete_confthre = atof(ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_CONFTHRE).c_str());
        }else{
            LOG(ERROR)<<"parse confthre failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_IOUTHRE))
        {
            dete_iouthre = atof(ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_IOUTHRE).c_str());
        }else{
            LOG(ERROR)<<"parse iouthre failed!";
            return false;
        }
        //解析net_grid
        std::string temp_net_grid;
        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_NET_GRID))
        {
            temp_net_grid = ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_NET_GRID);
        }else{
            LOG(ERROR)<<"parse net_grid failed!";
            return false;
        }
        std::vector<std::string> net_grid_list=Utils::get_instance().split(temp_net_grid,",");
        for (int index = 0; index < net_grid_list.size(); index++)
        {
            dete_net_grid[index] = atoi(net_grid_list[index].c_str());
        }

        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_ANCHOR_NUM))
        {
            dete_anchor_num = atoi(ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_ANCHOR_NUM).c_str());
        }else{
            LOG(ERROR)<<"parse anchor_num failed!";
            return false;
        }
        int temp_fp16,temp_int8;
        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_FP16))
        {
            temp_fp16 = atoi(ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_FP16).c_str());
        }else{
            LOG(ERROR)<<"parse fp16 failed!";
            return false;
        }
        if( ""!= ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_INT8))
        {
            temp_int8 = atoi(ConfigOperator::getIns().getValue(CFG_DETECTION, config_name+CFG_INT8).c_str());
        }else{
            LOG(ERROR)<<"parse int8 failed!";
            return false;
        }
        if (temp_fp16 ==1 and temp_int8==1)
        {
            LOG(ERROR)<<"check the settings of fp16 & int8 !";
            return false;
        }
        dete_FP16 = temp_fp16;
        dete_INT8=temp_int8;

        strcpy(config.onnx_path,dete_onnx_path.c_str());
        strcpy(config.bin_path,dete_bin_path.c_str());
        strcpy(config.input_name,dete_input_name.c_str());
        strcpy(config.output_name,dete_output_name.c_str());
        config.cuda_id = dete_cuda_id;
        config.input_size = dete_input_size;
        config.batch_size = dete_batch_size;
        memcpy(config.meanVals, dete_meanVals, sizeof(dete_meanVals));
        memcpy(config.normVals, dete_normVals, sizeof(dete_normVals));
        config.item_num = dete_item_num;
        config.confthre = dete_confthre;
        config.iouthre = dete_iouthre;
        memcpy(config.net_grid, dete_net_grid, sizeof(dete_net_grid));
        config.anchor_num = dete_anchor_num;
        config.FP16 = dete_FP16;
        config.INT8 = dete_INT8;
    }
    catch(...)
    {
        return false;
    }
    return true;
}

