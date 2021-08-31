#pragma once
#include <string>
#include <map>
#include "classification_interface.h"
#include "detection_config.h"
using namespace std;
typedef void* (*register_func)();

class Store
{
public:
    static Store* getInstance();
    void* findInstance(const string& class_name);
    void registerInstance(const string& class_name, register_func func);
    void get_keys();
    //分类
    classify_config FindClassifyConfig(const string& config_name);
    void RegisterClassifyConfig(const string& config_name, classify_config config);
    void GetClassifyConfigKeys();
    //检测
    detection_config FindDetectionConfig(const string& config_name);
    void RegisterDetectionConfig(const string& config_name, classify_config config);
    void GetDetectionConfigKeys();

    
private:
    map<string, register_func> m_register;
    map<string, classify_config> m_register_classify;
    map<string, detection_config> m_register_detection;
    static Store* instance;
    
};



