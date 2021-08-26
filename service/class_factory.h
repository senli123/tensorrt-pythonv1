#pragma once
#include <string>
#include <map>
#include "classification_interface.h"
using namespace std;
typedef void* (*register_func)();

class Store
{
public:
    static Store* getInstance();
    void* findInstance(const string& class_name);
    void registerInstance(const string& class_name, register_func func);
    void get_keys();

    classify_config FindClassifyConfig(const string& config_name);
    void RegisterClassifyConfig(const string& config_name, classify_config config);
    void GetClassifyConfigKeys();
    
private:
    map<string, register_func> m_register;
    map<string, classify_config> m_register_classify;
    static Store* instance;
    
};



