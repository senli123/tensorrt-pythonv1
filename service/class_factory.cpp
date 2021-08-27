#include "class_factory.h"
Store* Store::instance = nullptr;

Store* Store::getInstance()
{
    if (!instance) instance = new Store;
    return instance;
}

void* Store::findInstance(const string& class_name)
{
// for (map<string, register_func>::iterator it = m_register.begin(); it != m_register.end(); ++it)
// {
// 	std::cout<<it->first<<std::endl;
// }
    map < string, register_func>::iterator it = m_register.find(class_name);
    if (it == m_register.end())
    {
        return nullptr;
    }
    else
    {
        return it->second();
    }
}
void Store::registerInstance(const string& class_name, register_func func)
{
    m_register[class_name] = func;
}
void Store::get_keys()
{
    for (map<string, register_func>::iterator it = m_register.begin(); it != m_register.end(); ++it)
    {
        std::cout<<it->first<<", ";
    }
    std::cout<<std::endl;
}

classify_config Store::FindClassifyConfig(const string& config_name)
{
    map <string,classify_config>::iterator it = m_register_classify.find(config_name);
    if (it == m_register_classify.end())
    {
        classify_config temp_config;
        temp_config.input_size = 0;
        return temp_config;
    }	
    else {
        return it->second;
    }
}
void Store::RegisterClassifyConfig(const string& config_name, classify_config config)
{
    m_register_classify[config_name] = config;
}
void Store::GetClassifyConfigKeys()
{
    for (map<string, classify_config>::iterator it = m_register_classify.begin(); it != m_register_classify.end(); ++it)
    {
        std::cout<<it->first<<", ";
    }
    std::cout<<std::endl;

}