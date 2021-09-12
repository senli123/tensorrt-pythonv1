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

detection_config Store::FindDetectionConfig(const string& config_name)
{
    map <string,detection_config>::iterator it = m_register_detection.find(config_name);
    if (it == m_register_detection.end())
    {
        detection_config temp_config;
        temp_config.input_size = 0;
        return temp_config;
    }	
    else {
        return it->second;
    }
}
void Store::RegisterDetectionConfig(const string& config_name, detection_config config)
{
    m_register_detection[config_name] = config;
}
void Store::GetDetectionConfigKeys()
{
    for (map<string, detection_config>::iterator it = m_register_detection.begin(); it != m_register_detection.end(); ++it)
    {
        std::cout<<it->first<<", ";
    }
    std::cout<<std::endl;

}
segmentation_config Store::FindSegmentationConfig(const string& config_name)
{
   map <string,segmentation_config>::iterator it = m_register_segmentation.find(config_name);
    if (it == m_register_segmentation.end())
    {
        segmentation_config temp_config;
        temp_config.input_size = 0;
        return temp_config;
    }	
    else {
        return it->second;
    }
}
void Store::RegisterSegmentationConfig(const string& config_name, segmentation_config config)
{
  m_register_segmentation[config_name] = config;
}
void Store::GetSegmentationConfigKeys()
{
  for (map<string, segmentation_config>::iterator it = m_register_segmentation.begin(); it != m_register_segmentation.end(); ++it)
    {
        std::cout<<it->first<<", ";
    }
    std::cout<<std::endl;
}