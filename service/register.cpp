#include "register.h"
Register::Register(string class_name, register_func func)
{
  Store::getInstance()->registerInstance(class_name, func);
}

Register::~Register() {}


RegisterClassifyConfig::RegisterClassifyConfig(string classify_config_name, classify_config config)
{
  Store::getInstance()->RegisterClassifyConfig(classify_config_name, config);
}
RegisterClassifyConfig::~RegisterClassifyConfig(){}