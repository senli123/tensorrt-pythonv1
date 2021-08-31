#pragma once
#include "class_factory.h"
#define REGISTER_CLASS(class_name) \
    class class_name##Register \
    { \
    public: \
        static void* newInstance() \
        { \
            return new class_name; \
        } \
    private: \
        static const Register reg; \
    }; \
const Register class_name##Register::reg(#class_name, class_name##Register::newInstance);

#define REGISTER_CLA_CONFIG(classify_config_name) \
    class classify_config_name##Register \
    { \
    public: \
        static classify_config newInstance() \
        { \
            return classify_config_name; \
        } \
    private: \
        static const RegisterClassifyConfig reg; \
    }; \
const RegisterClassifyConfig classify_config_name##Register::reg(#classify_config_name, classify_config_name##Register::newInstance());



class Register
{
public:
    Register(string class_name, register_func func);
    ~Register();
};

class RegisterClassifyConfig
{
public:
	RegisterClassifyConfig(string classify_config_name, classify_config config);
	~RegisterClassifyConfig();
};

class RegisterDetectionConfig
{
public:
	RegisterDetectionConfig(string classify_config_name, detection_config config);
	~RegisterDetectionConfig();
};


