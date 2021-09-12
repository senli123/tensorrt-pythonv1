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

#define REGISTER_DETE_CONFIG(detection_config_name) \
    class detection_config_name##Register \
    { \
    public: \
        static detection_config newInstance() \
        { \
            return detection_config_name; \
        } \
    private: \
        static const RegisterDetectionConfig reg; \
    }; \
const RegisterDetectionConfig detection_config_name##Register::reg(#detection_config_name, detection_config_name##Register::newInstance());

#define REGISTER_SEG_CONFIG(segmentation_config_name) \
    class segmentation_config_name##Register \
    { \
    public: \
        static segmentation_config newInstance() \
        { \
            return segmentation_config_name; \
        } \
    private: \
        static const RegisterSegmentationConfig reg; \
    }; \
const RegisterSegmentationConfig segmentation_config_name##Register::reg(#segmentation_config_name, segmentation_config_name##Register::newInstance());

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
	RegisterDetectionConfig(string detection_config_name, detection_config config);
	~RegisterDetectionConfig();
};
class RegisterSegmentationConfig
{
public:
	RegisterSegmentationConfig(string segmentation_config_name, segmentation_config config);
	~RegisterSegmentationConfig();
};


