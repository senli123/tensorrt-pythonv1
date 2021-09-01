#pragma once
#include "class_factory.h"
#include "classification_interface.h"
#include "detection_interface.h"
using namespace std;

//-----------------------模型--------------------------------------
class ClassifierFactory
{
public:
  static ClassifierInterface* createClassifierInterface(const string& name)
  {
    return (ClassifierInterface*)Store::getInstance()->findInstance(name);
  }
};

class DetectionFactory
{
public:
  static DetectionInterface* createDetectionInterface(const string& name)
  {
    return (DetectionInterface*)Store::getInstance()->findInstance(name);
  }
};
//--------------------配置---------------------------------
class ClassifyConfigFactory
{
public:
	static classify_config createStruct(const string& name)
	{
		return (classify_config)Store::getInstance()->FindClassifyConfig(name);
	}
};

class DetectionConfigFactory
{
public:
	static detection_config createStruct(const string& name)
	{
		return (detection_config)Store::getInstance()->FindDetectionConfig(name);
	}
};