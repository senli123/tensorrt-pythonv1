#pragma once
#include "class_factory.h"
using namespace std;

//-----------------------模型--------------------------------------
//分类
class ClassifierFactory
{
public:
  static ClassifierInterface* createClassifierInterface(const string& name)
  {
    return (ClassifierInterface*)Store::getInstance()->findInstance(name);
  }
};
//检测
class DetectionFactory
{
public:
  static DetectionInterface* createDetectionInterface(const string& name)
  {
    return (DetectionInterface*)Store::getInstance()->findInstance(name);
  }
};
//分割
class SegmentationFactory
{
public:
  static SegmentationInterface* createSegmentationInterface(const string& name)
  {
    return (SegmentationInterface*)Store::getInstance()->findInstance(name);
  }
};
//--------------------配置---------------------------------
//分类
class ClassifyConfigFactory
{
public:
	static classify_config createStruct(const string& name)
	{
		return (classify_config)Store::getInstance()->FindClassifyConfig(name);
	}
};
//检测
class DetectionConfigFactory
{
public:
	static detection_config createStruct(const string& name)
	{
		return (detection_config)Store::getInstance()->FindDetectionConfig(name);
	}
};
//分割
class SegmentationConfigFactory
{
public:
	static segmentation_config createStruct(const string& name)
	{
		return (segmentation_config)Store::getInstance()->FindSegmentationConfig(name);
	}
};