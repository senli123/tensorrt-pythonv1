#pragma once
#include "class_factory.h"
#include "classification_interface.h"
using namespace std;
class ClassifierFactory
{
public:
  static ClassifierInterface* createClassifierInterface(const string& name)
  {
    return (ClassifierInterface*)Store::getInstance()->findInstance(name);
  }
};
// class DeteFactory
// {
// public:
//   static Detection* createDetection(const string& name)
//   {
//     return (Detection*)Store::getInstance()->findInstance(name);
//   }
// };
class ClassifyConfigFactory
{
public:
	static classify_config createStruct(const string& name)
	{
		return (classify_config)Store::getInstance()->FindClassifyConfig(name);
	}
};