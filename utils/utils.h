#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "detection_interface.h"
class Utils
{
public:
    ~Utils(){}
    Utils(const Utils&)=delete;
    Utils& operator=(const Utils&)=delete;
    static Utils& get_instance(){
        static Utils instance;
        return instance;

    }
private:
    Utils(){};
public:
    std::vector<std::string> split(std::string str, std::string pattern);
    std::vector<char*> split_name(char input_names[],const char split[]);
    void printBbox(cv::Mat img,std::vector<InstanceInfo> instances, std::string path);

};