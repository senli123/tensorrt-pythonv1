#pragma once
#include<chrono>
#include <string>
#include <vector>
#include "my_log.h"
#include<iostream>
#include<fstream>
#include<iomanip>
#define START "start"
#define END "end"
#define PREPROCESS "preprocess"
#define INFER "infer"
#define POSTPROCESS "postprocess"
class Debug_utils
{
    
public:
    bool init(std::string name);
    bool save_time_path();
	static void set_time(std::string stage);
	static void cal_time();
	void mean_time();
private:
	static std::chrono::steady_clock::time_point startPoint;
	static std::chrono::steady_clock::time_point nowPoint;
	static int batch_num;
	static int all_pipeline_time;
	static int all_preprocess_time;
	static int all_infer_time;
	static int all_postprocess_time;
	static int batch_pipeline_time;
	static int batch_preprocess_time;
	static int batch_infer_time;
	static int batch_postprocess_time;
    std::string time_path;
    std::string output_path;

};
