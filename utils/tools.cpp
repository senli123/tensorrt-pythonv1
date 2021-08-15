#include "tools.h"
std::chrono::steady_clock::time_point Debug_utils::startPoint = std::chrono::steady_clock::now();
std::chrono::steady_clock::time_point Debug_utils::nowPoint = std::chrono::steady_clock::now();
int Debug_utils::batch_num =0;
int Debug_utils::all_pipeline_time=0;
int Debug_utils::all_preprocess_time=0;
int Debug_utils::all_infer_time=0;
int Debug_utils::all_postprocess_time=0;
int Debug_utils::batch_pipeline_time=0;
int Debug_utils::batch_preprocess_time=0;
int Debug_utils::batch_infer_time=0;
int Debug_utils::batch_postprocess_time=0;
bool Debug_utils::init(std::string name)
{
    std::string directory;
    if(!CreateDirRecursively(directory, "/debug/"))
    {
        LOG(ERROR)<<"create deubg directory failed!";
        return false;
    }
    time_path = directory + name + "_time.txt";
    //写入表头
    std::ofstream outfile(time_path, std::ofstream::app);
    outfile<<"batch_index"<<std::setw(5)<<"preprocess_time(ms)"<<std::setw(5)<<"infer_time(ms)"
    <<std::setw(5)<<"postprocess_time(ms)"<<std::setw(5)<<"pipeline_time(ms)"<<std::endl;
	outfile.close();  
}
void Debug_utils::set_time(std::string stage)
{
	if (stage == START)
	{
		startPoint = std::chrono::steady_clock::now();
	}
	else if (stage == PREPROCESS)
	{
		std::chrono::steady_clock::time_point prePoint = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(prePoint - startPoint);
		batch_preprocess_time = duration.count();
		nowPoint = prePoint;
        std::ostringstream ostr;
		ostr << "preprocess cost time: " << batch_preprocess_time <<"ms";
		LOG(INFO)<<ostr.str().c_str();
	}
	else if (stage == INFER)
	{
		std::chrono::steady_clock::time_point inferPoint = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(inferPoint - nowPoint);
		batch_infer_time = duration.count();
		nowPoint = inferPoint;
        std::ostringstream ostr;
		ostr << "infer cost time: " << batch_infer_time <<"ms";
		LOG(INFO)<<ostr.str().c_str();
	}
	else if (stage == POSTPROCESS)
	{
		std::chrono::steady_clock::time_point postPoint = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(postPoint - nowPoint);
		batch_postprocess_time = duration.count();
		nowPoint = postPoint;
        std::ostringstream ostr;
		ostr << "postprocess cost time: " << batch_postprocess_time <<"ms";
		LOG(INFO)<<ostr.str().c_str();
	}
	else if (stage == END)
	{
		std::chrono::steady_clock::time_point endPoint = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endPoint - startPoint);
		batch_pipeline_time = duration.count();
        std::ostringstream ostr;
		ostr << "pipeline cost time: " << batch_pipeline_time <<"ms";
		LOG(INFO)<<ostr.str().c_str();
		cal_time();
		
	}
}
bool Debug_utils::save_time_path()
{
    //写入信息
    std::ofstream outfile(time_path, std::ofstream::app);
    outfile<<batch_num<<std::setw(5)<<batch_preprocess_time<<std::setw(5)<<batch_infer_time
    <<std::setw(5)<<batch_postprocess_time<<std::setw(5)<<batch_pipeline_time<<std::endl;
	outfile.close();  

}
void Debug_utils::cal_time()
{
	all_pipeline_time += batch_pipeline_time;
	all_preprocess_time += batch_preprocess_time;
	all_infer_time += batch_infer_time;
	all_postprocess_time += batch_postprocess_time;
	batch_num += 1;
}
void Debug_utils::mean_time()
{
	int mean_pipeline_time = all_pipeline_time /batch_num;
	int mean_preprocess_time =  all_preprocess_time/batch_num;
	int mean_infer_time = all_infer_time/batch_num;
	int mean_postprocess_time = all_postprocess_time/batch_num;
     //写入信息
    std::ofstream outfile(time_path, std::ofstream::app);
    outfile<<"average"<<std::setw(5)<<mean_preprocess_time<<std::setw(5)<<mean_infer_time
    <<std::setw(5)<<mean_postprocess_time<<std::setw(5)<<mean_pipeline_time<<std::endl;
	outfile.close();  
	
}