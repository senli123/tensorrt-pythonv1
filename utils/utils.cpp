#include "utils.h"
std::vector<std::string> Utils::split(std::string str, std::string pattern)
{
    std::string::size_type pos; 
    std::vector<std::string> result;
    str += pattern;
    int size = str.size();
    for (int i = 0; i < size; i++)
    {
        pos = str.find(pattern, i);
        if (pos<size)
        {
            std::string s = str.substr(i, pos-i);
            result.push_back(s);
            i = pos + pattern.size()-1;
        }
        
    }
    return result;
}

std::vector<char*> Utils::split_name(char input_names[],const char split[])
{
    std::vector<char*> res_split;
    char* res = strtok(input_names, split);//image_name必须为char[] 
    while (res != NULL)
	{
		res_split.push_back(res);
		res = strtok(NULL, split);
	}
    return res_split;
}
void Utils::printBbox(cv::Mat img, std::vector<InstanceInfo> instances, std::string path)
{
    for(int idx = 0; idx<instances.size();idx++)
    {
        InstanceInfo instance= instances[idx];
        cv::rectangle(img, instance.rect, cv::Scalar(255, 0, 0), 2, cv::LINE_8, 0);
        //类别坐标和socre坐标
		cv::Point class_point, score_point;
        class_point.x = instance.rect.x - 10;
		class_point.y = instance.rect.y - 20;
		score_point.x = instance.rect.x - 10;
		score_point.y = instance.rect.y - 10;
        cv::putText(img, std::to_string(instance.class_id), class_point, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, 8, 0);
		cv::putText(img, std::to_string(instance.score), score_point, cv::FONT_HERSHEY_COMPLEX, 0.5,cv::Scalar(0, 0, 255), 1, 8, 0);
    }
    cv::imwrite(path,img);
}