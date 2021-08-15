#include <vector>
#include <algorithm>
#include <functional>
typedef struct tag_image_info
{
	std::vector<int> indexs;
	std::vector<float> scores;    
} image_info;

class ClassificationUtils
{
public:
    bool TopNums(std::vector<std::vector<std::pair<float, int>>> &class_index, int topnum, std::vector<image_info>& outputinfo);

};