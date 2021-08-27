#include <vector>
#include <algorithm>
#include <functional>
#include "classification_interface.h"
class ClassificationUtils
{
public:
    bool TopNums(std::vector<std::vector<std::pair<float, int>>> &class_index, int topnum, std::vector<image_info>& outputinfo);

};