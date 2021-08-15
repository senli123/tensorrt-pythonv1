#include "classification_utils.h"
bool ClassificationUtils::TopNums(std::vector<std::vector<std::pair<float, int>>> &class_index, int topnum, std::vector<image_info>& outputinfo)
{
    for (int batch = 0; batch < class_index.size(); batch++)
    {
        std::vector<std::pair<float, int>> one_batch_class_index = class_index[batch];
        std::partial_sort(one_batch_class_index.begin(), one_batch_class_index.begin() + topnum, one_batch_class_index.end(),
            std::greater<std::pair<float, int>>());
        image_info info;
        for (int i = 0; i < topnum; i++)
        {
            info.indexs.push_back(one_batch_class_index[i].second);
            info.scores.push_back(one_batch_class_index[i].first);
        }
        outputinfo.push_back(info);
    }
    
    
}
