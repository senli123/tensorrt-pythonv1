#pragma once
#include <map>
#include <string>
#include <sys/types.h>  
#include <sys/stat.h>  
#include <unistd.h>


const std::string CFG_SECTION_STA = "[";
const std::string CFG_SECTION_END = "]";
const std::string CFG_ITEM_BREAK = "=";


const std::string CFG_CLASSIFICATION = "classification";
const std::string CFG_DETECTION = "detection";
const std::string CFG_SEGMENTATION = "segmentation";
const std::string CFG_ONNX_PATH = "_onnx_path";
const std::string CFG_BIN_PATH = "_bin_path";
const std::string CFG_INPUT_NAME = "_input_name";
const std::string CFG_OUTPUT_NAME = "_output_name";
const std::string CFG_CUDA_ID = "_cuda_id";
const std::string CFG_INPUT_DIM = "_input_dim";
const std::string CFG_OUTPUT_DIM = "_output_dim";
const std::string CFG_INPUT_SIZE = "_input_size";
const std::string CFG_INPUT_W = "_input_h";
const std::string CFG_INPUT_H = "_input_w";
const std::string CFG_BATCH_SIZE = "_batch_size";
const std::string CFG_MEAMVALS = "_meanVals"; 
const std::string CFG_NORMVALS = "_normVals";
const std::string CFG_CLASS_NUM = "_class_num";
const std::string CFG_TOP_NUM = "_top_num";
const std::string CFG_FP16 = "_fp16";
const std::string CFG_INT8 = "_int8";

const std::string CFG_ITEM_NUM = "_item_num";
const std::string CFG_CONFTHRE = "_confthre";
const std::string CFG_IOUTHRE = "_iouthre";
const std::string CFG_NET_GRID = "_net_grid";
const std::string CFG_ANCHOR_NUM = "_anchor_num";

typedef std::map<std::string,std::string> CFG_ITEMS;
typedef std::string CFG_SECTION;
typedef std::map<CFG_SECTION,CFG_ITEMS> MAP_CFGITEMS;

class ConfigOperator
{
private:
     ConfigOperator();

public:
     static ConfigOperator& getIns()
     {
         static ConfigOperator ins;
         return ins;
     }

public:
      std::string getValue(const std::string& section, const std::string& key);
      std::map<CFG_SECTION,CFG_ITEMS>& getCfginfo();
public:
    long init();
    static std::string  getRunningPath();

private:
    long readfile(const std::string& filename);
    long analysisline(std::string line,std::string& section);
    std::string removeBlankSpace(std::string& s);

private:
    std::map<CFG_SECTION,CFG_ITEMS> m_configInfo;
};

