#include "config_operator.h"
#include <fstream>
#include <iostream>
#include <algorithm>

#ifdef  _WIN32
#include <Windows.h>
#else

#endif //  _WIN32

ConfigOperator::ConfigOperator()
{

}

long ConfigOperator::init()
{
    m_configInfo.clear();
    return readfile(getRunningPath() + "/config.ini");
}

std::string ConfigOperator::getValue(const std::string& section, const std::string& key)
{
    std::map<CFG_SECTION, CFG_ITEMS>::iterator isection = m_configInfo.find(section);
    if (m_configInfo.end() != isection)
    {
        CFG_ITEMS::iterator ikey = isection->second.find(key);
        if (ikey != isection->second.end())
        {
            return ikey->second;
        }
    }
    return "";
}
std::map<CFG_SECTION,CFG_ITEMS>& ConfigOperator::getCfginfo()
{
    return m_configInfo;
}

long ConfigOperator::readfile(const std::string& filename)
{
    std::string strcfg = "";// getFullWKP();
    strcfg.append(filename.c_str());
    std::ifstream infile(strcfg.c_str());
    if (!infile)
    {
        return -1;
    }

    std::string line,section;
    while (getline(infile, line))
    {
        if(1 != analysisline(line,section))
        {
            //break;
        }
    }
    infile.close();
    return 1;
}

long ConfigOperator::analysisline(std::string line,std::string& section)
{
    if (line.empty())
    {
        return -1;
    }
    // remove last \\r
    if (line.at(line.size() - 1) == '\r')
    {
        line = line.substr(0, line.size() - 1);
    }

    //std::transform(line.begin(), line.end(),line.begin(), ::tolower);
    std::string::size_type nsta = line.find_first_of(CFG_SECTION_STA);
    if(std::string::npos != nsta)
    {
        //section
        std::string::size_type nend = line.find_first_of(CFG_SECTION_END);
        if(nend == std::string::npos)
        {
            return -1;
        }
        else
        {
            section = line.substr(nsta+1,nend - nsta-1);
            return 1;
        }
    }
    else
    {
        std::string::size_type nbreak = line.find_first_of(CFG_ITEM_BREAK);
        if(std::string::npos == nbreak)
        {
            return -1;
        }
        else
        {
            std::string fs = line.substr(0, nbreak);
            std::string ss = line.substr(nbreak + 1, line.size() - nbreak);

            fs = removeBlankSpace(fs);
            ss = removeBlankSpace(ss);

            m_configInfo[section].insert(std::make_pair(fs,ss));
            return 1;
        }
    }
}

std::string ConfigOperator::removeBlankSpace(std::string& s)
{
    if (s.empty())
    {
        return s;
    }
    s.erase(0, s.find_first_not_of("\""));
    s.erase(s.find_last_not_of("\"") + 1);
    //s.substr(s.find_first_not_of(' '), s.find_last_not_of(' ') + 1);
    s.erase(0, s.find_first_not_of(" "));
    s.erase(s.find_last_not_of(" ") + 1);
    return s;
}

std::string  ConfigOperator::getRunningPath()
{
#if defined(_WIN32)
    char szFilePath[MAX_PATH + 1] = { 0 };
    GetModuleFileNameA(NULL, szFilePath, MAX_PATH);
    (strrchr(szFilePath, '\\'))[0] = 0;
    std::string path = szFilePath;
    return path;
#else
    char *buffer;
    buffer = getcwd(NULL, 0);
    std::string directory;
    directory = buffer;
    return directory;
#endif
}
