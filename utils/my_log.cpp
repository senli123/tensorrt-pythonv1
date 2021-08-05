#include "my_log.h"
#include <sys/types.h>  
#include <sys/stat.h>  
#include <unistd.h>
bool CreateDirRecursively(std::string &directory, char* flag)
{
    char *buffer;
    buffer = getcwd(NULL, 0);
    directory = buffer;
    std::string s_time = "";
    char psDate[128] = { 0 };
    time_t nSeconds;
    struct tm * pTM;
    
    time(&nSeconds); // 同 nSeconds = time(NULL);
    pTM = localtime(&nSeconds);
    
    /* 系统日期,格式:YYYMMDD */
    sprintf(psDate,"%04d-%02d-%02d", 
            pTM->tm_year + 1900, pTM->tm_mon + 1, pTM->tm_mday);
    s_time.append(psDate);
    directory.append(flag);
    if (access(directory.c_str(),F_OK)!=0)
    {
        if(mkdir(directory.c_str(), S_IRUSR|S_IRGRP|S_IROTH|S_IWUSR|S_IWGRP|S_IWOTH) == -1)
        {
            printf("mkdir(%s) failed(%s)\n",directory.c_str(), strerror(errno));
            return false;
        }
    }
	directory += s_time;
	directory.append("/");
    if (access(directory.c_str(),F_OK)!=0)
    {
        if(mkdir(directory.c_str(), S_IRUSR|S_IRGRP|S_IROTH|S_IWUSR|S_IWGRP|S_IWOTH) == -1)
        {
            printf("mkdir(%s) failed(%s)\n",directory.c_str(), strerror(errno));
            return false;
        }
    }
    return true;
}
bool InitLogger(const char* logPrefix )
{
    //创建文件夹返回路径
    std::string path;

    if(!CreateDirRecursively(path,"/log/")){
        printf("mkdir log dir failed !\n");
        return false;
    }
    //初始化log文件
    google::InitGoogleLogging((const char *)(__FILE__));
	google::SetStderrLogging(google::GLOG_INFO);
	google::SetLogFilenameExtension("log_");
    
	//google::FlushLogFiles(google::GLOG_INFO);

	//google::InstallFailureSignalHandler();
	//google::InstallFailureWriter(&SignalHandle);

	google::SetLogDestination(google::GLOG_INFO, (path + logPrefix + "_info_").c_str());
	google::SetLogDestination(google::GLOG_WARNING, (path + logPrefix + "_warn_").c_str());
	google::SetLogDestination(google::GLOG_ERROR, (path + logPrefix + "_error_").c_str());
	google::SetLogDestination(google::GLOG_FATAL, (path + logPrefix + "_fatal_").c_str());
	FLAGS_max_log_size = 10;
    FLAGS_logbufsecs = 0;
	FLAGS_colorlogtostderr = true;
    return true;
}
void UnitLogger()
{
    google::ShutdownGoogleLogging();
}