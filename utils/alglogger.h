
#pragma once
#ifndef GLOG_NO_ABBREVIATED_SEVERITIES
#define GLOG_NO_ABBREVIATED_SEVERITIES
#endif // !GLOG_NO_ABBREVIATED_SEVERITIES

#ifndef GOOGLE_GLOG_DLL_DECL
#define GOOGLE_GLOG_DLL_DECL
#endif // !GOOGLE_GLOG_DLL_DECL

#include <glog/logging.h>

const unsigned int N_LOG_BUF_LEN = 2048;

#ifndef PRINT_LOG
#define PRINT_LOG(format, ...) {char szlog[N_LOG_BUF_LEN] = {0}; snprintf(szlog,N_LOG_BUF_LEN, format, __VA_ARGS__); LOG(INFO)<<szlog;}
#endif //PRINT_LOG

#ifndef PRINT_DEBUG
#define PRINT_DEBUG(format, ...) {char szlog[N_LOG_BUF_LEN] = {0}; snprintf(szlog,N_LOG_BUF_LEN, format, __VA_ARGS__); LOG(INFO)<<szlog;}
#endif //PRINT_DEBUG

#ifndef PRINT_WARN
#define PRINT_WARN(format, ...) {char szlog[N_LOG_BUF_LEN] = {0};snprintf(szlog,N_LOG_BUF_LEN, format, __VA_ARGS__); LOG(WARNING)<<szlog; }
#endif //PRINT_WARN

#ifndef PRINT_ERROR
#define PRINT_ERROR(format, ...) {char szlog[N_LOG_BUF_LEN] = {0};snprintf(szlog,N_LOG_BUF_LEN, format, __VA_ARGS__); LOG(ERROR)<<szlog;}
#endif //PRINT_WARN

#ifndef PRINT_FATAL
#define PRINT_FATAL(format, ...) {char szlog[N_LOG_BUF_LEN] = {0};s nprintf(szlog,N_LOG_BUF_LEN, format, __VA_ARGS__); LOG(FATAL)<<szlog;}
#endif //PRINT_FATAL	

#ifndef ALG_LOGGER_LOG
#define ALG_LOGGER_LOG  PRINT_LOG
#endif //ALG_LOGGER_LOG

#ifndef ALG_LOGGER_DEBUG
#define ALG_LOGGER_DEBUG PRINT_DEBUG
#endif //ALG_LOGGER_DEBUG

#ifndef ALG_LOGGER_WARN
#define ALG_LOGGER_WARN PRINT_WARN
#endif //ALG_LOGGER_WARN

#ifndef ALG_LOGGER_ERROR
#define ALG_LOGGER_ERROR PRINT_ERROR
#endif //ALG_LOGGER_ERROR

#ifndef ALG_LOGGER_FATAL
#define ALG_LOGGER_FATAL PRINT_FATAL
#endif //ALG_LOGGER_FATAL





