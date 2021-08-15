#pragma once
#include "alglogger.h"
#include <string>
bool CreateDirRecursively(std::string &directory, char* flag);
bool InitLogger(const char* logPrefix );
void UnitLogger();