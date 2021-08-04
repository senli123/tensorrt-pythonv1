#pragma once
#include "alglogger.h"
#include <string>
bool CreateDirRecursively(const std::string &directory);
bool InitLogger(const char* logPrefix );
void UnitLogger();