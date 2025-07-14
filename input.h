#pragma once
#include <fstream>
#include <sstream>
#include <utility>
#include <string>
#include <tuple>
#include <cctype>
#include <iomanip>
#include <iostream>
#include "HostStructs.h"

static inline void ltrim(std::string& s)
{
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    s.erase(0, i);
}

static bool getlineValid(std::ifstream& fin, std::string& line)
{
    while (std::getline(fin, line))
    {
        ltrim(line);
        if (line.empty() || line[0] == '#') continue;
        return true;
    }
    return false;
}

static int countValidNumbersStrict(const std::string& line)
{
    std::istringstream iss(line);
    std::string token;
    int count = 0;
    while (iss >> token)
    {
        std::istringstream tokStream(token);
        double d;
        char c;
        if (tokStream >> d && !(tokStream >> c))
        {
            ++count;
        }
    }
    return count;
}

void loadTriangleWallInfo(const std::string& file, HostTriangleWall& TW);

void loadContactParameterInfo(const std::string& file, HostContactParameter& CP);
