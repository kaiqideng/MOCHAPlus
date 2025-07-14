#pragma once
#include "input.h"

#ifdef _WIN32
#include <io.h>
#include <direct.h>                 // _mkdir

#define MKDIR(path) _mkdir(path)    // returns 0 on success, -1 if already exists

static int removeVtuFiles(const std::string& dir)
{
    std::string pattern = dir + "\\*.vtu";
    struct _finddata_t fdata;
    intptr_t h = _findfirst(pattern.c_str(), &fdata);
    if (h == -1) return 0;

    int removed = 0;
    do {
        std::string full = dir + "\\" + fdata.name;
        if (std::remove(full.c_str()) == 0) ++removed;
    } while (_findnext(h, &fdata) == 0);
    _findclose(h);
    return removed;
}

static int removeDatFiles(const std::string& dir)
{
    std::string pattern = dir + "\\*.dat";
    struct _finddata_t fdata;
    intptr_t h = _findfirst(pattern.c_str(), &fdata);
    if (h == -1) return 0;

    int removed = 0;
    do {
        std::string full = dir + "\\" + fdata.name;
        if (std::remove(full.c_str()) == 0) ++removed;
    } while (_findnext(h, &fdata) == 0);
    _findclose(h);
    return removed;
}

#else
#include <dirent.h>
#include <sys/stat.h>               // mkdir
#include <cstring>

#define MKDIR(path) mkdir(path, 0755)

static bool hasVtuExt(const char* fname)
{
    const char* dot = strrchr(fname, '.');
    return dot && std::strcmp(dot, ".vtu") == 0;
}

static bool hasDatExt(const char* fname)
{
    const char* dot = strrchr(fname, '.');
    return dot && std::strcmp(dot, ".dat") == 0;
}

static int removeVtuFiles(const std::string& dir)
{
    DIR* dp = opendir(dir.c_str());
    if (!dp) return 0;

    int removed = 0;
    struct dirent* ent;
    while ((ent = readdir(dp)) != NULL)
    {
        if (ent->d_type == DT_DIR) continue;
        if (!hasVtuExt(ent->d_name)) continue;

        std::string full = dir + "/" + ent->d_name;
        if (std::remove(full.c_str()) == 0) ++removed;
    }
    closedir(dp);
    return removed;
}

static int removeDatFiles(const std::string& dir)
{
    DIR* dp = opendir(dir.c_str());
    if (!dp) return 0;

    int removed = 0;
    struct dirent* ent;
    while ((ent = readdir(dp)) != NULL)
    {
        if (ent->d_type == DT_DIR) continue;
        if (!hasDatExt(ent->d_name)) continue;

        std::string full = dir + "/" + ent->d_name;
        if (std::remove(full.c_str()) == 0) ++removed;
    }
    closedir(dp);
    return removed;
}
#endif

void writeSpheresVTU(const std::string& fileName, const HostSphere& s, int frame, double time, int step);

void writeBasicInteractionsVTU(const std::string& fileName, const HostBasicInteraction& inter, int frame, double time, int step);

void writeBondedInteractionsVTU(const std::string& fileName, const HostBondedInteraction& inter, int frame, double time, int step);

void writeTriangleWallVTU(const std::string& fileName, const HostTriangleVertex& vertices, const HostTriangleFace& faces, const HostDynamicState& wallState, int frame, double time, int step);

void writeTriangleWallPressureVTU(const std::string& fileName, const HostBasicInteraction& faceI, const HostBasicInteraction& edgeI, const HostBasicInteraction& vertexI, const HostTriangleWall& TW, int frame, double time, int step);

void writeSPHSpheresVTU(const std::string& fileName, const HostSPH& SPHP, const HostSphere& s, int frame, double time, int step);

void writeSolidSpheresVTU(const std::string& fileName, const HostSphere& s, int frame, double time, int step);

void writeBoxSurfaceVTU(const std::string& fileName, const double3& minCorner, const double3& maxCorner);

void writeHostDynamicStateToDat(const HostDynamicState& state, const std::string& filename, double time);
