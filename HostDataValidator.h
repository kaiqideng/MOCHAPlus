#pragma once
#include <cmath>
#include <iostream>
#include <string>
#include "HostStructs.h"

static bool isFinite(double x) { return std::isfinite(x); }

static bool isFinite3(const double3& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

static bool fail(const std::string& where, int idx = -1)
{
    if (idx >= 0)
        std::cerr << "Error: [" << where << "] invalid value at index "
        << idx << '\n';
    else
        std::cerr << "Error: [" << where << "] length mismatch\n";
    return false;
}

static bool checkDirectional(const HostDirectionalTerms& d,
    int expectedLen,
    const std::string& tag)
{
    if ((int)d.normal.size() != expectedLen ||
        (int)d.sliding.size() != expectedLen ||
        (int)d.rolling.size() != expectedLen ||
        (int)d.torsion.size() != expectedLen)
        return fail(tag + ".directional size");

    for (int i = 0; i < expectedLen; ++i)
    {
        if (!(d.normal[i] >= 0 && isFinite(d.normal[i])))  return fail(tag + ".normal", i);
        if (!(d.sliding[i] >= 0 && isFinite(d.sliding[i]))) return fail(tag + ".sliding", i);
        if (!(d.rolling[i] >= 0 && isFinite(d.rolling[i]))) return fail(tag + ".rolling", i);
        if (!(d.torsion[i] >= 0 && isFinite(d.torsion[i]))) return fail(tag + ".torsion", i);
    }
    return true;
}

bool validateContactParameter(const HostContactParameter& cp);

bool validateSpatialGrid(const HostSpatialGrid& g);

bool validateSphereData(const HostSphere& sph, double3 minBound, double3 maxBound, const int numClump, const int numSPH, int numMaterial);

bool validateSPHData(const HostSPH& SPHP);

bool validateClumpData(const HostClump& clumps, int numSphere);

bool validateTriangleWall(const HostTriangleWall& w, int numMaterial);

bool validateSimulationParameter(const HostSimulationParameter& p, int numMaterial);

bool validateHostData(const HostData& h);