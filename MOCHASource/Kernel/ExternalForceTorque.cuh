#pragma once
#include "CalculateContactForceTorque.cuh"

__global__ void calHydroForce(Sphere sph, double3 currentVel, double waterDensity, double waterLevel0, double Cd);

__global__ void calGlobalDampingForceTorque(Sphere sph, double globalDampingCoff);

void calculateHydroForce(Sphere sph, double3 currentVel, double waterDensity, double waterLevel0, double Cd, int maxThreadsPerBlock);

void calculateGlobalDampingForceTorque(Sphere sph, double globalDampingCoff, int maxThreadsPerBlock);
