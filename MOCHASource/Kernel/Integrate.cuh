#pragma once
#include "CalculateForceTorque.cuh"

__global__ void integrateBeforeContactCalculation(DynamicState state, double3 gravity, double timeStep, int num);

__global__ void integrateAfterContactCalculation(DynamicState state, double3 gravity, double timeStep, int num);

__global__ void correctWCSPHMotion(Sphere sph, SPH SPHP, double timeStep);

__global__ void updatePebblesMotionBeforeContact(Clump clump, Sphere sph, double timeStep);

void integrateBeforeContact(DeviceData& d, double3 gravity, double timeStep, int maxThreadsPerBlock);

void integrateAfterContact(DeviceData& d, double3 gravity, double timeStep, int maxThreadsPerBlock);
